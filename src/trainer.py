""" Main trainer class for BabyLM. """

import copy
import logging
import os
import shutil
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub.hf_api import create_repo
from huggingface_hub.repository import Repository
from huggingface_hub.utils._errors import HfHubHTTPError
from omegaconf import OmegaConf

# Data loading
from torch.utils.data import DataLoader 
from tqdm import tqdm

# Model Training
from transformers import PreTrainedTokenizerFast, Trainer, TrainerCallback 
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_utils import (
    HubStrategy,
    IntervalStrategy,
    speed_metrics,
    PREFIX_CHECKPOINT_DIR,
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import (
    get_full_repo_name,
    is_torch_neuroncore_available,
)

# Model Loading
from src.models import load_base_model
from src.utils.data import base_collate_fn, POSLookup
from src.utils.inference import (
    compute_trainer_perplexity,
)

# typing imports
from .config import BabyLMConfig

# Model Evaluation
from .evaluators.blimp_evaluator import BlimpEvaluator
from .evaluators.finetune_evaluator import FinetuneEvaluator, collect_results
from .evaluators.blimp_bias_evaluator import BlimpBiasEvaluator
from .evaluators.perplexity_bias_evaluator import PerplexityBiasEvaluator

# Objective Curriculum
from .objective_task import load_objective_task


logger = logging.getLogger(__name__)

POSSIBLE_METRICS = ["blimp", "perplexity", "blimp_bias", "aoa", "glue", "msgs"]

class TaskTrainerCallback(TrainerCallback):
    """
    A TrainerCallback that handles updating the task heads of the model.
    """

    def __init__(self, objective_task) -> None:
        self.objective_task = objective_task

    def on_step_end(self, *args, **kwargs) -> None:
        self.objective_task.optimizer_step()

class CustomTrainer(Trainer):
    def __init__(
        self,
        hydra_config: BabyLMConfig,
        dry_run: bool,
        args: TrainingArguments,
        tokenizer: PreTrainedTokenizerFast,
        pos_lookup: Optional[POSLookup] = None,
        **kwargs,
    ) -> None:
        """
        We need to override the __init__ method to add the experiment group and experiment name.

        We use the group name and experiment name for version controlling/identifying the current
        run in, for example, huggingface, wandb ...

        Args:
            * hydra_config: (BabyLMConfig): The config object.
            * dry_run (bool): Whether the experiment is being run in dry run mode
            * args (TrainingArguments): The training arguments, unpacked from the kwargs dict
                in order to have access to possible arguments meant to be used in the Custom
                Trainer class.
            * tokenizer (PreTrainedTokenizerFast): The tokenizer used for the current run.
            * pos_lookup (Optional[POSLookup], *optional*, defaults to `None`): The POS lookup
                object used to convert POS tags to indices.
        """

        self.hydra_config = hydra_config
        self.dry_run = dry_run

        self.experiment_group = hydra_config.experiment.group
        self.experiment_name = hydra_config.experiment.name
        self.evaluation_metrics = hydra_config.trainer.evaluation_metrics                    

        super().__init__(args=args, **kwargs)

        model = kwargs.get("model", None)

        if model is None:
            raise ValueError(
                "The model needs to be passed in as a keyword argument to the Trainer"
            )
        
        for metric in self.evaluation_metrics:
            if metric not in POSSIBLE_METRICS:
                raise ValueError(
                    f"Invalid evaluation metric {metric}. Possible metrics are {POSSIBLE_METRICS}"
                )

        self.objective_task = load_objective_task(
            cfg=hydra_config,
            hidden_rep_size=model.config.hidden_size,
            tokenizer=tokenizer,
            device=self.args.device,
            local_rank=self.args.local_rank,
            base_model=model,
        )

        self.data_collator = self.objective_task.objective_collator

        self.tokenizer = tokenizer
        self.pos_lookup = pos_lookup

        self.add_callback(TaskTrainerCallback(self.objective_task))

    def init_git_repo(self, at_init: bool = False) -> None:
        """
        Initializes a git repo in `self.args.hub_model_id`.
        Args:
            at_init (`bool`, *optional*, defaults to `False`):
                Whether this function is called before any training or not. If `self.args.overwrite_output_dir` is
                `True` and `at_init` is `True`, the path to the repo (which is `self.args.output_dir`) might be wiped
                out.
        """
        if not self.is_world_process_zero():
            return
        if self.args.hub_model_id is None:
            repo_name = Path(self.args.output_dir).absolute().name
        else:
            repo_name = self.args.hub_model_id
        if "/" not in repo_name:
            repo_name = get_full_repo_name(
                repo_name, token=self.args.hub_token
            )

        # NOTE Fix huggingface_hub.utils._errors.HfHubHTTPError: 500 Server Error known issue
        _repo_sleep_time = 1
        _repo_created = False
        while not _repo_created:
            try:
                # Make sure the repo exists.
                create_repo(
                    repo_name,
                    token=self.args.hub_token,
                    private=self.args.hub_private_repo,
                    exist_ok=True,
                )
                _repo_created = True
            except HfHubHTTPError:
                if _repo_sleep_time > 64:
                    raise RuntimeError(
                        f"Could not create huggingface repo {repo_name} after {64} seconds."
                    )
                time.sleep(_repo_sleep_time)
                _repo_sleep_time *= 2

        assert self.args.hub_token is not None

        try:
            self.repo = Repository(
                self.args.output_dir,
                clone_from=repo_name,
                token=self.args.hub_token,
                revision=self.experiment_name,
            )
        except EnvironmentError:
            if self.args.overwrite_output_dir and at_init:
                # Try again after wiping output_dir
                shutil.rmtree(self.args.output_dir)
                self.repo = Repository(
                    self.args.output_dir,
                    clone_from=repo_name,
                    token=self.args.hub_token,
                    revision=self.experiment_name,
                )
            else:
                raise

        try:
            # the branch name should have been created already by the `create_repo` call
            self.repo.git_pull()
        except OSError:
            # if the repo is empty, the git_pull will fail
            pass

        # By default, ignore the checkpoint folders
        if (
            not os.path.exists(
                os.path.join(self.args.output_dir, ".gitignore")
            )
            and self.args.hub_strategy != HubStrategy.ALL_CHECKPOINTS
        ):
            with open(
                os.path.join(self.args.output_dir, ".gitignore"),
                "w",
                encoding="utf-8",
            ) as writer:
                writer.writelines(["checkpoint-*/"])

        self.push_in_progress = None

        config_output_path = os.path.join(
            self.args.output_dir, f"hydra_config_{time.time()}.yaml"
        )
        OmegaConf.save(self.hydra_config, config_output_path)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )

    def compute_loss(self, model, inputs, **kwargs):
        """
        We compute the loss for each objective unit, and then sum them up.
        """
        loss_metrics = {}

        if self.state.global_step >= self.args.max_steps:
            raise Exception(
                """
                Reached max_steps already - training should have stopped.
                NOTE: You are probably using a resume_from_checkpoint flag with max_steps set to a
                value smaller than the number of steps in the checkpoint.
                """
            )

        optional_kwargs = {}
        objective_task_name = self.objective_task.objective_task_name

        if objective_task_name == "pos_smoothing":
            # NOTE: We need to pass in the pos lookup to the POS SMOOTHING objective
            # as well as the global step in order to be able to use the temperature schedule
            optional_kwargs["pos_lookup"] = self.pos_lookup
            optional_kwargs["global_step"] = self.state.global_step
        rank_loss = self.objective_task.compute_loss(model, inputs, loss_kwargs=optional_kwargs)

        # averaging over the processes
        total_loss = self._nested_gather(rank_loss).mean().item()  # type: ignore
        loss_metrics[f"loss_{objective_task_name}"] = total_loss

        if (
            self.args.logging_strategy == IntervalStrategy.STEPS
            and self.state.global_step % self.args.logging_steps == 0
        ):

            self.log(loss_metrics)

        return rank_loss

    def evaluate(
        self,
        metric_key_prefix: str = "eval",
        **kwargs,
    ) -> Dict[str, float]:
        """
        Override the Trainer.evaluate() method to evaluate on BLIMP using the evaluation pipeline submodule.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        # NOTE: This code runs on all processes (i.e. multiple GPUs) in a distributed settings.

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        start_time = time.time()

        is_best_run = "best" in metric_key_prefix

        metrics = {}

        if 'perplexity' in self.evaluation_metrics:

            # Additional behavior - evaluate perplexity
            # Get 10_000 samples from the eval dataset
            eval_subset = self.eval_dataset.select(  # type: ignore
                range(
                    self.args.process_index,  # local process rank
                    self.eval_dataset.num_rows,  # type: ignore
                    self.eval_dataset.num_rows // ((100 if self.dry_run else 10_000) // self.args.world_size),  # type: ignore
                )
            )

            logging.info("Evaluating perplexity...")
            logging.info(f" ** Number of samples: {eval_subset.num_rows}")
            pad_idx = self.tokenizer.pad_token_id
            mask_idx = self.tokenizer.mask_token_id

            assert pad_idx is not None and mask_idx is not None

            perplexities = []
            per_token_losses = []

            with torch.no_grad():

                eval_batch_size = 4

                inference_dataloader = DataLoader(
                    eval_subset,  # type: ignore
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=base_collate_fn,
                    pin_memory=True,
                )

                for batch in tqdm(inference_dataloader):

                    batch_perplexity, per_token_loss = compute_trainer_perplexity(
                        batch, self.tokenizer, self, output_per_token_loss=True
                    )

                    perplexities.extend(batch_perplexity)
                    per_token_losses.extend(per_token_loss)

            # NOTE: a list of sample-level perplexities
            tensor_perplexities = torch.tensor(
                perplexities, device=self.args.device
            )

            # NOTE: a list of token-level losses stored as a list (list of list)
            tensor_per_token_losses = torch.tensor(
                per_token_losses, device=self.args.device
            )

            # NOTE: If there are multiple processes we need to group gather the perplexities and 
            # token_level_losses
            if self.args.world_size > 1:
                # setup barrier for all processes
                dist.barrier()

                gathered_perplexities = [
                    torch.zeros_like(tensor_perplexities)
                    for _ in range(self.args.world_size)
                ]

                gathered_per_token_losses = [
                    torch.zeros_like(tensor_per_token_losses)
                    for _ in range(self.args.world_size)
                ]

                dist.all_gather(gathered_perplexities, tensor_perplexities)
                dist.all_gather(gathered_per_token_losses, tensor_per_token_losses)

                # combine list of tensors into a single tensor

                gathered_perplexities = torch.cat(gathered_perplexities)
                gathered_per_token_losses = torch.cat(gathered_per_token_losses)
            else: 
                gathered_perplexities = tensor_perplexities
                gathered_per_token_losses = tensor_per_token_losses

            gathered_perplexity_mean = torch.mean(
                gathered_perplexities
            ).item()
            gathered_perplexity_std = torch.mean(
                gathered_perplexities
            ).item() 

            metrics["perplexity_mean"] = gathered_perplexity_mean
            metrics["perplexity_std"] = gathered_perplexity_std

            # On the main process we need to save out the perplexities to the output directory
            if self.is_world_process_zero():

                # create an output perplexity.json file 
                output_dir = os.path.join(self.args.output_dir, "lm_model")
                ppl_dir = os.path.join(output_dir, "perplexity")
                os.makedirs(ppl_dir, exist_ok=True)

                perplexity_output_path = os.path.join(
                    ppl_dir, "predictions.json"
                )

                perplexity_dict = {
                    "task": "perplexity",
                    "predictions": [],
                }

                for _process_idx in range(self.args.world_size):
                    _curr_eval_subset = self.eval_dataset.select(  # type: ignore
                        range(
                            _process_idx,  # local process rank
                            self.eval_dataset.num_rows,  # type: ignore
                            self.eval_dataset.num_rows // ((100 if self.dry_run else 10_000) // self.args.world_size),  # type: ignore
                        )
                    )


                    _curr_eval_subset_dataloader = DataLoader(
                        _curr_eval_subset,  # type: ignore
                        batch_size=1,
                        shuffle=False,
                        collate_fn=base_collate_fn,
                    )

                    for idx, batch in enumerate(_curr_eval_subset_dataloader):

                        _input_ids = batch["input_ids"].squeeze(0).tolist()
                        _perplexity = gathered_perplexities[
                            _process_idx * len(_curr_eval_subset_dataloader) + idx
                        ].item()
                        _per_token_loss = gathered_per_token_losses[
                            _process_idx * len(_curr_eval_subset_dataloader) + idx
                        ].tolist()

                        sample_dict = {
                            "id": "perplexity_" + str(_process_idx * len(_curr_eval_subset_dataloader) + idx),
                            "input_ids": _input_ids,
                            "perplexity": _perplexity, 
                            "per_token_loss": _per_token_loss,
                        }

                        perplexity_dict["predictions"].append(sample_dict)

                # save out perplexity_dict to perplexity_output_path
                with open(perplexity_output_path, "w") as f:
                    json.dump(perplexity_dict, f) 

                # Run perplexity bias evaluation
                logging.info("Evaluating perplexity bias...")
                perplexity_bias_evaluator = PerplexityBiasEvaluator(
                    perplexity_output_path,
                    tokenizer=self.tokenizer,
                    pos_lookup=self.pos_lookup,
                )
                perplexity_bias_metrics = perplexity_bias_evaluator()
                metrics.update(perplexity_bias_metrics)  # type: ignore

        self.save_model(self.args.output_dir, _internal_call=True)
        # if world size > 1, then we need to synchronize the model across all processes
        if self.args.world_size > 1:
            dist.barrier()  # Ensure all processes have access to the same model

        inference_model_dir = os.path.join(self.args.output_dir, "lm_model")

        # Additional behaviour - evaluate on BLIMP
        if 'blimp' in self.evaluation_metrics:
            logging.info("Evaluating on BLIMP and AOA...")
            blimp_evaluator = BlimpEvaluator(
                inference_model_dir,
                device=self.args.device,
                process_index=self.args.process_index,  # world (global) process index
                world_size=self.args.world_size,
                dry_run=self.dry_run,
                keep_predictions=True, # NOTE: For POS SMOOTHING we need to keep the predictions
                run_aoa='aoa' in self.evaluation_metrics,
            )
            # Get average of blimp metrics
            blimp_metrics = blimp_evaluator()
            metrics.update(blimp_metrics)  # type: ignore

        if 'glue' in self.evaluation_metrics or 'msgs' in self.evaluation_metrics:
            logging.info("Evaluating on finetuning tasks...")
            finetune_evaluator = FinetuneEvaluator(
                inference_model_dir,
                device=self.args.device,
                process_index=self.args.process_index,  # world (global) process index
                world_size=self.args.world_size,
                dry_run=self.dry_run,
                run_glue='glue' in self.evaluation_metrics,
                run_msgs='msgs' in self.evaluation_metrics,
                keep_predictions=True, # NOTE: For POS SMOOTHING we need to keep the predictions
            )
            # Get average of glue metrics
            finetune_metrics = finetune_evaluator()
            metrics.update(finetune_metrics)  # type: ignore

         # Save results to an all_predictions.json file
        collect_results(inference_model_dir)

        if 'blimp_bias' in self.evaluation_metrics and self.is_world_process_zero():
            logging.info('Evaluating bias on BLIMP predictions and getting finegrained BLIMP scores...')
            blimp_prediction_evaluator = BlimpBiasEvaluator(
                os.path.join(inference_model_dir, "all_predictions.json"),
                tokenizer=self.tokenizer,
                pos_lookup=self.pos_lookup,
                dry_run=self.dry_run,
            )
            blimp_bias_metrics = blimp_prediction_evaluator()
            metrics.update(blimp_bias_metrics)  # type: ignore

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[
                    f"{metric_key_prefix}_{key}"
                ] = metrics.pop(key)

        metrics.update(metrics)

        if f"{metric_key_prefix}_jit_compilation_time" in metrics:
            start_time += metrics[f"{metric_key_prefix}_jit_compilation_time"]
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
            )
        )

        # Log step of best model if running final evaluation
        if is_best_run:
            metrics[f"{metric_key_prefix}_model_step"] = int(
                self.state.best_model_checkpoint.split("checkpoint-")[-1]
            )

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def _initialize_full_lm_model(self):
        """
        Initialize a full language model that includes the base model and the mlm head.
        """

        # copy hydra config and change base_model to include mlm head
        lm_config = copy.deepcopy(self.hydra_config)
        lm_config.model.name = lm_config.model.name + "_lm"

        lm_model = load_base_model(lm_config)

        # unwrapping the base model and the mlm task head and copying that over into the lm model
        setattr(
            lm_model,
            f"{lm_model.base_model_prefix}",
            unwrap_model(self.model.base_model),
        )

        lm_model.lm_head = unwrap_model(
            self.objective_task.task_head
        )

        return lm_model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Override the Trainer._save() method to save the objective curriculum state as well,
        and to save the full language model (base model + mlm head).
        """

        if self.args.should_save:
            super()._save(output_dir=output_dir, state_dict=state_dict)

            # Saving should be done only on the main process

            # NOTE: We need to save the objective curriculum state as well
            output_dir = (
                output_dir if output_dir is not None else self.args.output_dir
            )

            mlm_model_dir = os.path.join(output_dir, "lm_model")
            task_head_dir = os.path.join(output_dir, "task_head")
            os.makedirs(mlm_model_dir, exist_ok=True)
            os.makedirs(task_head_dir, exist_ok=True)

            # save the full language model + the associated tokenizer (for inference)
            lm_model = self._initialize_full_lm_model()
            lm_model.save_pretrained(mlm_model_dir)

            self.tokenizer.save_pretrained(mlm_model_dir)

            self.objective_task.save(output_dir=task_head_dir)
            
    def _save_checkpoint(self, model, trial, metrics=None):
        """ Also save out the prediction files"""

        super()._save_checkpoint(model, trial, metrics=metrics)

        if self.is_world_process_zero():
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, "lm_model")
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            checkpoint_output_dir = os.path.join(run_dir, checkpoint_folder, "lm_model")

            # copy over aoa_predction, finetune, zeroshot and all_predictions.json
            # from the output_dir to the checkpoint folder

            for prediction_folder in ["aoa_prediction", "finetune", "zeroshot", "perplexity"]:

                src_prediction_folder_path = os.path.join(output_dir, prediction_folder)
                dst_prediction_folder_path = os.path.join(checkpoint_output_dir, prediction_folder)

                if not os.path.exists(src_prediction_folder_path):
                    continue

                shutil.copytree(
                    src_prediction_folder_path,
                    dst_prediction_folder_path
                )

            # copy over all_predictions.json
            src_all_predictions_path = os.path.join(output_dir, "all_predictions.json")
            dst_all_predictions_path = os.path.join(checkpoint_output_dir, "all_predictions.json")

            if os.path.exists(src_all_predictions_path):
                shutil.copy(
                    src_all_predictions_path,
                    dst_all_predictions_path
                )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        task_head_dir = os.path.join(resume_from_checkpoint, "task_head")
        self.objective_task.load(task_head_dir)

    def _load_best_model(self):
        super()._load_best_model()

        task_head_dir = os.path.join(
            self.state.best_model_checkpoint, "task_head"
        )
        self.objective_task.load(task_head_dir)

    def _wrap_model(self, model, training=True, dataloader=None):
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs[
                    "find_unused_parameters"
                ] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                kwargs[
                    "find_unused_parameters"
                ] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb
            if is_torch_neuroncore_available():
                return model
            if any(p.requires_grad for p in model.parameters()):
                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.args.local_rank]
                    if self.args._n_gpu != 0
                    else None,
                    output_device=self.args.local_rank
                    if self.args._n_gpu != 0
                    else None,
                    broadcast_buffers=False,  # NOTE: Important for DDP with obj. curriculum
                    **kwargs,
                )

        return model
