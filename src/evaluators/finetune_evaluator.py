""" Class for calling the finetuning portion of the evaluation pipeline on a model """

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, Union

# typing imports
import torch
import torch.distributed as dist

from src.utils.setup import TORCH_RUN_ENV_KEYS

logger = logging.getLogger(__name__)


class FinetuneEvaluator(object):

    GLUE_TASKS = [
        "cola",
        "sst2",
        "mrpc",
        "qqp",
        "mnli",
        "mnli-mm",
        "qnli",
        "rte",
        "boolq",
        "multirc",
        "wsc",
    ]

    MSGS_TASKS = [
        "main_verb_control",
        "control_raising_control",
        "syntactic_category_control",
        "lexical_content_the_control",
        "relative_position_control",
        "main_verb_lexical_content_the",
        "main_verb_relative_token_position",
        "syntactic_category_lexical_content_the",
        "syntactic_category_relative_position",
        "control_raising_lexical_content_the",
        "control_raising_relative_token_position",
    ]

    def __init__(
        self,
        out_dir: str,
        device: torch.device,
        process_index: int,
        world_size: int,
        dry_run: bool = False,
        run_glue: bool = True,
        run_msgs: bool = False,
        keep_predictions: bool = False,
    ):
        """
        Args:
            * out_dir (str): Path to the output directory
            * device (torch.device): Device to run the evaluation on
            * process_index (int): Index of the current process
            * world_size (int): Number of processes
            * dry_run (bool): If True, don't actually run the evaluation script
            * run_glue (bool): If True, finetune on all GLUE tasks
            * run_msgs (bool): If True, finetune on all MSGS tasks
            * keep_predictions (bool): If True, keep the predictions from the finetuning
        """

        if not run_glue and not run_msgs:
            raise ValueError(
                "run_glue and run_msgs cannot both be False. Must run at least one of GLUE or MSGS tasks"
            )

        self.out_dir = out_dir
        self.device = device
        self.process_index = process_index
        self.world_size = world_size
        self.dry_run = dry_run
        self.run_glue = run_glue
        self.run_msgs = run_msgs
        self.keep_predictions = keep_predictions

    def run_script(self, task: str):

        logger.info(f"Running finetuning script for {task}...")

        if task == "mnli":
            valid_name = "validation_matched"
            out_dir = "mnli"
        elif task == "mnli-mm":
            valid_name = "validation_mismatched"
            task = "mnli"
            out_dir = "mnli-mm"
        else:
            valid_name = "validation"
            out_dir = task

        os.makedirs(
            os.path.join(self.out_dir, "finetune", out_dir), exist_ok=True
        )

        task_group = "glue" if task in self.GLUE_TASKS else "msgs"

        cmd = (
            "cd lib/evaluation-pipeline; python finetune_classification.py"
            + f" --model_name_or_path ../../{self.out_dir}"
            + f" --output_dir ../../{self.out_dir}/finetune/{out_dir}"
            + f" --train_file filter-data/{task_group}_filtered/{task}.train.json"
            + f" --validation_file filter-data/{task_group}_filtered/{task}.{valid_name}.json"
            + " --do_train"
            + " --do_eval"
            + " --do_predict"
            + " --use_fast_tokenizer True"  # Set to True to use fast tokenizer
            + " --max_seq_length 128"
            + " --per_device_train_batch_size 64"
            + " --learning_rate 5e-5"
            + " --num_train_epochs 10"
            + " --evaluation_strategy steps"
            + " --patience 10"
            + " --eval_every 200"
            + " --eval_steps 200"
            + " --overwrite_output_dir"
            + " --seed 12"
            # + f" --logging_steps 1" NOTE: ENABLE THIS FOR DEBUGGING
        )

        # print all the key names of the envrioment variables

        subprocess_env = os.environ.copy()
        # remove from subprocess_env all torch_run related variables
        for key in list(subprocess_env.keys()):
            if key in TORCH_RUN_ENV_KEYS:
                del subprocess_env[key]

        if self.world_size > 1:
            # Set CUDA_VISIBLE_DEVICES to the local process index (assuming 4 GPUs per node)
            subprocess_env["CUDA_VISIBLE_DEVICES"] = str(
                self.process_index % 4
            )

        # Disable W&B on subprocess
        # NOTE: COMMENT OUT FOR DEBUGGING
        subprocess_env["WANDB_DISABLED"] = "true"
        subprocess_env["WANDB_MODE"] = "disabled"

        logging.info(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, env=subprocess_env)
        logging.info(f"Finished finetuning {task}.")

    def __call__(self) -> Union[Dict[str, Any], None]:
        """
        Runs the GLUE evaluation pipeline.
        """

        # Start a subprocess to run the lib/evaluation-pipeline/babylm_eval.py script
        logger.info("Running Finetuning evaluation script...")

        tasks = []
        if self.run_glue:
            if self.dry_run:
                tasks.extend(["cola"])
                logger.info("Running dry run. Only running on CoLA from GLUE.")
            else:
                tasks.extend(self.GLUE_TASKS)
                logger.info(
                    "Running on all GLUE tasks: " + ", ".join(self.GLUE_TASKS)
                )
        if self.run_msgs:
            if self.dry_run:
                tasks.extend(["main_verb_control"])
                logger.info(
                    "Running dry run. Only running on main_verb_control from MSGS."
                )
            else:
                tasks.extend(self.MSGS_TASKS)
                logger.info(
                    "Running on all MSGS tasks: " + ", ".join(self.MSGS_TASKS)
                )

        for task_idx, task in enumerate(tasks):
            if task_idx % self.world_size != self.process_index:
                continue
            self.run_script(task)

        if self.world_size > 1:
            dist.barrier()

        # Iterate through all directories in out_dir/zeroshot
        # and get the accuracies from the eval_results.json files
        logger.info(
            "Finetuning Evaluation script finished. Getting accuracies..."
        )
        accuracies = {}

        for task in os.listdir(os.path.join(self.out_dir, "finetune")):
            with open(
                os.path.join(
                    self.out_dir, "finetune", task, "eval_results.json"
                )
            ) as f:
                data = json.load(f)
                task_group = "glue" if task in self.GLUE_TASKS else "msgs"
                accuracies[f"{task_group}_" + task + "_accuracy"] = data[
                    "eval_accuracy"
                ]
                if "eval_f1" in data:
                    accuracies[f"{task_group}_" + task + "_f1"] = data[
                        "eval_f1"
                    ]

        if self.world_size > 1:
            dist.barrier()

        # Delete the finetune directory
        if self.process_index == 0 and not self.keep_predictions:
            shutil.rmtree(os.path.join(self.out_dir, "finetune"))

        return accuracies


def collect_results(out_dir: str):
    """Attempts to run the the collect_results.py script from the evaluation pipeline"""

    cmd = (
        "cd lib/evaluation-pipeline; python collect_results.py"
        + f" ../../{out_dir}"
    )

    output = subprocess.run(
        cmd, shell=True, capture_output=True, env=os.environ.copy()
    )
    if output.returncode != 0:
        logger.warning("Failed to run collect_results.py script. Skipping...")
    return
