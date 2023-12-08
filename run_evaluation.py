# Script to run evaluation on older runs by resuming the run and loading the best model.
# Can take a single run or a group of runs.
# Usage: python run_evaluation.py <wandb_name> <metrics>
# Example: python run_evaluation.py baby-lm/baseline/q0uqkygx blimp,perplexity,blimp_bias,aoa,glue,msgs

from src.models import load_base_model
from src.utils.data import DatasetPreprocessor, POSLookup
from src.utils.setup import set_seed
from src.trainer import CustomTrainer

import argparse
import logging
import os
import sys
import wandb
from datasets import load_dataset
from multiprocessing import freeze_support
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerState

logger = logging.getLogger(__name__)

def run_evaluation_from_wandb_run(run, metrics):

    logger.info(f'Running evaluation of metrics: {metrics} for run: {run.name}')
    checkpoint_path = run.config['output_dir']
    if not os.path.exists(checkpoint_path):
        logger.info(f'Path {checkpoint_path} does not exist. Skipping.')
        return

    # Find the hydra config file and update it with the new metrics
    hydra_conf_file = [f for f in os.listdir(checkpoint_path) if 'hydra_config' in f][0]
    cfg = OmegaConf.load(os.path.join(checkpoint_path, hydra_conf_file))
    cfg.trainer.evaluation_metrics = metrics
    cfg.experiment.resume_checkpoint_path = checkpoint_path
    cfg.experiment.resume_run_id = run.id
    os.environ["WANDB_RUN_ID"] = run.id
    os.environ["WANDB_RESUME"] = "allow"

    # Set seed, load dataset, tokenizer, and pos_lookup
    set_seed(cfg.experiment.seed)
    dataset = load_dataset(cfg.dataset.name, cfg.dataset.subconfig)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = load_base_model(cfg)
    data_preprocessor = DatasetPreprocessor(cfg, tokenizer)
    train_dataset = dataset["train"].map(
        data_preprocessor,
        batched=True,
        num_proc=64,
        remove_columns=dataset["train"].column_names,
    )
    pos_lookup = POSLookup(train_dataset, tokenizer, similarity_metric=cfg.pos_lookup.similarity_metric)
    eval_dataset = dataset["validation"].map(
        data_preprocessor,
        batched=True,
        num_proc=64,
        remove_columns=dataset["validation"].column_names,
        load_from_cache_file=False,
    )

    # Resume wandb run on process 0
    if int(os.environ.get("RANK", "0")) == 0:
        wandb.init(
            entity=run.entity,
            project=run.project,
            name=run.name,
            config=run.config,
            id=run.id,
            resume="allow",
        )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"checkpoints/{cfg.experiment.group}/{cfg.experiment.name}",
        overwrite_output_dir=False,
        do_train=False,
        do_eval=True,
        do_predict=False,
        per_device_train_batch_size=cfg.trainer.batch_size,  # NOTE: We can should maybe use auto_find_batch_size
        learning_rate=cfg.trainer.lr,
        max_steps=cfg.trainer.max_training_steps,
        warmup_steps=cfg.trainer.num_warmup_steps,
        seed=cfg.experiment.seed,
        evaluation_strategy="no",
        logging_steps=1,
        run_name=cfg.experiment.name,
        report_to=["wandb"]
        if not cfg.experiment.offline_run
        else None,  # wandb deactivated for offline runs
        save_strategy="no",
        hub_token=os.environ["HF_WRITE_TOKEN"]
        if not cfg.experiment.offline_run
        else None,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_perplexity_mean",
        greater_is_better=False,  # smaller perplexity is better
        ddp_find_unused_parameters=False,
        ddp_timeout=28800,  # 8 hours (default is 30 minutes)
    )

    # Set up trainer
    trainer = CustomTrainer(
        hydra_config=cfg,
        dry_run=cfg.experiment.dry_run,
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        pos_lookup=pos_lookup,
    )

    # Get last folder named 'checkpoint-xxx' in the checkpoint path
    # NOTE: This is a hacky way to get the last checkpoint folder
    checkpoint_folder = [f for f in os.listdir(checkpoint_path) if 'checkpoint' in f][-1]
    checkpoint_folder = os.path.join(checkpoint_path, checkpoint_folder)
    trainer._load_from_checkpoint(checkpoint_folder)
    trainer.state = TrainerState.load_from_json(os.path.join(checkpoint_folder, "trainer_state.json"))
    trainer._load_best_model()

    logger.info('Loaded best model. Overriding config to evaluate on all tasks.')
    trainer.evaluate(metric_key_prefix="eval_best") # Note that this will save the best model into the main output dir

if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('wandb_name', type=str, help='name of the run or group of runs. e.g. "baby-lm/baseline/q0uqkygx"')
    parser.add_argument('metrics', type=str, help='comma separated list of metrics to evaluate')
    args = parser.parse_args()

    metrics = args.metrics.split(',')

    # Check if wandb_name is a single run or a group of runs
    runs = []
    api = wandb.Api()
    if args.wandb_name.count('/') >= 2:
        run_evaluation_from_wandb_run(api.run(args.wandb_name), metrics)
    else:
        for run in api.runs(args.wandb_name):
            run_evaluation_from_wandb_run(run, metrics)
