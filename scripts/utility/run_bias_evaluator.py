# Utility script to run extra blimp evaluation metrics on older runs. Run_path can either be a single run or a group of runs.
# Usage: python run_bias_evaluator.py <wandb_name> <checkpoint_path>
# Example: python scripts/utils/run_bias_evaluator.py baby-lm/baseline/q0uqkygx checkpoints/baseline/model1

from src.evaluators.blimp_bias_evaluator import BlimpBiasEvaluator
from src.utils.data import DatasetPreprocessor, POSLookup

import argparse
from multiprocessing import freeze_support
import os
import sys
import wandb
from transformers import AutoTokenizer
from datasets import load_dataset
from omegaconf import OmegaConf

def run_better_blimp(wandb_name):

    # Check if run_path is a single run or a group of runs
    runs = []
    api = wandb.Api()
    if wandb_name.count('/') >= 2:
        runs = [api.run(wandb_name)]
    else:
        runs = api.runs(wandb_name)

    # Load each run
    for run in runs:

        checkpoint_path = run.config['output_dir']
        print('Rerunning bias evaluator for run: ', run.name)

        # Find the config file in the directory that contains the substring "hydra_config"
        hydra_conf_file = [f for f in os.listdir(checkpoint_path) if 'hydra_config' in f][0]
        cfg = OmegaConf.load(os.path.join(checkpoint_path, hydra_conf_file))

        predictions_path = os.path.join(checkpoint_path, 'lm_model/all_predictions.json')
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        dataset = load_dataset(run.config['dataset']['name'], run.config['dataset']['subconfig'])
        data_preprocessor = DatasetPreprocessor(cfg, tokenizer)
        train_dataset = dataset["train"].map(
            data_preprocessor,
            batched=True,
            num_proc=64,
            remove_columns=dataset["train"].column_names,
        )
        pos_lookup = POSLookup(train_dataset, tokenizer, similarity_metric=cfg.pos_lookup.similarity_metric)
        evaluator = BlimpBiasEvaluator(predictions_path, tokenizer, pos_lookup, dry_run=True)
        eval_best = evaluator.get_blimp_scores()
        
        # Init run 
        run = wandb.init(
            id=run.id,
            project=run.project,
            entity=run.entity,
            resume='must',
        )
        run.log(eval_best)
        run.finish()
        
if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('wandb_name', type=str)
    args = parser.parse_args()

    run_better_blimp(args.wandb_name)
