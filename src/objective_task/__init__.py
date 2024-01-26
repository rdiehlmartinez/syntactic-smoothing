from .base_task import BaseObjectiveTask
from .mlm import MLMTask
from .causal import CausalLMTask
from .pos_smoothing import POSSMOOTHINGTask
from .registry import OBJECTIVE_TASK_REGISTRY

# Typing imports

import torch
from ..config import BabyLMConfig
from transformers import PreTrainedTokenizerFast

# Typing imports

def load_objective_task( 
    cfg: BabyLMConfig,
    hidden_rep_size: int,
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
    local_rank: int,
    base_model: torch.nn.Module, 
) -> BaseObjectiveTask:

    objective_task_params = cfg.objective_task

    task_name = objective_task_params['name']

    return OBJECTIVE_TASK_REGISTRY[task_name](
        objective_task_params=objective_task_params,
        num_training_steps=cfg.trainer.max_training_steps,
        hidden_rep_size=hidden_rep_size,
        tokenizer=tokenizer,
        device=device,
        local_rank=local_rank,
        base_model=base_model,
    )