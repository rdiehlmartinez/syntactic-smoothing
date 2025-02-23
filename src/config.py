"""Defines the set of hyperparameters to be specified in the config file."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from omegaconf import MISSING, DictConfig


@dataclass
class ExperimentParams(DictConfig):
    seed: int

    # Name of the experiment - needs to be set at runtime
    name: str = MISSING

    # Name of the group that the current experiment belongs to
    # analogous to 'project' in wandb
    group: str = MISSING

    # whether to run a minimal version of the experiment
    dry_run: bool = False

    # whether to run the experiment only offline
    offline_run: bool = False

    # Optional checkpoint path to resume training from
    resume_checkpoint_path: Optional[str] = None

    # If resume_checkpoint_path is not None and we are logging to wandb,
    # we need to specify the run_id of the run we are resuming from
    resume_run_id: Optional[str] = None


@dataclass
class DatasetParams(DictConfig):
    # name of the dataset on huggingface
    name: str
    # subconfig i.e. strict-small
    subconfig: str


@dataclass
class TokenizerParams(DictConfig):
    # data processing parameters
    name: str

    # additional optional kwargs
    add_prefix_space: Optional[bool] = None


@dataclass
class DataPreprocessingParams(DictConfig):
    # params for preprocessing the dataset (i.e. tokenization)
    include_punctuation: bool
    join_sentences: bool
    max_input_length: int
    callback_functions: Optional[List[str]] = None


@dataclass
class ModelParams(DictConfig):
    # model parameters
    name: str

    # NOTE: At least 'hidden_size' needs to be specified
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerParams(DictConfig):
    batch_size: int
    lr: float
    num_warmup_steps: int
    max_training_steps: int
    evaluation_metrics : List[str]

### Curriculum learning parameter: can be either objective or data-driven ###


## Objective curriculum learning parameters ##
@dataclass
class ObjectiveTaskParams(DictConfig):

    name: str

    # parameters for the task head architecture
    task_head_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    # parameters for the optimizer
    optimizer_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    # parameters for the scheduler
    scheduler_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    # Additional optional kwargs dependent on the objective curriculum unit
    optional_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


# Paramters for POS lookup class 
@dataclass
class POSLookupParams(DictConfig):
    similarity_metric: str

### Container for entire config ###

@dataclass
class BabyLMConfig(DictConfig):
    experiment: ExperimentParams
    dataset: DatasetParams
    tokenizer: TokenizerParams
    data_preprocessing: DataPreprocessingParams
    model: ModelParams
    trainer: TrainerParams
    objective_task: ObjectiveTaskParams
    pos_lookup: POSLookupParams
