""" Base class for objective tasks . """

import os
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Union, Tuple

from torch import Tensor, device
from torch import load as torch_load
from torch import save as torch_save
from torch.nn import Module
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# typing imports
from transformers import PreTrainedTokenizerFast
from transformers.modeling_utils import unwrap_model

class BaseObjectiveTask(metaclass=ABCMeta):


    SUPPORTED_MODEL_TYPES = []

    def __init__(
        self,
        objective_task_params: Mapping[str, Any],
        num_training_steps: int,
        hidden_rep_size: int,
        tokenizer: PreTrainedTokenizerFast,
        device: device,
        local_rank: int,
        base_model: Module,
    ) -> None:
        """
        Initializes the abc class for objective tasks. 

        Args:
            * tokenizer (PreTrainedTokenizerFast): The tokenizer used for tokenizing the input,
                used primarily for the objective collator.
            * num_training_steps (int): The number of training steps for the objective task
            * objective_task_params (Mapping[str, Any]): The parameters for the objective task 
            * hidden_rep_size (int): The size of the hidden representation of the model [this
                is the size of the last hidden layer of the base model, which is the input to the
                task head]
            * base_model (torch.nn.Module): The model that is trained (used for possibly
                intializing the task head with the output matrix to be tied to the input embeddings)
        """


        self.objective_task_name = objective_task_params["name"]
        self.objective_task_params = objective_task_params

        self.num_training_steps = num_training_steps
        self.hidden_rep_size = hidden_rep_size

        self.tokenizer = tokenizer

        self.device = device
        self.local_rank = local_rank

        self.base_model = base_model

        self.check_valid_config()

    def optimizer_step(self) -> None:
        """
        Performs an optimizer step for the objective task.
        """
        self.optimizer.step()
        self.scheduler.step()
        self.task_head.zero_grad()

    @abstractmethod
    def check_valid_config(self) -> None:
        """
        Checks to see if the objective_task_params contain all required params
        and keyword args to succesfully run the objective task.
        """
        
        assert (self.base_model.MODEL_TYPE in self.SUPPORTED_MODEL_TYPES), (
            f"Model type {self.base_model.MODEL_TYPE} is not supported for "
            f"objective task {self.objective_task_name}"
        )

    @property
    @abstractmethod
    def optimizer(self) -> Optimizer:
        """
        Returns the optimizer used for training the task head
        """
        ...

    @property
    @abstractmethod
    def scheduler(self) -> _LRScheduler:
        """
        Returns the scheduler used for training the task head
        """
        ...

    @property
    @abstractmethod
    def objective_collator(
        self,
    ) -> Callable[
        [List[Union[List[int], Any, Dict[str, Any]]]], Dict[str, Tensor]
    ]:
        """
        Returns the objective collator used for setting up the objective that is used
        to train the model.
        """
        ...

    @property
    @abstractmethod
    def task_head(self) -> Module:
        """
        Returns the task head that is used for training the model on the given task
        """
        ...

    @task_head.setter
    @abstractmethod
    def task_head(self, new_head: Module) -> None:
        """
        Sets the task head for the task
        """
        ...

    def compute_loss(
        self,
        model: Module,
        inputs: Dict[str, Tensor],
        override_input_ids: Optional[Tensor] = None,
        override_lables: Optional[Tensor] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        Given a batch of data, computes the loss for the given task.

        Args:
            * module (Module): The model to be trained on the given task
            * inputs (Dict[str, Tensor]): The inputs to the task head
            * override_input_ids (Optional[Tensor], optional): Overrides the input ids for the task
            * override_lables (Optional[Tensor], optional): Overrides the labels for the task,
                usually we assume that the labels are in the inputs, but in some cases we may want
                to override the labels. Defaults to None.
            * loss_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to be
                passed to the loss function. Defaults to None.
        """

        input_ids = (
            override_input_ids
            if override_input_ids is not None
            else inputs["input_ids"]
        )

        base_model_outputs = model(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"]
            if "attention_mask" in inputs
            else None,
        )
        base_model_hidden_states = base_model_outputs[0]

        # compute the logits
        logits = self.task_head(base_model_hidden_states).transpose(-1, -2)

        labels = (
            override_lables
            if override_lables is not None
            else inputs["labels"]
        )

        logits, labels = self.logit_label_transform(logits, labels)

        # compute the loss
        loss = cross_entropy(logits, labels, **(loss_kwargs or {}))

        return loss

    def logit_label_transform(self, logits: Tensor = None, labels: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Called before computing the loss, transforms the logits and labels if necessary just 
        before computing the loss.

        By default performs a no-op.
        """
        return logits, labels


    def save(self, output_dir: str) -> None:
        """
        Saves the objective task to the given directory.
        """

        torch_save(
            unwrap_model(self.task_head).state_dict(),
            os.path.join(output_dir, f"{self.objective_task_name}_task_head.pt"),
        )

        torch_save(
            self.optimizer.state_dict(),
            os.path.join(output_dir, f"{self.objective_task_name}_optimizer.pt"),
        )
        torch_save(
            self.scheduler.state_dict(),
            os.path.join(output_dir, f"{self.objective_task_name}_scheduler.pt"),
        )

    def _possibly_wrap_state_dict(
        self, state_dict: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Wraps the state dict in a DistributedDataParallel state dict if the objective task is
        distributed.
        """

        if self.local_rank != -1:
            state_dict = {
                f"module.{key}": value for key, value in state_dict.items()
            }

        return state_dict

    def load(self, input_dir: str) -> None:
        """
        Loads the objective task from the given directory.
        """
        self.task_head.load_state_dict(
            self._possibly_wrap_state_dict(
                torch_load(
                    os.path.join(
                        input_dir, f"{self.objective_task_name}_task_head.pt"
                    ),
                    map_location=self.device,
                )
            )
        )

        self.optimizer.load_state_dict(
            torch_load(
                os.path.join(input_dir, f"{self.objective_task_name}_optimizer.pt"),
                map_location=self.device,
            )
        )
        self.scheduler.load_state_dict(
            torch_load(
                os.path.join(input_dir, f"{self.objective_task_name}_scheduler.pt"),
                map_location=self.device,
            )
        )
