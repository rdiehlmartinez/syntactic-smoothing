""" Sets up causal (auto-regressive) language modeling task. """

from typing import Tuple
from torch import Tensor
from torch.nn import Linear
from torch.nn.parallel import DistributedDataParallel
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

from .base_task import BaseObjectiveTask
from .registry import register_objective_task

from .utils import CustomDataCollatorForLanguageModeling

@register_objective_task("causal_lm")
class CausalLMTask(BaseObjectiveTask):

    SUPPORTED_MODEL_TYPES = ["decoder"]

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the causal language modeling objective.
        """
        super().__init__(*args, **kwargs)

        self._causal_lm_head = Linear(self.hidden_rep_size, self.tokenizer.vocab_size, bias=False).to(
            self.device
        )

        should_tie_weights = self.objective_task_params["task_head_params"].get(
            "tie_weights", False
        )
        if should_tie_weights:
            model_input_embedding_layer = self.base_model.get_input_embeddings()
            self._causal_lm_head.weight = model_input_embedding_layer.weight

        if self.local_rank != -1:
            self._causal_lm_head = DistributedDataParallel(
                self._causal_lm_head,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        # Setting up optimizer and scheduler
        self._optimizer = AdamW(
            self._causal_lm_head.parameters(),
            **self.objective_task_params["optimizer_params"],
        )

        num_warmup_steps = self.objective_task_params["scheduler_params"].get(
            "num_warmup_steps", self.num_training_steps // 10
        )

        self._scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

    def check_valid_config(self) -> None:
        """
        Checks to see if the objective_task_params contain all required params and keyword args 
        to succesfully run the objective task.
        """
        
        super().check_valid_config()

    @property
    def objective_collator(self):
        """
        Returns the instance objective collator.
       """

        return CustomDataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def logit_label_transform(self, logits: Tensor = None, labels: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        We need to shift the labels over by one to match the logits. So that all logits in 
        positions < n predict the token n. 
        """

        # NOTE: Very important; in our code we expect logits to already be transformed so that 
        # logits is shape (batch_size, vocab_size, sequence_length). The transposition from the 
        # default (batch_size, sequence_length, vocab_size) is done in the forward pass of the
        # base model.

        if logits is not None:
            logits = logits[..., :-1]

        if labels is not None:
            labels = labels[..., 1:]

        return (logits, labels)

    @property
    def task_head(self):
        """
        Returns the instance causal head.
        """
        return self._causal_lm_head

    @task_head.setter
    def task_head(self, new_head):
        """
        Sets the instance causal head.
        """
        self._causal_lm_head = new_head

    @property
    def optimizer(self):
        """
        Returns the instance optimizer
        """
        return self._optimizer

    @property
    def scheduler(self):
        """
        Returns the instance scheduler
        """
        return self._scheduler
