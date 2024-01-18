""" Sets up causal (auto-regressive) language modeling base task. """

from torch.nn import Linear
from torch.nn.parallel import DistributedDataParallel
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

from .base_task import BaseTaskUnit
from .registry import register_task_unit

from ..utils import CustomDataCollatorForLanguageModeling

@register_task_unit("causallm")
class CausalLMTask(BaseTaskUnit):

    SUPPORTED_MODEL_TYPES = ["decoder"]

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the causal language modeling task unit.
        """
        super().__init__(*args, **kwargs)

        self._causal_lm_head = Linear(self.hidden_rep_size, self.tokenizer.vocab_size).to(
            self.device
        )

        should_tie_weights = self.task_unit_params["task_head_params"].get(
            "tie_weights", False
        )
        if should_tie_weights:
            self._causal_lm_head.weight = self.base_model.embeddings.word_embeddings.weight

        if self.local_rank != -1:
            self._causal_lm_head = DistributedDataParallel(
                self._causal_lm_head,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        # Setting up optimizer and scheduler
        self._optimizer = AdamW(
            self._causal_lm_head.parameters(),
            **self.task_unit_params["optimizer_params"],
        )

        self._scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.task_num_steps // 10,
            num_training_steps=self.task_num_steps,
        )

    def check_valid_config(self) -> None:
        """
        Checks to see if the task_unit_params contain all required params
        and keyword args to succesfully run the task unit.
        """
        pass

    @property
    def objective_collator(self):
        """
        Returns the instance objective collator.
       """

        return CustomDataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            task_unit_name=self.task_unit_name,
        )

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
