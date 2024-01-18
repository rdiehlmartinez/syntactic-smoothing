""" Sets up the masked language modeling base task. """

from torch.nn.parallel import DistributedDataParallel
from transformers import (
    AdamW,
    RobertaConfig,
    get_linear_schedule_with_warmup,
)
from transformers.models.roberta_prelayernorm.modeling_roberta_prelayernorm import (
    RobertaPreLayerNormLMHead,
)

from .base_task import BaseTaskUnit
from .registry import register_task_unit

from ..utils import CustomDataCollatorForLanguageModeling


@register_task_unit("mlm")
class MLMTask(BaseTaskUnit):

    SUPPORTED_MODEL_TYPES = ["encoder"]

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the masked language modeling task unit.
        """
        super().__init__(*args, **kwargs)

        mlm_head_config = RobertaConfig(
            vocab_size=self.tokenizer.vocab_size,  # type: ignore
            hidden_size=self.hidden_rep_size,
            **self.task_unit_params["task_head_params"],
        )

        self._mlm_head = RobertaPreLayerNormLMHead(mlm_head_config).to(
            self.device
        )

        should_tie_weights = self.task_unit_params["task_head_params"].get(
            "tie_weights", False
        )
        if should_tie_weights:
            self._mlm_head.decoder.weight = self.base_model.embeddings.word_embeddings.weight

        if self.local_rank != -1:
            self._mlm_head = DistributedDataParallel(
                self._mlm_head,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        # Setting up optimizer and scheduler
        self._optimizer = AdamW(
            self._mlm_head.parameters(),
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
        assert (
            "mask_probability" in self.task_unit_params["optional_kwargs"]
        ), "Mask probability needs to be provided to use MLM task unit"

        assert ( 
            "unmask_probability" in self.task_unit_params["optional_kwargs"]
        ), "Unmask probability needs to be provided to use MLM task unit"

        assert (
            "label_smoothing" in self.task_unit_params["optional_kwargs"]
        ), "Label smoothing needs to be provided to use MLM task unit"

    @property
    def objective_collator(self):
        """
        Returns the instance objective collator.
       """

        return CustomDataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            task_unit_name=self.task_unit_name,
            mlm_probability=self.task_unit_params["optional_kwargs"][
                "mask_probability"
            ],
            unmask_probability=self.task_unit_params["optional_kwargs"][
                "unmask_probability"
            ],
        )

    @property
    def task_head(self):
        """
        Returns the instance mlm head.
        """
        return self._mlm_head

    @task_head.setter
    def task_head(self, new_head):
        """
        Sets the instance mlm head.
        """
        self._mlm_head = new_head

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
