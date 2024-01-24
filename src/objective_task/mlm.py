""" Sets up the masked language modeling task. """

from torch.nn.parallel import DistributedDataParallel
from transformers import (
    AdamW,
    RobertaConfig,
    get_linear_schedule_with_warmup,
)
from transformers.models.roberta_prelayernorm.modeling_roberta_prelayernorm import (
    RobertaPreLayerNormLMHead,
)

from .base_task import BaseObjectiveTask
from .registry import register_objective_task

from .utils import CustomDataCollatorForLanguageModeling


@register_objective_task("mlm")
class MLMTask(BaseObjectiveTask):

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


        # TODO: The MLM Task should be able to support non-Roberta models as well. (LOW PRIORITY)

        mlm_head_config = RobertaConfig(
            vocab_size=self.tokenizer.vocab_size,  # type: ignore
            hidden_size=self.hidden_rep_size,
            **self.objective_task_params["task_head_params"],
        )

        self._mlm_head = RobertaPreLayerNormLMHead(mlm_head_config).to(
            self.device
        )

        should_tie_weights = self.objective_task_params["task_head_params"].get(
            "tie_weights", False
        )
        if should_tie_weights:
            model_input_embedding_layer = self.base_model.get_input_embeddings()
            self._mlm_head.decoder.weight = model_input_embedding_layer.weight


        if self.local_rank != -1:
            self._mlm_head = DistributedDataParallel(
                self._mlm_head,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        # Setting up optimizer and scheduler
        self._optimizer = AdamW(
            self._mlm_head.parameters(),
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
        Checks to see if the objective_task_params contain all required params
        and keyword args to succesfully run the task unit.
        """
        assert (
            "mask_probability" in self.objective_task_params["optional_kwargs"]
        ), "Mask probability needs to be provided to use MLM task unit"

        assert ( 
            "unmask_probability" in self.objective_task_params["optional_kwargs"]
        ), "Unmask probability needs to be provided to use MLM task unit"

        super().check_valid_config()

    @property
    def objective_collator(self):
        """
        Returns the instance objective collator.
       """

        return CustomDataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.objective_task_params["optional_kwargs"][
                "mask_probability"
            ],
            unmask_probability=self.objective_task_params["optional_kwargs"][
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
