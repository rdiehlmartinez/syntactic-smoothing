from .registry import register_model

from transformers import RobertaConfig
from transformers import RobertaForMaskedLM as _RobertaForMaskedLM
from transformers import RobertaModel as _RobertaModel
from transformers import RobertaPreLayerNormConfig
from transformers import (
    RobertaPreLayerNormForMaskedLM as _RobertaPreLayerNormForMaskedLM,
)
from transformers import RobertaPreLayerNormModel as _RobertaPreLayerNormModel

### Wrapping the Roberta models to make them compatible with the model registry ###


@register_model("roberta_pre_layer_norm_lm", RobertaPreLayerNormConfig)
class RobertaPreLayerNormForMaskedLM(_RobertaPreLayerNormForMaskedLM):
    MODEL_TYPE = "encoder"

@register_model("roberta_pre_layer_norm", RobertaPreLayerNormConfig)
class RobertaPreLayerNormModel(_RobertaPreLayerNormModel):
    MODEL_TYPE = "encoder"

@register_model("roberta_lm", RobertaConfig)
class RobertaForMaskedLM(_RobertaForMaskedLM):
    MODEL_TYPE = "encoder"

@register_model("roberta", RobertaConfig)
class RobertaModel(_RobertaModel):
    MODEL_TYPE = "encoder"
