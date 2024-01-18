from .registry import register_model
from transformers import OPTConfig 
from transformers import OPTModel as _OPTModel

@register_model("opt", OPTConfig)
class OPTModel(_OPTModel):
    MODEL_TYPE = "decoder"