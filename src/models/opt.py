from .registry import register_model
from transformers import OPTConfig 
from transformers import OPTModel as _OPTModel
from transformers import OPTModelForCausalLM as _OPTModelForCausalLM

@register_model("opt", OPTConfig)
class OPTModel(_OPTModel):
    MODEL_TYPE = "decoder"

@register_model("opt_causal_lm", OPTConfig)
class OPTModelForCausalLM(_OPTModelForCausalLM):
    MODEL_TYPE = "decoder"