from transformers.models import gemma

from spInfer.speculative_enablement import enable_speculative_inference

@enable_speculative_inference
class spGemmaForCausalLM(gemma.GemmaForCausalLM):
    pass