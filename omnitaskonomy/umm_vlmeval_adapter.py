"""Expose a UMM adapter to VLMEvalKit's benchmark prompts and scoring."""

from vlmeval.vlm.base import BaseModel

from omnitaskonomy.umm import load_adapter, record_adapter_provenance


class OmniTaskonomyUMM(BaseModel):
    INTERLEAVE = True
    INSTALL_REQ = False
    allowed_types = ["image", "text"]

    def __init__(self, adapter, model_path, checkpoint=None, seed=42,
                 device="cuda:0", adapter_options=None, provenance_path=None):
        super().__init__()
        from transformers import set_seed

        set_seed(seed)
        self.adapter = load_adapter(adapter, model_path, checkpoint, device, adapter_options)
        if provenance_path is not None:
            record_adapter_provenance(provenance_path, self.adapter, adapter, adapter_options)
        self.adapter.model.eval()
        self.seed = seed

    def generate_inner(self, message, dataset=None):
        import torch

        with torch.inference_mode():
            answer = self.adapter.generate(message, dataset=dataset, seed=self.seed)
        if not isinstance(answer, str):
            raise TypeError("Benchmark adapters must generate a text answer")
        return answer
