"""Custom VLMEvalKit model wrappers for the shortlisted models that the pinned
VLMEvalKit build does not register natively, plus runtime registration.

We never modify VLMEvalKit (no fork): `register_custom_models()` injects entries
into `vlmeval.config.supported_VLM` at runtime — the same pattern as
`run_eval.patch_tablevqa_scorers()`.

- FastVLM-0.5B (Apple): trust_remote_code, manual <image>-token splicing.
- Qwen3.5-0.8B (Alibaba): AutoProcessor + AutoModelForMultimodalLM.
- LFM2.5-VL-450M (Liquid AI): reuses VLMEvalKit's `LFM2VL` class (needs Env B,
  transformers>=4.57, for the native Lfm2VlProcessor).

Wrappers are device-agnostic (cuda -> mps -> cpu) and store generation config in
`self.kwargs` so run_eval's `apply_gen_cap` (max_new_tokens) applies to them.
"""
from functools import partial


def _pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _base_model_cls():
    from vlmeval.vlm.base import BaseModel
    return BaseModel


def _make_fastvlm_cls():
    import torch
    from PIL import Image
    BaseModel = _base_model_cls()

    class FastVLM(BaseModel):
        """apple/FastVLM-0.5B via trust_remote_code (LLaVA-style image-token splice)."""
        INTERLEAVE = False
        IMAGE_TOKEN_INDEX = -200

        def __init__(self, model_path="apple/FastVLM-0.5B", **kwargs):
            super().__init__()
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.device = _pick_device()
            dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, trust_remote_code=True,
            ).to(self.device).eval()
            kwargs_default = {"max_new_tokens": 128, "use_cache": True}
            kwargs_default.update(kwargs)
            self.kwargs = kwargs_default

        def generate_inner(self, message, dataset=None):
            prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"<image>\n{prompt}"}],
                add_generation_prompt=True, tokenize=False,
            )
            pre, post = rendered.split("<image>", 1)
            pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
            post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
            img_tok = torch.tensor([[self.IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
            input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.model.device)
            attention_mask = torch.ones_like(input_ids, device=self.model.device)

            img = Image.open(image_path).convert("RGB")
            px = self.model.get_vision_tower().image_processor(
                images=img, return_tensors="pt")["pixel_values"]
            px = px.to(self.model.device, dtype=self.model.dtype)

            with torch.no_grad():
                out = self.model.generate(
                    inputs=input_ids, attention_mask=attention_mask, images=px, **self.kwargs)
            # FastVLM is LLaVA-style: the single -200 image token expands to many
            # visual tokens inside the model, so positional slicing by
            # input_ids.shape[1] lands in the wrong place and truncates answers.
            # Extract the assistant turn by marker instead (Qwen2 chat template).
            full = self.tokenizer.decode(out[0], skip_special_tokens=False)
            marker = "<|im_start|>assistant\n"
            ans = full.split(marker)[-1] if marker in full else full
            for stop in ("<|im_end|>", "<|endoftext|>"):
                ans = ans.split(stop)[0]
            return ans.strip()

    return FastVLM


def _make_qwen35vl_cls():
    import torch
    from PIL import Image
    BaseModel = _base_model_cls()

    class Qwen35VL(BaseModel):
        """Qwen/Qwen3.5-0.8B via AutoProcessor + AutoModelForMultimodalLM."""
        INTERLEAVE = False

        def __init__(self, model_path="Qwen/Qwen3.5-0.8B", **kwargs):
            super().__init__()
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForMultimodalLM as _AutoVLM
            except Exception:
                from transformers import AutoModelForImageTextToText as _AutoVLM
            self.device = _pick_device()
            dtype = torch.bfloat16 if self.device == "cuda" else (
                torch.float16 if self.device == "mps" else torch.float32)
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = _AutoVLM.from_pretrained(
                model_path, torch_dtype=dtype).to(self.device).eval()
            kwargs_default = {"max_new_tokens": 128, "use_cache": True}
            kwargs_default.update(kwargs)
            self.kwargs = kwargs_default

        def generate_inner(self, message, dataset=None):
            prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
            content = []
            images = []
            if image_path is not None:
                content.append({"type": "image"})
                images.append(Image.open(image_path).convert("RGB"))
            content.append({"type": "text", "text": prompt})
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True, tokenize=False,
            )
            inputs = self.processor(
                text=[text], images=images or None, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, **self.kwargs)
            gen = out[0][inputs["input_ids"].shape[1]:]
            return self.processor.decode(gen, skip_special_tokens=True).strip()

    return Qwen35VL


def _make_lfm25vl_cls():
    import torch
    from vlmeval.vlm import LFM2VL

    class LFM25VL(LFM2VL):
        """LiquidAI/LFM2.5-VL-450M: same generate path as VLMEvalKit's LFM2VL.
        The 2.5 processor (Lfm2VlProcessor) is recognized natively only in
        transformers>=4.57, so this must run in Env B (--bleeding-edge). The repo
        ships no remote processor code; trust_remote_code=True is a harmless
        fallback (the older fix mistook this version gap for a remote-code one)."""

        def __init__(self, model_path="LiquidAI/LFM2.5-VL-450M", **kwargs):
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.default_instruction_prompt = (
                "\nPlease answer directly with only the final answer, "
                "do not give any explanation."
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True)
            self.model = (
                AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                    torch_dtype=torch.bfloat16,
                )
                .cuda()
                .eval()
            )
            kwargs_default = {"max_new_tokens": 1024, "use_cache": True}
            kwargs_default.update(kwargs)
            self.kwargs = kwargs_default

    return LFM25VL


def _make_qwen35vl_finetuned_cls():
    """Qwen3.5-0.8B + a LoRA adapter (Part-2 TableVQA PoC). Reuses the base wrapper's
    loading and generate_inner verbatim — so train-time and eval-time prompts match —
    and just layers a PEFT adapter on top of the loaded model. Kept un-merged so the
    base is restorable; the adapter dir comes from $MVDE_LORA_ADAPTER at eval time."""
    Qwen35VL = _make_qwen35vl_cls()

    class Qwen35VLTuned(Qwen35VL):
        def __init__(self, model_path="Qwen/Qwen3.5-0.8B", adapter_path=None, **kwargs):
            if not adapter_path:
                raise SystemExit(
                    "Qwen3.5-0.8B-TableLoRA needs an adapter: set MVDE_LORA_ADAPTER to the "
                    "adapter dir from train_lora_qwen.py before running eval.")
            super().__init__(model_path=model_path, **kwargs)
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path).eval()

    return Qwen35VLTuned


def register_custom_models():
    """Inject our custom models into VLMEvalKit's registry (idempotent)."""
    import os
    from vlmeval.config import supported_VLM

    supported_VLM["FastVLM-0.5B"] = partial(_make_fastvlm_cls(), model_path="apple/FastVLM-0.5B")
    supported_VLM["Qwen3.5-0.8B"] = partial(_make_qwen35vl_cls(), model_path="Qwen/Qwen3.5-0.8B")
    # LFM2.5-VL-450M: same generate path as VLMEvalKit's LFM2VL; runs in Env B
    # (transformers>=4.57) where Lfm2VlProcessor is recognized natively.
    supported_VLM["LFM2.5-VL-450M"] = partial(_make_lfm25vl_cls(), model_path="LiquidAI/LFM2.5-VL-450M")
    # Part-2 PoC: fine-tuned Qwen with a LoRA adapter (path via $MVDE_LORA_ADAPTER).
    supported_VLM["Qwen3.5-0.8B-TableLoRA"] = partial(
        _make_qwen35vl_finetuned_cls(), model_path="Qwen/Qwen3.5-0.8B",
        adapter_path=os.environ.get("MVDE_LORA_ADAPTER"))
