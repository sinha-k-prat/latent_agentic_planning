"""Backbone loading, LoRA planner, frozen executor, and a thin forward/generate API.

The executor shares one base model with the planner: it is the planner with the
LoRA adapters disabled (`disable_adapter()`), so its weights are frozen forever.
We backprop *through* the executor's forward into the plan vectors, never into its
weights (base params have requires_grad=False).
"""
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left padding so the last token is real (encode + generate)
    return tok


def _dtype(cfg):
    return torch.bfloat16 if cfg.train.bf16 else torch.float32


class ModelBundle:
    """Holds the tokenizer, the LoRA planner, and (optionally) a separate frozen executor."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.dtype = _dtype(cfg)
        self.tokenizer = load_tokenizer(cfg.model.base)
        self.share_base = bool(cfg.model.share_base)

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model.base, torch_dtype=self.dtype, attn_implementation="eager"
        )
        base.config.use_cache = False

        if cfg.model.full_finetune:
            self.planner = base  # train everything (no adapters)
        else:
            lora = LoraConfig(
                r=cfg.model.lora_r,
                lora_alpha=cfg.model.lora_alpha,
                lora_dropout=cfg.model.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(cfg.model.lora_target_modules),
            )
            self.planner = get_peft_model(base, lora)  # base params frozen, LoRA trainable

        self.planner.to(device)
        if cfg.model.gradient_checkpointing:
            self.planner.gradient_checkpointing_enable()
            # Needed so that backprop reaches inputs_embeds through the checkpointed blocks.
            if hasattr(self.planner, "enable_input_require_grads"):
                self.planner.enable_input_require_grads()

        # Optional two-instance fallback: a separate frozen executor.
        self.executor = None
        if not self.share_base:
            ex = AutoModelForCausalLM.from_pretrained(
                cfg.model.base, torch_dtype=self.dtype, attn_implementation="eager"
            )
            ex.config.use_cache = False
            for p in ex.parameters():
                p.requires_grad_(False)
            ex.eval()
            ex.to(device)
            if cfg.model.gradient_checkpointing:
                ex.gradient_checkpointing_enable()
            self.executor = ex

        self.embed_module = self.planner.get_input_embeddings()
        self.hidden_size = base.config.hidden_size

    # ---- embeddings -------------------------------------------------------
    def embed_tokens(self, ids):
        """ids: LongTensor -> embeddings in the model dtype."""
        ids = ids.to(self.device)
        return self.embed_module(ids).to(self.dtype)

    # ---- planner (LoRA active) -------------------------------------------
    def planner_forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                        output_hidden_states=False):
        return self.planner(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )

    # ---- executor (frozen base; adapters disabled) -----------------------
    @contextmanager
    def _as_executor(self):
        if self.share_base:
            # disable_adapter() is a no-op for full_finetune models; guard for it.
            if hasattr(self.planner, "disable_adapter"):
                with self.planner.disable_adapter():
                    yield self.planner
            else:
                yield self.planner
        else:
            yield self.executor

    def executor_logits(self, inputs_embeds=None, input_ids=None, attention_mask=None):
        with self._as_executor() as ex:
            out = ex(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
        return out.logits

    @torch.no_grad()
    def executor_generate(self, inputs_embeds, attention_mask, max_new_tokens,
                          temperature, top_p):
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        with self._as_executor() as ex:
            # With inputs_embeds and no input_ids, generate returns only the NEW tokens.
            out = ex.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                              **gen_kwargs)
        return out

    # ---- trainable params -------------------------------------------------
    def trainable_backbone_params(self):
        return [p for p in self.planner.parameters() if p.requires_grad]

    def weight_signature(self):
        """A cheap hash of the frozen base weights, to assert the executor never changes."""
        h = 0.0
        with torch.no_grad():
            for name, p in self.planner.named_parameters():
                if not p.requires_grad:  # base / frozen weights
                    h += float(p.float().sum().item())
        return h
