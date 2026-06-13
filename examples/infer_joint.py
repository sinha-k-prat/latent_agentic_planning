"""Inference for the one-model joint checkpoint: unplanned (vanilla Qwen) vs planned.

  python examples/infer_joint.py --question "..."
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from peft import PeftModel

from src.models import ModelBundle
from src.utils import get_device, load_config
from train_joint import Joint, SYS, N_PLAN

HERE = os.path.dirname(os.path.abspath(__file__))


def id2name():
    inv = json.load(open(os.path.join(os.path.dirname(HERE), "dataset", "operations.json")))
    m = {o["id"]: o["name"] for fam in inv["families"].values() for o in fam}
    m[64] = "EOP"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "joint_ckpt"))
    ap.add_argument("--question", required=True)
    ap.add_argument("--max_new", type=int, default=120)
    ap.add_argument("--max_plan", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(os.path.join(os.path.dirname(HERE), "configs/default.yaml"), overrides=[
        "model.base=Qwen/Qwen2.5-0.5B-Instruct", "model.gradient_checkpointing=false",
        "train.bf16=false", "device=cpu",
    ])
    device = get_device(cfg)
    bundle = ModelBundle(cfg, device)
    base_model = bundle.planner.get_base_model()
    bundle.planner = PeftModel.from_pretrained(base_model, args.ckpt).to(device)
    bundle.embed_module = bundle.planner.get_input_embeddings()
    joint = Joint(bundle)
    joint.load_state_dict(torch.load(os.path.join(args.ckpt, "heads.pt"), map_location=device))
    joint.eval()
    tok = bundle.tokenizer
    names = id2name()

    prompt_ids = tok(f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{args.question}"
                     f"<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).input_ids
    prompt_emb = bundle.embed_tokens(torch.tensor(prompt_ids, device=device))     # [P, d]

    gen_kw = dict(max_new_tokens=args.max_new, do_sample=False,
                  pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=True)

    # ---- UNPLANNED: vanilla base Qwen (adapters disabled, no plan) ----
    with torch.no_grad(), bundle.planner.disable_adapter():
        g = bundle.planner.generate(inputs_embeds=prompt_emb.unsqueeze(0),
                                    attention_mask=torch.ones(1, len(prompt_ids), dtype=torch.long), **gen_kw)
    unplanned = tok.decode(g[0], skip_special_tokens=True)

    # ---- PLANNED: generate plan tokens (plan head), then response (language head, LoRA on) ----
    with torch.no_grad():
        ctx = prompt_emb.unsqueeze(0)
        plan = []
        for _ in range(args.max_plan):
            out = bundle.planner_forward(inputs_embeds=ctx, output_hidden_states=True)
            tid = int(joint.plan_head(out.hidden_states[-1][0, -1].float()).argmax())
            plan.append(tid)
            emb = joint.plan_emb(torch.tensor([tid], device=device)).to(bundle.dtype)
            ctx = torch.cat([ctx, emb.unsqueeze(0)], dim=1)
            if tid == 64:
                break
        full = torch.cat([prompt_emb, joint.plan_emb(torch.tensor(plan, device=device)).to(bundle.dtype)], 0)
        g = bundle.planner.generate(inputs_embeds=full.unsqueeze(0),
                                    attention_mask=torch.ones(1, full.shape[0], dtype=torch.long), **gen_kw)
    planned = tok.decode(g[0], skip_special_tokens=True)

    print("QUESTION:", args.question)
    print("\nPLAN TOKENS:", [names.get(t, t) for t in plan])
    print("\n================ UNPLANNED (vanilla Qwen-0.5B) ================\n")
    print(unplanned.strip())
    print("\n================ PLANNED (LoRA + plan prefix) =================\n")
    print(planned.strip())


if __name__ == "__main__":
    main()
