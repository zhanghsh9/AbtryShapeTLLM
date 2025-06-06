#!/usr/bin/env python3
# evaluate_qwen.py
"""
Evaluate a LoRA-tuned Qwen-3 model on the same 80 / 20 split
used during training and save predictions + MSE to a .mat file.
"""

import os, re, ast, json, shutil, time, math, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
from tqdm import tqdm
from scipy.io import savemat, loadmat

import torch
from torch.nn.functional import mse_loss
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# ───────────────────────── User-specific paths ───────────────────────── #
RUN_DIR   = "results_qwen3/run_20250514_103239"        # ← your run folder
MODEL_DIR = f"{RUN_DIR}/qwen3_lora"
OUT_MAT   = f"{RUN_DIR}/evaluation_predictions.mat"
# ----------------------------------------------------------------------- #

BATCH_SIZE = 1

# ═════════════════════════  Helpers ════════════════════════════════════ #
def fix_grammar(obj):
    """Same patch that was applied during fine-tuning."""
    inst, out = obj["instruction"], obj["output"]
    inst = re.sub(r"\bgrid\s+as\s*:", "grid:", inst, flags=re.I)
    out  = re.sub(r"\bvalues\s+[^[]*?\bis\s+\[",
                  lambda m: m.group(0).replace(" is ", " are "), out, flags=re.I)
    if not out.rstrip().endswith((".", "!", "?")):
        out = out.rstrip() + "."
    return {"instruction": inst, "input": obj.get("input", ""), "output": out}

GRID_RE = re.compile(r"grid\s*: (\[\[.*?\]\])", re.I | re.S)
def extract_grid(instr):
    m = GRID_RE.search(instr)
    if not m:
        raise ValueError(f"Grid not found in:\n{instr}")
    return np.asarray(ast.literal_eval(m.group(1))).reshape(-1)

def build_chat_prompt(example: dict, tokenizer) -> str:
    """
    Craft a prompt identical to training *except* we leave the assistant
    segment OPEN, so generation starts right after </think>.
    """
    instr = example["instruction"].strip()
    inp   = example.get("input", "").strip()
    user  = f"{instr}\n\n### Input:\n{inp}" if inp else instr

    return (
        "<|im_start|>user\n"
        f"{user}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"     #  ← same empty think block
        #  NO <|im_end|>               #  ← keep segment open
    )



def find_spectrum(txt):
    """First [ … ] after </think> → list of floats (or None)."""
    s0 = txt.find("</think>")
    if s0 == -1: s0 = 0
    l  = txt.find('[', s0)
    r  = txt.find(']', l + 1)
    if l == -1 or r == -1: return None
    try:  # ensure all cast to float
        return [float(x.strip()) for x in txt[l + 1:r].split(',')]
    except ValueError:
        return None
# ═══════════════════════════════════════════════════════════════════════ #

def batch_generate(prompts, true_specs, grids, model, tok,
                   max_new_tokens, bs=BATCH_SIZE):
    device = next(model.parameters()).device
    enc    = tok(prompts, return_tensors='pt',
                 padding=True, truncation=True).to(device)
    preds, mses, cps, gts = [], [], [], []
    bad_p, bad_t, bad_g   = [], [], []

    for i in tqdm(range(0, len(prompts), bs), desc="Generating", unit="batch"):
        slc = slice(i, i + bs)
        with torch.no_grad():
            outs = model.generate(**{k: v[slc] for k, v in enc.items()},
                                  max_new_tokens=max_new_tokens)
        texts = tok.batch_decode(outs, skip_special_tokens=False)

        for j, txt in enumerate(texts):
            idx  = i + j
            print(txt)
            nums = find_spectrum(txt)
            print('Extracted spectrum:', nums)
            if (nums is None or len(nums) != len(true_specs[idx])
                    or np.any(np.array(nums) > 1) or np.any(np.array(nums) < 0)):
                bad_p.append(prompts[idx]); bad_t.append(true_specs[idx])
                bad_g.append(grids[idx]);   continue

            preds.append(nums); gts.append(true_specs[idx]); cps.append(grids[idx])
            mses.append(float(mse_loss(torch.Tensor(nums),
                                       torch.Tensor(true_specs[idx]))))

    # ---- retry failures one-by-one (≤100 tries each) ----
    for p, t, g in tqdm(zip(bad_p, bad_t, bad_g),
                        total=len(bad_p), desc="Retries", unit="seq"):
        inp = tok(p, return_tensors='pt').to(device)
        for _ in range(100):
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=max_new_tokens)
            nums = find_spectrum(tok.decode(out[0], skip_special_tokens=False))
            if (nums is not None and len(nums) == len(t)
                    and np.all(np.array(nums) <= 1) and np.all(np.array(nums) >= 0)):
                preds.append(nums); gts.append(t); cps.append(g)
                mses.append(float(mse_loss(torch.Tensor(nums), torch.Tensor(t))))
                break
    return np.stack(cps), np.stack(gts), np.stack(preds), mses

def main():
    # 1) Re-load training config to recover paths + hyper-params
    cfg = loadmat(Path(RUN_DIR) / "training_stats.mat")["config"][0, 0]
    max_seq_len = int(cfg["max_seq_length"][0, 0])
    lora_rank   = int(cfg["lora_rank"][0, 0])
    dataset_path = cfg["dataset_path"][0]

    # 2) Read dataset, apply grammar fix, replicate 80/20 split
    with open(dataset_path, encoding="utf-8") as fh:
        data = [fix_grammar(x) for x in json.load(fh)]
    split_at = math.floor(len(data) * float(cfg["train_split"][0, 0]))
    test_set = data[split_at:]

    # 3) Load model for inference
    model, tok = FastLanguageModel.from_pretrained(
        model_name     = MODEL_DIR,
        max_seq_length = max_seq_len,
        load_in_4bit   = True,
        fast_inference = False,
        max_lora_rank  = lora_rank,
        gpu_memory_utilization = 0.99,
    )
    FastLanguageModel.for_inference(model)

    # --- compute max_new_tokens from a dummy prompt -------------
    dummy = build_chat_prompt(test_set[0], tok)
    max_new = max_seq_len - len(tok(dummy).input_ids)
    print("max_new_tokens =", max_new)

    # 4) Build evaluation lists
    prompts = [build_chat_prompt(ex, tok) for ex in test_set]
    grids   = [extract_grid(ex["instruction"]) for ex in test_set]
    truths  = []
    for ex in test_set:
        s, e = ex["output"].find('['), ex["output"].find(']')
        truths.append([float(x.strip()) for x in ex["output"][s + 1:e].split(',')])

    # 5) Run generation + MSE
    t0 = time.perf_counter()
    C, G, P, mse_vals = batch_generate(prompts, truths, grids,
                                       model, tok, max_new)
    t1 = time.perf_counter()

    # 6) Save
    savemat(OUT_MAT, {
        "test_CPs"        : C,
        "test_ground_truth": G,
        "test_predictions": P,
        "MSE_test"        : mse_vals,
        "MSE_test_mean"   : np.mean(mse_vals),
        "time_used"       : t1 - t0,
    })
    torch.cuda.empty_cache()
    print("Test MSE:", np.mean(mse_vals))
    print("Saved to", OUT_MAT)

if __name__ == "__main__":
    # copy evaluator into run dir for reproducibilit
    # shutil.copy(__file__, Path(RUN_DIR) / Path(__file__).name)
    main()
