#!/usr/bin/env python3
# evaluate_metasurface_inverse.py
"""
Evaluate a Meta-Llama-3.1-8B-Instruct model fine-tuned (with LoRA) to perform the
*inverse* mapping:

    31-point transmission spectrum  →  one 4 × 4 metasurface grid

The structure mirrors evaluate_metasurface.py from the forward-mapping task.
Only the value-extraction helpers and shape checks are different.
"""

import os, json, ast, re, time, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import mse_loss
from tqdm import tqdm
from scipy.io import savemat, loadmat
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# --------------------------- User configuration --------------------------- #
SPLIT_DIR   = "results_inverse_llama/run_20250506_040909"   # ← replace run folder
MODEL_DIR   = f"{SPLIT_DIR}/llama_ft"                      # auto-loaded adapters
OUTPUT_MAT  = os.path.join(SPLIT_DIR, "evaluation_predictions.mat")

BATCH_SIZE      = 24    # generation batch size
BUFFER_TOKENS   = 0     # safety buffer for max_new_tokens
torch.cuda.empty_cache()
# ------------------------------------------------------------------------- #

# Alpaca-style prompt builder (unchanged)
ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
)
def build_prompt(ex):
    return ALPACA_PROMPT.format(ex["instruction"], ex["input"], "")


# ------------------------------  Helpers  -------------------------------- #
def extract_spectrum(instruction: str) -> np.ndarray:
    """
    Pull the 31-value transmission spectrum out of the instruction *string*.
    Assumes the first bracketed list is the spectrum.
    """
    m = re.search(r"\[([^\]]+)\]", instruction)
    if not m:
        raise ValueError(f"Could not find spectrum in:\n{instruction}")
    vals = [float(x.strip()) for x in m.group(1).split(",")]
    if len(vals) != 31:
        raise ValueError(f"Spectrum length ≠ 31: got {len(vals)}")
    return np.array(vals, dtype=np.float32)


def parse_grid(text: str) -> np.ndarray | None:
    """
    Find the first “[ … ]” in *text*, interpret it as a 4×4 grid, and return a
    flat 16-element array (row major).  Return None if parsing fails.
    """

    r = text.find('Response:')
    s = text.find('[', r)
    e = text.find('].', s)
    if s == -1 or e == -1:
        return None
    try:
        grid = ast.literal_eval(text[s:e + 1])
        arr  = np.array(grid).reshape(-1)
    except Exception:
        return None
    return arr if arr.size == 16 and np.all((0.0 <= arr) & (arr <= 1.0)) else None


def determine_max_new_tokens(tokenizer, prompt: str, max_seq_length: int,
                              buffer: int = 0) -> int:
    ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    return max_seq_length - ids.size(0) - buffer


# -------------------------- Generation core ------------------------------ #
def batch_generate(prompts, true_grids, spectra, tokenizer, model,
                   max_new_tokens, batch_size=16):
    """
    • Generate in batches.
    • Parse each 4×4 grid; if parsing fails or shape wrong, queue for retries.
    • Compute MSE (pred vs true grid) on successes.
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompts, return_tensors='pt', padding=True,
                    truncation=True).to(device)

    preds, mse_list, tru_grids, tru_specs = [], [], [], []
    failed_prompts, failed_truths, failed_specs = [], [], []

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for b in tqdm(range(n_batches), desc="Generating", unit="batch"):
            s, e = b * batch_size, (b + 1) * batch_size
            out_ids = model.generate(**{k: v[s:e] for k, v in enc.items()},
                                     max_new_tokens=max_new_tokens)
            texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

            for i, txt in enumerate(texts):
                # print(txt)
                # print()
                g = parse_grid(txt)
                idx = s + i

                # raise
                if g is None:
                    failed_prompts.append(prompts[idx])
                    failed_truths.append(true_grids[idx])
                    failed_specs.append(spectra[idx])
                    continue
                # print('Extracted grid: ', g.tolist())
                # print('Corresponding spectrum: ', spectra[idx].tolist())
                preds.append(g)
                tru_grids.append(true_grids[idx])
                tru_specs.append(spectra[idx])
                # mse_list.append(float(mse_loss(torch.tensor(g), torch.tensor(true_grids[idx]))))

    # One-by-one retries for failures
    for p, tgrid, spec in tqdm(zip(failed_prompts, failed_truths, failed_specs),
                               total=len(failed_prompts),
                               desc="Re-generating failures", unit="item"):
        single = tokenizer(p, return_tensors="pt").to(device)
        for _ in range(100):                          # up to 100 attempts
            out = model.generate(**single, max_new_tokens=max_new_tokens)
            g   = parse_grid(tokenizer.decode(out[0], skip_special_tokens=True))
            if g is not None:
                preds.append(g)
                tru_grids.append(tgrid)
                tru_specs.append(spec)
                # mse_list.append(float(mse_loss(torch.tensor(g), torch.tensor(tgrid))))
                break

    return (np.stack(tru_grids),             # C (ground-truth grids)
            np.stack(tru_specs),             # G (input spectra)
            np.stack(preds),                 # P (predicted grids)
            mse_list)


# ---------------------------  Split driver  ------------------------------ #
def evaluate_split(name: str, model, tokenizer,
                   max_new_tokens: int, batch_size: int):
    f = os.path.join(SPLIT_DIR, f"{name}.json")
    ds = load_dataset("json", data_files={name: f})[name]

    prompts  = [build_prompt(ex) for ex in ds]
    spectra  = [extract_spectrum(ex["instruction"]) for ex in ds]

    true_grd = []
    truths = [ex['output'] for ex in ds]
    for text in truths:
        spec_start = text.find('[')
        spec_end = text.find('].')
        spec_str = text[spec_start + 1: spec_end]
        if spec_start == -1 or spec_end == -1:
            raise ValueError(f"Could not locate grid in:\n{text}")
        inner = text[spec_start + 1: spec_end]  # strip the leading “[” … keep trailing “]”
        nested = ast.literal_eval(f"[{inner}]")  # ← wrap & parse → [[row1], [row2], ...]
        flat = np.asarray(nested, dtype=np.float32).reshape(-1)
        if flat.size != 16:
            raise ValueError(f"Bad grid length ({flat.size}) in:\n{text}")
        true_grd.append(flat)
    # true_grd = [parse_grid(ex) for ex in prompts]

    C, G, P, mse = batch_generate(prompts, true_grd, spectra, tokenizer, model, max_new_tokens, batch_size)
    return C, G, P, mse


# ------------------------------  main  ----------------------------------- #
def main() -> None:
    # Self-copy for reproducibility
    try:
        shutil.copy(__file__, os.path.join(SPLIT_DIR, Path(__file__).name))
    except Exception:
        pass

    # Recover max_seq_length / lora_rank from training_stats.mat
    cfg = loadmat(os.path.join(SPLIT_DIR, "training_stats.mat"))["config"][0, 0]
    max_seq_length = int(cfg["max_seq_length"][0, 0])
    lora_rank      = int(cfg["lora_rank"][0, 0])

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_DIR,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.99,
    )
    FastLanguageModel.for_inference(model)

    # Determine max_new_tokens once from a dummy prompt
    dummy = build_prompt(json.load(open(os.path.join(SPLIT_DIR, "train.json")))[0])
    max_new_tokens = determine_max_new_tokens(tokenizer, dummy,
                                              max_seq_length, BUFFER_TOKENS)
    print("max_new_tokens =", max_new_tokens)

    # Evaluate test split
    t0 = time.perf_counter()
    C, G, P, mse = evaluate_split("test", model, tokenizer,
                                  max_new_tokens, BATCH_SIZE)
    t1 = time.perf_counter()

    # Save .mat
    savemat(OUTPUT_MAT, {
        "true_CPs" : C,          # ground-truth grids
        "target_spectra" : G,          # input spectra
        "predicted_CPs" : P,          # predicted grids
        "time_used" : t1 - t0,
    })
    # print(f"MSE (test) = {np.mean(mse):.6f}")
    print("Saved results to", OUTPUT_MAT)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
