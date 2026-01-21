import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import ast
import re
import numpy as np
import unsloth
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from multiprocessing import Pool
from scipy.io import savemat, loadmat
import torch
from torch.nn.functional import mse_loss
from tqdm import tqdm
from pathlib import Path
import shutil
import time


# ---------------------------
# User configuration
# ---------------------------
# Path to the folder containing the fine-tuned model (contains config, pytorch_model.bin, adapter files)
SPLIT_DIR = 'results_gpt_oss_20b/run_20250819_200023'
MODEL_DIR = SPLIT_DIR + '/llama_ft'
# Path to the folder containing validation.json and test.json

# Output .mat file
OUTPUT_MAT = os.path.join(SPLIT_DIR, "evaluation_predictions.mat")
# Batch size for generation
BATCH_SIZE = 64
BUFFER_TOKENS = 0  # Safety buffer when computing max_new_tokens

# ───────────────────────────────────────────────────────────────────── #
# helper: make Alpaca-style prompt (same as during training)
ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
)

def build_prompt(ex):
    return ALPACA_PROMPT.format(ex["instruction"], ex["input"], "")   # output left blank

# ---------------------------
# Utilities
# ---------------------------
def extract_grid(instruction: str) -> np.ndarray:
    """
    Extract the 4x4 grid from the instruction string and return as a flat array of length 16.
    Assumes instruction contains 'grid as: [[...]]'.
    """
    m = re.search(r"grid: (\[\[.*?\]\])", instruction)
    if not m:
        raise ValueError(f"Could not parse grid from: {instruction}")
    grid = ast.literal_eval(m.group(1))
    return np.array(grid).reshape(-1)


def make_prompt(example: dict) -> str:
    prompt = f"Instruction: {example['instruction']}\n"
    if example.get('input', '').strip():
        prompt += f"Input: {example['input']}\n"
    prompt += "\nResponse: "
    return prompt


def determine_max_new_tokens(tokenizer, prompt: str, max_seq_length: int, buffer: int=0) -> int:
    ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    return max_seq_length - ids.size(0) - buffer


def batch_generate(prompts, model, true_spectrum, grids, tokenizer, max_new_tokens, batch_size=16):
    """
    1) Fire off all prompts in nice big batches.
    2) Any time we can’t do the simple `float(x) for x in spec_str.split(",")`
       *or* the length doesn’t match, we record that prompt (and its truth/grid).
    3) After the big pass, we walk through each failed prompt one at a time,
       re-generating until it finally yields a clean “[num, num, …]” of the right length.
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}

    all_spectra, mse_list, tsp, cps = [], [], [], []

    # buffers for the ones that failed to parse cleanly
    failed_prompts, failed_truths, failed_grids = [], [], []

    num_batches = (len(prompts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating", unit="batch"):
            start = batch_idx * batch_size
            end   = start + batch_size
            batch_enc = {k: v[start:end] for k, v in enc.items()}
            out_ids = model.generate(**batch_enc, max_new_tokens=max_new_tokens)
            texts  = tokenizer.batch_decode(out_ids.cpu(), skip_special_tokens=True)
            mse_temp = []
            for i, text in enumerate(texts):
                print(text)
                global_idx = start + i
                # locate the '[' ... ']'
                r = text.find('Response:')
                s = text.find('[', r)
                e = text.find(']', s)
                # must exist and parse cleanly
                if s == -1 or e == -1:
                    failed_prompts.append(prompts[global_idx])
                    failed_truths.append(true_spectrum[global_idx])
                    failed_grids.append(grids[global_idx])
                    continue

                spec_str = text[s+1:e]
                try:
                    nums = [float(x.strip()) for x in spec_str.split(',')]
                except ValueError:
                    # NO fallback parsing — treat as failure
                    failed_prompts.append(prompts[global_idx])
                    failed_truths.append(true_spectrum[global_idx])
                    failed_grids.append(grids[global_idx])
                    continue

                if len(nums) != len(true_spectrum[global_idx]) or np.any(np.array(nums) > 1) or np.any(np.array(nums) < 0):
                    failed_prompts.append(prompts[global_idx])
                    failed_truths.append(true_spectrum[global_idx])
                    failed_grids.append(grids[global_idx])
                    continue

                # clean success
                all_spectra.append(nums)
                tsp.append(true_spectrum[global_idx])
                cps.append(grids[global_idx])
                mse_list.append(float(mse_loss(torch.Tensor(nums),
                                             torch.Tensor(true_spectrum[global_idx]))))
                mse_temp.append(mse_list[-1])
            print('MSE for this batch is ', np.mean(mse_temp))
    num_failed=len(failed_prompts)
    # one-by-one re-generation of the failures
    for p, truth, grid in tqdm(
            zip(failed_prompts, failed_truths, failed_grids),
            total=len(failed_prompts),
            desc="Re-generating failures",
            unit="prompt"
        ):
        single_enc = tokenizer(p, return_tensors='pt').to(device)
        temp_counter = 0
        while True:
            temp_counter = temp_counter + 1
            if temp_counter >= 100:
                break
            with torch.no_grad():
                try:
                 out_id = model.generate(**single_enc, max_new_tokens=max_new_tokens)
                except RuntimeError:
                    # print(prompts)
                    continue
            txt = tokenizer.decode(out_id[0], skip_special_tokens=True)
            print(txt)
            r = txt.find('Response:')
            s = txt.find('[', r)
            e = txt.find(']', s)
            if s == -1 or e == -1:
                continue

            spec_str = txt[s+1:e]
            try:
                nums = [float(x.strip()) for x in spec_str.split(',')]
            except ValueError:
                continue

            if len(nums) != len(truth) or np.any(np.array(nums) > 1) or np.any(np.array(nums) < 0):
                continue

            # finally got it
            all_spectra.append(nums)
            tsp.append(truth)
            cps.append(grid)
            mse_list.append(float(mse_loss(torch.Tensor(nums), torch.Tensor(truth))))
            break

    return all_spectra, mse_list, tsp, cps, num_failed



def generate_spectrum(text: str) -> np.ndarray:
    try:
        return np.array(ast.literal_eval(text.strip()))
    except Exception:
        return None


def evaluate_split(split_name: str, model, tokenizer, max_new_tokens, batch_size: int):
    # Load split
    path = os.path.join(SPLIT_DIR, f"{split_name}.json")
    ds = load_dataset('json', data_files={split_name: path})[split_name]
    # Extract inputs
    prompts = [build_prompt(ex) for ex in ds]
    grids = [extract_grid(ex['instruction']) for ex in ds]
    truths = [ex['output'] for ex in ds]
    true_spectrum=[]
    for text in truths:
        spec_start = text.find('[')
        spec_end = text.find(']')
        spec_str = text[spec_start + 1: spec_end]
        nums = [float(x.strip()) for x in spec_str.split(",")]
        true_spectrum.append(nums)

    # Generate all responses in batches
    pds, mse, tsp, cps, num_failed = batch_generate(prompts, model, true_spectrum, grids, tokenizer, max_new_tokens, batch_size)

    C = np.stack(cps)
    G = np.stack(tsp)
    P = np.stack(pds)
    return C, G, P, mse, num_failed


def main():

    # ───────────────────── Self-copy for reproducibility ─────────────────── #
    try:
        this_file = Path(__file__)
        shutil.copy(this_file, os.path.join(SPLIT_DIR, os.path.basename(__file__)))
    except NameError:  # interactive / notebook
        pass  # ignore – no __file__

    # 1) read training hyper-params to recover max_seq_length
    cfg = loadmat(os.path.join(SPLIT_DIR, "training_stats.mat"))["config"][0,0]
    max_seq_length = int(cfg["max_seq_length"][0,0])
    lora_rank=int(cfg["lora_rank"][0,0])

    # 1. Load fine-tuned model + tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=max_seq_length,
        #device_map="auto",
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank = lora_rank,
        # offload_folder=os.path.join(output_dir, "offload"),
        gpu_memory_utilization = 0.99,
    )
    FastLanguageModel.for_inference(model)
    # LoRA adapters have been saved in MODEL_DIR, auto-loaded by FastLanguageModel

    # 3) compute max_new_tokens once
    dummy_prompt = build_prompt(json.load(open(os.path.join(SPLIT_DIR, "train.json")))[0])
    prompt_len = len(tokenizer(dummy_prompt).input_ids)
    max_new_tokens = max_seq_length - prompt_len
    print("max_new_tokens = ", max_new_tokens)

    # 3. Evaluate validation and test
    start = time.perf_counter()
    test_C, test_G, test_P, mse_test, num_failed = evaluate_split("test", model, tokenizer, max_new_tokens, BATCH_SIZE)
    end = time.perf_counter()

    # 4. Save to .mat
    savemat(OUTPUT_MAT, {
        'test_CPs': test_C,
        'test_ground_truth': test_G,
        'test_predictions': test_P,
        'MSE_test': mse_test,
        'MSE_test_mean': np.mean(mse_test),
        'time_used': end - start,
        'num_failed': num_failed
    })

    torch.cuda.empty_cache()
    print(f'MSE for test set is: {np.mean(mse_test)}')
    print(f"Saved evaluation results to {OUTPUT_MAT}")

if __name__ == '__main__':
    main()
