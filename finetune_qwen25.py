#!/usr/bin/env python3
# fine_tune_qwen25.py
"""
LoRA-fine-tune **Qwen-2.5-Chat** on a proprietary metasurface dataset
(4×4 grid  →  31-point transmission spectrum).

Unchanged from the Llama script:
  • grammar fix + 80/20 split + JSON snapshots
  • dynamic max_seq_length (+50 tokens head-room)
  • UnsLoTH LoRA config / memory stats / .mat artifacts
"""

import os, json, random, shutil, logging, math, re
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from scipy.io import savemat

# ───────────────────────────── Helpers ────────────────────────────── #
def fix_grammar(obj: Dict[str, str]) -> Dict[str, str]:
    """Patch the two typos + trailing period (exactly as before)."""
    inst, out = obj["instruction"], obj["output"]
    inst = re.sub(r"\bgrid\s+as\s*:", "grid:", inst, flags=re.I)
    out  = re.sub(r"\bvalues\s+[^[]*?\bis\s+\[",
                  lambda m: m.group(0).replace(" is ", " are "), out, flags=re.I)
    out  = out.rstrip() + ("" if out.rstrip().endswith((".", "!", "?")) else ".")
    return {"instruction": inst, "input": obj.get("input", ""), "output": out}

def qwen_chat(instruction: str, inp: str, answer: str, tokenizer) -> str:
    """One user–assistant turn (no <think> channel for Qwen-2.5)."""
    user = f"{instruction}\n\n### Input:\n{inp}" if inp.strip() else instruction
    conv = [
        {"role": "user",      "content": user},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(conv,
                                         tokenize=False,
                                         enable_thinking=False)

# ─────────────────────────────── Main ─────────────────────────────── #
def main() -> None:
    # --------------- configuration ---------------- #
    CFG: Dict[str, Any] = dict(
        seed            = 3407,
        dataset_path    = "simulation_data_31_least.json",
        train_split     = 0.80,
        base_model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",  # ← only change needed
        lora_rank       = 32,
        lora_alpha      = 32,
        lora_dropout    = 0.0,
        learning_rate   = 4e-4,
        batch_size      = 72,
        grad_accum      = 8,
        weight_decay    = 1e-2,
        epochs          = 8,
        max_steps       = 1000000,
        warmup_ratio    = 0.01,     # % of optimiser steps
    )

    OUTPUT_ROOT = "results_qwen25"
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    outd = Path(OUTPUT_ROOT) / f"run_{ts}"
    outd.mkdir(parents=True, exist_ok=True)

    # --------------- reproducibility snapshot ---------------- #
    try:
        shutil.copy(Path(__file__), outd / Path(__file__).name)
    except Exception:
        pass

    # --------------- logging ---------------- #
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(outd / "run.log", "w")],
    )
    logger = logging.getLogger(__name__)
    logger.info("Run ID: %s", ts)

    torch.manual_seed(CFG["seed"])
    random.seed(CFG["seed"])

    # --------------- load + patch data ---------------- #
    with open(CFG["dataset_path"], encoding="utf-8") as fh:
        data = [fix_grammar(x) for x in json.load(fh)]
    n      = len(data)
    train  = data[: int(n * CFG["train_split"])]
    test   = data[int(n * CFG["train_split"]):]
    for nm, split in [("train", train), ("test", test)]:
        with open(outd / f"{nm}.json", "w") as f:
            json.dump(split, f, indent=2)
    logger.info("Split: %d train ‖ %d test", len(train), len(test))

    # --------------- first tiny load just for tokenizer -------------- #
    model0, tok0 = FastLanguageModel.from_pretrained(
        CFG["base_model_name"], max_seq_length=32,
        load_in_4bit=True, dtype=None)
    EOS = tok0.eos_token

    sample = qwen_chat(train[0]["instruction"], train[0]["input"],
                       train[0]["output"], tok0) + EOS
    CFG["max_seq_length"] = len(tok0.encode(sample)) + 50
    logger.info("MAX_SEQ_LENGTH set to %d", CFG["max_seq_length"])
    del model0, tok0

    # --------------- final model load ---------------- #
    model, tokenizer = FastLanguageModel.from_pretrained(
        CFG["base_model_name"],
        max_seq_length = CFG["max_seq_length"],
        load_in_4bit   = True,
        dtype          = None,
        fast_inference = False,
        max_lora_rank  = CFG["lora_rank"],
        gpu_memory_utilization = 0.99,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                = CFG["lora_rank"],
        lora_alpha       = CFG["lora_alpha"],
        lora_dropout     = CFG["lora_dropout"],
        target_modules   = ["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
        bias             = "none",
        use_gradient_checkpointing = "unsloth",
        random_state     = CFG["seed"],
    )

    # --------------- dataset mapping ---------------- #
    def map_func(ex):
        return {"text": [
            qwen_chat(i, inp, out, tokenizer) + EOS
            for i, inp, out in zip(ex["instruction"], ex["input"], ex["output"])
        ]}
    train_ds = load_dataset(
        "json", data_files=str(outd / "train.json"), split="train"
    ).map(map_func, batched=True, remove_columns=["instruction","input","output"])

    # --------------- optimiser steps + warm-up ---------------- #
    eff_batch   = CFG["batch_size"] * CFG["grad_accum"]
    steps_epoch = math.ceil(len(train_ds) / eff_batch)
    total_steps = steps_epoch * CFG["epochs"]
    warm_steps  = max(1, int(total_steps * CFG["warmup_ratio"]))
    logger.info("Total steps = %d  •  Warm-up = %d (%.2f%%)",
                total_steps, warm_steps, 100*warm_steps/total_steps)

    # --------------- trainer ---------------- #
    args = TrainingArguments(
        per_device_train_batch_size = CFG["batch_size"],
        gradient_accumulation_steps = CFG["grad_accum"],
        learning_rate = CFG["learning_rate"],
        optim = "adamw_8bit",
        weight_decay = CFG["weight_decay"],
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        output_dir = str(outd),
        report_to = "none",
        seed = CFG["seed"],
        save_strategy = "epoch",
        num_train_epochs = CFG["epochs"],
        lr_scheduler_type = "linear",
        overwrite_output_dir = True,
        torch_compile = True,
        torch_compile_mode = "default",
        dataloader_pin_memory = True,
        dataloader_num_workers = 10,
        dataloader_prefetch_factor = 2,
        warmup_steps = warm_steps,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_ds,
        dataset_text_field = "text",
        max_seq_length = CFG["max_seq_length"],
        packing = False,
        args = args,
    )

    # --------------- memory stats before training ------------- #
    gpu = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved()/1024/1024/1024, 3)
    logger.info("GPU %s | Reserved %.3f GB / %.1f GB",
                gpu.name, start_mem,
                gpu.total_memory/1024/1024/1024)

    # --------------- train ------------------------------------ #
    logger.info("Training …")
    stats = trainer.train()
    logger.info("Training done.")

    # --------------- save LoRA adapters ----------------------- #
    lora_dir = outd / "qwen25_lora"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    for ck in outd.glob("checkpoint-*"):
        shutil.rmtree(ck, ignore_errors=True)

    # --------------- post-training stats ---------------------- #
    peak   = round(torch.cuda.max_memory_reserved()/1024/1024/1024, 3)
    delta  = round(peak - start_mem, 3)
    pct    = round(peak / (gpu.total_memory/1024/1024/1024) * 100, 3)
    pct_dl = round(delta / (gpu.total_memory/1024/1024/1024) * 100, 3)
    logger.info("%s s (%.2f min) total", stats.metrics["train_runtime"],
                                          stats.metrics["train_runtime"]/60)
    logger.info("Peak reserved mem %.3f GB (%.3f GB for training, %.2f %% / %.2f %%)",
                peak, delta, pct, pct_dl)

    # --------------- save loss curve -------------------------- #
    hist = [(lg["step"], lg["loss"]) for lg in trainer.state.log_history if "loss" in lg]
    savemat(outd / "training_stats.mat",
            {"config": CFG, "steps": [s for s,_ in hist],
             "losses": [l for _,l in hist]})
    logger.info("Artifacts saved to %s", outd)

# ────────────────────────── Entrypoint ────────────────────────────── #
if __name__ == "__main__":
    main()
