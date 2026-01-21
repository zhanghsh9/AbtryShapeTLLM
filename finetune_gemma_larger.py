#!/usr/bin/env python3
# fine_tune_metasurface.py
"""
Fine-tune Meta-Llama-3.1-8B-Instruct (LoRA) to map a 4×4 metasurface grid
→ 31-point transmission spectrum.

Changes v2:
  • Self-copies at start (source_snapshot.py) for perfect reproducibility.
  • Wrapped main logic in  if __name__ == "__main__".
"""

import os, json, random, shutil, logging, math, re
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
from scipy.io import savemat


def split_data(data: List[Dict[str, str]], p_train: float):
    random.shuffle(data)
    k = int(len(data) * p_train)
    return data[:k], data[k:]


def fix_grammar(obj: Dict[str, str]) -> Dict[str, str]:
    """Return a *new* dict with the instruction/output corrected."""
    inst = obj["instruction"]
    out = obj["output"]

    # ---- 1) Delete the spurious “as” in the instruction --------------
    inst = re.sub(r"\bgrid\s+as\s*:", "grid:", inst, flags=re.I)

    # ---- 2) Change “… values … is [” → “… values … are [” -------------
    out = re.sub(r"\bvalues\s+[^[]*?\bis\s+\[",
                 lambda m: m.group(0).replace(" is ", " are "),
                 out, flags=re.I)

    # ---- 3) Ensure the output ends with a period ----------------------
    out = out.rstrip()
    if not out.endswith((".", "!", "?")):  # already punctuated?
        out += "."

    fixed = dict(obj)
    fixed["instruction"] = inst
    fixed["output"] = out
    return fixed

def main() -> None:
    # ─────────────────────────── Configuration ──────────────────────────── #
    CFG: Dict[str, Any] = dict(
        seed=3407,
        dataset_path="simulation_data_31_least.json",
        train_split=0.80,  # 4 : 1
        base_model_name="unsloth/gemma-2-27b-it-bnb-4bit",
        lora_rank=16,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=1e-4,
        batch_size=64,
        grad_accum=3,
        max_steps=1000000,
        weight_decay=1e-2,
        epochs=8,
    )

    # Derived paths
    OUTPUT_ROOT = "results_gemma2"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(OUTPUT_ROOT) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ───────────────────── Self-copy for reproducibility ─────────────────── #
    try:
        this_file = Path(__file__)
        shutil.copy(this_file, os.path.join(output_dir, os.path.basename(__file__)))
    except NameError:  # interactive / notebook
        pass  # ignore – no __file__

    # ─────────────────────────────────────────────────────────────────────── #

    # ===== Logging ==================================================== #
    log_file = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, "w")],
    )
    logger = logging.getLogger(__name__)
    logger.info("Run ID: %s", timestamp)

    # ===== Environment & Seed ========================================= #
    torch.manual_seed(CFG["seed"])

    # ===== Data load / split ========================================== #


    with open(CFG["dataset_path"], 'r', encoding="utf-8") as f:
        full_data = json.load(f)
    '''
    train_samples, test_samples = split_data(all_samples, CFG["train_split"])
    for nm, blob in [("train", train_samples), ("test", test_samples)]:
        with open(output_dir / f"{nm}.json", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in blob)
    '''
    full_data = [fix_grammar(x) for x in full_data]  # ← overwrite with patched entries
    n = len(full_data)
    train, test = full_data[:int(CFG['train_split'] * n)], full_data[int(CFG['train_split'] * n):]
    for name, split in [('train', train), ('test', test)]:
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(split, f, indent=2)
        logger.info(f"Saved {name} data to {path}")
    logger.info("Split: %d train ‖ %d test", len(train), len(test))

    # ===== Prompt util & token length ================================= #
    alpaca_prompt = (
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. Write a response that appropriately "
        "completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n"
        "### Response:\n{}"
    )
    '''
    tmp_tok = FastLanguageModel.from_pretrained(
        CFG["base_model_name"],
        max_seq_length=32,
        load_in_4bit=True,
        dtype=None
    )[1]

    prompt_tok_len = len(tmp_tok.encode(
        alpaca_prompt.format(*(train[0][k] for k in ("instruction", "input", "output")))
    ))
    logger.info("Prompt tokens=%s  → max_seq_length=%s", prompt_tok_len, CFG["max_seq_length"])
    del tmp_tok
    '''
    # Determine max_seq_length from 95th percentile
    tokenizer = AutoTokenizer.from_pretrained(CFG["base_model_name"], use_fast=True)
    text = alpaca_prompt.format(*(full_data[0][k] for k in
                                ("instruction", "input", "output"))) + tokenizer.eos_token
    max_len = int(len(tokenizer.encode(text))) + 50
    logger.info(f"Setting MAX_SEQ_LENGTH to {max_len}")
    CFG["max_seq_length"] = max_len


    # ===== Model + LoRA =============================================== #
    model, tokenizer = FastLanguageModel.from_pretrained(
        CFG["base_model_name"],
        max_seq_length=CFG["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
        fast_inference=False,
        max_lora_rank=CFG['lora_rank'],
        gpu_memory_utilization=1,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=CFG["lora_rank"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CFG["seed"],
    )

    EOS = tokenizer.eos_token
    def fmt(ex):
        return {"text": [
            alpaca_prompt.format(i, inp, out) + EOS
            for i, inp, out in zip(ex["instruction"], ex["input"], ex["output"])
        ]}

    train_ds = load_dataset( "json",  data_files=str(output_dir / "train.json"),
                             split="train").map(fmt, batched=True, remove_columns=["instruction", "input", "output"])

    # ===== Trainer ==================================================== #
    args = TrainingArguments(
        per_device_train_batch_size=CFG["batch_size"],
        gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=CFG["learning_rate"],
        optim="adamw_8bit",
        weight_decay=CFG["weight_decay"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        output_dir=str(output_dir),
        report_to="none",
        seed=CFG["seed"],
        save_strategy="epoch",
        num_train_epochs=CFG["epochs"],
        lr_scheduler_type="linear",
        overwrite_output_dir=True,
        torch_compile=True,
        torch_compile_mode="default",
        dataloader_pin_memory=True,
        dataloader_num_workers=10,
        dataloader_prefetch_factor=2,
        warmup_steps=5,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=CFG["max_seq_length"],
        packing=False,
        args=args,
    )

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")

    logger.info("Starting training …")
    trainer_stats = trainer.train()
    logger.info("Training done.")

    # ===== Save ======================================================= #
    lora_dir = output_dir / "llama_ft"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    for ck in output_dir.glob("checkpoint-*"):
        shutil.rmtree(ck, ignore_errors=True)

    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logger.info(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    logger.info(f"Peak reserved memory = {used_memory} GB.")
    logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # ===== Loss curve → .mat ========================================== #
    history = [(log["step"], log["loss"])
               for log in trainer.state.log_history if "loss" in log]
    savemat(output_dir / "training_stats.mat",
            {"config": CFG, "steps": [s for s, _ in history],
             "losses": [l for _, l in history]})

    logger.info("Artifacts saved in  %s", output_dir)

# ──────────────────────────── Entrypoint ────────────────────────────── #
if __name__ == "__main__":
    main()
