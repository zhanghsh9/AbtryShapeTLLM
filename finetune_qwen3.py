#!/usr/bin/env python3
# fine_tune_qwen3.py
"""
Fine‑tune **Qwen3‑14B** (LoRA, *non‑reasoning* mode only) on a proprietary
metasurface dataset.  No external corpora are used.  Training rows are created
in Qwen chat format, saved to JSON‑Lines, and loaded with `datasets.load_dataset`.
The sequence length is chosen dynamically (first sample length +50) exactly as
requested.
"""

import os, json, random, shutil, logging, sys, subprocess, re, math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments, AutoTokenizer
from trl import SFTTrainer
from scipy.io import savemat

# ─────────────────── Debug printing callback ──────────────────── #
from transformers import TrainerCallback, TrainerControl, TrainerState

class ShowExampleCallback(TrainerCallback):
    """
    Print one example at the beginning of every epoch:
      • Raw prompt text after chat-templating
      • What the current model would generate from that prompt
    """
    def __init__(self, tokenizer, prompt_texts, max_new_tokens: int = 64):
        self.tokenizer = tokenizer
        self.prompt_texts = prompt_texts          # a *list* of finished prompts
        self.max_new_tokens = max_new_tokens

    def on_epoch_begin(self, args, state: TrainerState,
                       control: TrainerControl, **kwargs):
        # Always take the same (first) prompt so comparisons are fair
        sample_prompt = self.prompt_texts[0]
        prompt_ids    = self.tokenizer.encode(sample_prompt,
                                              return_tensors="pt").to(kwargs["model"].device)

        # Greedy generation so it’s fast and deterministic
        with torch.no_grad():
            gen_ids = kwargs["model"].generate(
                prompt_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]

        seen_text   = self.tokenizer.decode(prompt_ids[0],
                                            skip_special_tokens=False)
        output_text = self.tokenizer.decode(gen_ids,
                                            skip_special_tokens=False)

        print("\n──────── BEGIN DEBUG EXAMPLE (epoch %s) ────────", state.epoch)
        print("» Model *input* (what it sees):\n%s", seen_text)
        print("» Model *output* (current generation):\n%s", output_text)
        print("───────────────────────────────────────────────\n")



def main() -> Path:
    # ───────────────────────── Configuration ────────────────────────── #
    CFG: Dict[str, Any] = dict(
        seed=3407,
        train_split=0.80,          # metasurface data 4:1
        dataset_path="simulation_data_31_least.json",  # metasurface JSON
        base_model_name="unsloth/Qwen3-8B-unsloth-bnb-4bit",
        lora_rank=32,
        lora_alpha=32,
        lora_dropout=0.0,
        learning_rate=4e-4,
        batch_size=12,
        grad_accum=8,
        max_steps=1000000,
        weight_decay=1e-2,
        epochs=5,
        warmup_ratio=0.01,
    )

    # Derived paths
    OUTPUT_ROOT = "results_qwen3"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(OUTPUT_ROOT) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ───────────────────── Self‑copy for reproducibility ─────────────────── #
    try:
        this_file = Path(__file__)
        shutil.copy(this_file, output_dir / this_file.name)
    except NameError:
        pass  # interactive / notebook

    # ───────────────────────────── Logging ──────────────────────────────── #
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "run.log", "w")],
    )
    logger = logging.getLogger(__name__)
    logger.info("Run ID: %s", timestamp)

    torch.manual_seed(CFG["seed"])
    random.seed(CFG["seed"])

    # ─────────────────────── Load metasurface data ──────────────────────── #
    with open(CFG["dataset_path"], "r", encoding="utf-8") as fh:
        data = json.load(fh)

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

    data = [fix_grammar(x) for x in data]  # ← overwrite with patched entries
    n = len(data)
    train_meta = data[: int(CFG["train_split"] * n)]
    test_meta  = data[int(CFG["train_split"] * n):]
    logger.info("Metasurface split: %d train ‖ %d test", len(train_meta), len(test_meta))

    # ───────────────────────── Tokenizer setup ──────────────────────────── #
    tokenizer = AutoTokenizer.from_pretrained(CFG["base_model_name"], use_fast=True)

    # Qwen chat template helper
    def format_chat(u: str, a: str) -> str:
        conv = [
            {"role": "user",      "content": u},
            {"role": "assistant", "content": a},
        ]
        return tokenizer.apply_chat_template(conv, tokenize=False, enable_thinking=False)

    # Build chat‑formatted lines for training set
    # train_text: List[str] = [format_chat(str(s["input"]), str(s["output"])) for s in train_meta]
    # test_text: List[str] = [format_chat(str(s["input"]), str(s["output"])) for s in test_meta]

    # ─────────────── Build chat-formatted lines for training set ────────── #
    def build_user_prompt(s: Dict[str, Any]) -> str:
        """
        Re-create the Alpaca-style (instruction, optional input) prompt,
        but expressed as a single *user* message for Qwen chat format.
        """
        instr = str(s["instruction"]).strip()
        inp = str(s.get("input", "")).strip()

        # If the dataset entry has “input”, append it under a heading so the
        # model can distinguish the two parts.  Otherwise just use the instruction.
        if inp:
            return f"{instr}\n\n### Input:\n{inp}"
        return instr

    train_text: List[str] = [format_chat(build_user_prompt(s), str(s["output"])) for s in train_meta]
    test_text: List[str] = [format_chat(build_user_prompt(s), str(s["output"])) for s in test_meta]

    # Determine max_seq_length from first example length +50
    sample_len = len(tokenizer.encode(train_text[0] + tokenizer.eos_token)) + 50
    CFG["max_seq_length"] = sample_len
    logger.info("Setting MAX_SEQ_LENGTH to %d", sample_len)

    # Write JSON‑Lines file for datasets.load_dataset
    train_jsonl = output_dir / "train_chat.jsonl"
    with open(train_jsonl, "w", encoding="utf-8") as fh:
        for txt in train_text:
            fh.write(json.dumps({"text": txt}) + "\n")
    logger.info("Saved chat training file → %s", train_jsonl)

    test_jsonl = output_dir / "test_chat.jsonl"
    with open(test_jsonl, "w", encoding="utf-8") as fh:
        for txt in test_text:
            fh.write(json.dumps({"text": txt}) + "\n")
    logger.info("Saved chat training file → %s", test_jsonl)

    # Load with datasets.load_dataset
    train_ds = load_dataset("json", data_files=str(train_jsonl), split="train")

    # ──────────────── derive total optimiser steps & warm-up ──────────────── #
    effective_batch = CFG["batch_size"] * CFG["grad_accum"]  # one-GPU run
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
    total_steps = steps_per_epoch * CFG["epochs"]
    warmup_steps = max(1, int(total_steps * CFG["warmup_ratio"]))  # ≥1

    logger.info("Total optimiser steps = %d  •  Warm-up = %d (≈ %.2f %%)",
                total_steps, warmup_steps,
                100 * warmup_steps / total_steps)

    # ─────────────────────── Model + LoRA setup ─────────────────────────── #
    model, tokenizer = FastLanguageModel.from_pretrained(
        CFG["base_model_name"],
        max_seq_length=CFG["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
        fast_inference=False,
        max_lora_rank=CFG["lora_rank"],
        gpu_memory_utilization=0.99,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=CFG["lora_rank"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CFG["seed"],
    )

    # ────────────────────────── Training setup ──────────────────────────── #
    train_args = TrainingArguments(
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
        # save_steps=50,
        num_train_epochs=CFG["epochs"],
        lr_scheduler_type="linear",
        overwrite_output_dir=True,
        torch_compile=True,
        torch_compile_mode="default",
        dataloader_pin_memory=True,
        dataloader_num_workers=10,
        dataloader_prefetch_factor=2,
        warmup_steps=warmup_steps,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=CFG["max_seq_length"],
        packing=False,
        args=train_args,
        # callbacks=[ShowExampleCallback(tokenizer, train_text)],
    )

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")

    # ───────────────────────────── Training ─────────────────────────────── #
    logger.info("Starting training …")
    trainer_stats = trainer.train()
    logger.info("Training complete.")

    # ─────────────────────────── Saving artifacts ───────────────────────── #
    lora_dir = output_dir / "qwen3_lora"
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

    # Save losses for post‑analysis
    hist = [(log["step"], log["loss"]) for log in trainer.state.log_history if "loss" in log]
    savemat(output_dir / "training_stats.mat", {
        "config": CFG,
        "steps": [s for s, _ in hist],
        "losses": [l for _, l in hist],
    })
    logger.info("Artifacts saved to %s", output_dir)
    torch.cuda.empty_cache()
    return output_dir


# ──────────────────────────── Entrypoint ────────────────────────────── #
if __name__ == "__main__":
    output_dir = main()
    # kick off eval_qwen3.py
    torch.cuda.empty_cache()
    cmd = [sys.executable, "eval_qwen3.py",
           "--split_dir", str(output_dir),
           "--cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", "")]
    subprocess.run(cmd, check=True)
