import os
# Silence Tokenizers parallelism warning and disable fused cross-entropy
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUT_CROSS_ENTROPY_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import json
import random
import datetime
import logging
import shutil
import ast
import numpy as np
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer
import torch
from scipy.io import savemat

# ---------------------------
# Configuration parameters
# ---------------------------
SEED = 42
LORA_RANK = 32
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NUM_EPOCHS = 3
PER_DEVICE_BATCH = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 3e-4
MAX_NEW_TOKENS = 300
OUTPUT_ROOT = "results"
DATASET_PATH = "simulation_data_31_least.json"

# ---------------------------
# Setup logging
# ---------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def preprocess_function(example):
    prompt = f"Instruction: {example['instruction']}\n"
    if example.get('input', '').strip():
        prompt += f"Input: {example['input']}\n"
    prompt += "\nResponse: "
    full_text = prompt + example['output']
    return {"text": full_text, "prompt": prompt, "target": example['output']}


def tokenize_function(examples, tokenizer, max_seq_length):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_seq_length,
    )


def generate_spectrum(text):
    try:
        return np.array(ast.literal_eval(text.strip()))
    except Exception:
        return None


def generate_response(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def save_loss_curve(trainer):
    steps, losses = [], []
    for log in trainer.state.log_history:
        if 'loss' in log:
            steps.append(log.get('step', None))
            losses.append(log['loss'])
    return steps, losses


def main():
    random.seed(SEED)

    # 1. Create output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_ROOT, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Copy script
    script_path = os.path.abspath(__file__)
    shutil.copy(script_path, os.path.join(output_dir, os.path.basename(__file__)))
    logger.info(f"Script copied to {output_dir}")

    # Load full dataset
    with open(DATASET_PATH, 'r') as f:
        full_data = json.load(f)
    random.shuffle(full_data)

    # Determine max_seq_length from 95th percentile
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    ex=full_data[0]
    text = (f"Instruction: {ex['instruction']}\n" +
            (f"Input: {ex['input']}\n" if ex['input'].strip() else "") +
            "\nResponse: " + ex['output'])
    max_len = int(len(tokenizer(text).input_ids)) + 50
    logger.info(f"Setting MAX_SEQ_LENGTH to {max_len}")

    # Split data
    n = len(full_data)
    train, val, test = full_data[:int(0.8*n)], full_data[int(0.8*n):int(0.9*n)], full_data[int(0.9*n):]
    for name, split in [('train', train), ('validation', val), ('test', test)]:
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(split, f, indent=2)
        logger.info(f"Saved {name} data to {path}")

    # Load splits
    ds = {name: load_dataset('json', data_files={name: os.path.join(output_dir, f"{name}.json")})[name]
          for name in ['train','validation','test']}
    for name in ds:
        ds[name] = ds[name].map(preprocess_function)

    # 2. Fine-tuning setup
    logger.info("Loading model with device_map=auto...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        #device_map="auto",
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=LORA_RANK,
        # offload_folder=os.path.join(output_dir, "offload"),
        gpu_memory_utilization=0.9,
    )
    # model = model.to(torch.cuda.current_device())

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Tokenization
    tokenized_train = ds['train'].map(lambda ex: tokenize_function(ex, tokenizer, max_len), batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'llama_ft'),
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=50,
        save_steps=200,
        fp16=True,
        dataloader_num_workers=8,
        report_to='none',
    )
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=tokenized_train, data_collator=data_collator)

    # Launch with torchrun for DDP if desired
    # e.g.: torchrun --nproc_per_node=4 finetune.py

    logger.info("Starting fine-tuning...")
    trainer.train()
    logger.info("Fine-tuning done.")

    # Save model & tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Model saved.")

    # Clean up checkpoint directories to save disk space
    ckpt_root = training_args.output_dir
    for item in os.listdir(ckpt_root):
        path = os.path.join(ckpt_root, item)
        if os.path.isdir(path) and item.startswith('checkpoint-'):
            shutil.rmtree(path)
            logger.info(f"Removed checkpoint {path}")

    results = {
        'loss_steps': save_loss_curve(trainer)[0],
        'loss_values': save_loss_curve(trainer)[1],
        'hyperparams': {
            'epochs': NUM_EPOCHS,
            'batch': PER_DEVICE_BATCH,
            'accum': GRAD_ACCUM_STEPS,
            'lr': LEARNING_RATE,
            'lora_rank': LORA_RANK,
            'max_seq_length': max_len,
            'max_new_tokens': MAX_NEW_TOKENS,
        }
    }
    savemat(os.path.join(output_dir, f"results.mat"), results)
if __name__ == '__main__':
    main()
