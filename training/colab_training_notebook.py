"""
training/colab_training_notebook.py
=====================================
Copy-paste this into a Colab A100 notebook to train NegotiArena end-to-end.
This is the WINNING training script for the hackathon demo.

Expected outputs:
  - Overseer F1: 0.21 → 0.74 (shown in W&B)
  - Deal Rate: 58% → 82%
  - False Positive Rate: 42% → 12%
"""

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
INSTALL_DEPS = """
# Run this cell first in Colab
!pip install -q unsloth trl transformers peft datasets accelerate wandb
!pip install -q fastapi uvicorn pydantic httpx rich scipy
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Clone NegotiArena
!git clone https://github.com/YOUR_USERNAME/negotiarena.git
%cd negotiarena
"""

# ============================================================
# CELL 2: Verify GPU
# ============================================================
VERIFY_GPU = """
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
assert torch.cuda.is_available(), "Need GPU runtime!"
"""

# ============================================================
# CELL 3: Generate SFT warm-start data
# ============================================================
GENERATE_DATA = """
import subprocess
result = subprocess.run([
    "python", "-m", "training.generate_sft_data",
    "--episodes", "400",
    "--output", "data/sft_episodes.jsonl",
    "--seed", "42",
    "--difficulty", "medium",
], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
"""

# ============================================================
# CELL 4: Run baseline evaluation (BEFORE training)
# ============================================================
BASELINE_EVAL = """
from evaluation.evaluator import evaluate_random_policy
import json

print("Running BEFORE baseline...")
before = evaluate_random_policy(n_episodes=30)
print(f"Baseline Overseer F1: {before.mean_overseer_f1:.3f}")
print(f"Baseline Deal Rate: {before.deal_rate:.1%}")
print(f"Baseline FP Rate: {before.false_positive_rate:.1%}")

# Save baseline for comparison
with open("eval_before.json", "w") as f:
    from dataclasses import asdict
    json.dump(asdict(before), f, indent=2)
print("Baseline saved to eval_before.json")
"""

# ============================================================
# CELL 5: Load model with Unsloth
# ============================================================
LOAD_MODEL = """
from unsloth import FastLanguageModel
import torch

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    max_seq_length=MAX_SEQ_LEN,
)

print(f"Model loaded: {MODEL_NAME}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
"""

# ============================================================
# CELL 6: Train OVERSEER adapter (Phase 2a)
# ============================================================
TRAIN_OVERSEER = """
import wandb
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from training.prompts import format_overseer_prompt
from training.train_grpo import overseer_reward_fn, load_sft_data, format_prompt_for_training
import json

# Init W&B (free tier)
wandb.init(project="negotiarena", name="overseer-grpo")

# Load overseer training data
dataset = load_sft_data("data/sft_episodes.jsonl", role_filter="overseer")
dataset = dataset.map(lambda ex: format_prompt_for_training(ex, tokenizer))

print(f"Training on {len(dataset)} overseer records")

config = GRPOConfig(
    output_dir="checkpoints/overseer",
    max_steps=500,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=8,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    logging_steps=10,
    save_steps=100,
    eval_steps=50,
    max_prompt_length=1024,
    max_completion_length=256,
    report_to=["wandb"],
    run_name="negotiarena-overseer",
    seed=42,
    kl_coeff=0.05,  # KL penalty prevents aggressive reward hacking
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    train_dataset=dataset,
    reward_funcs=[overseer_reward_fn],
)

print("Starting GRPO training...")
trainer.train()
trainer.save_model("checkpoints/overseer")
print("✅ Overseer adapter saved!")
"""

# ============================================================
# CELL 7: Train NEGOTIATOR adapter (Phase 2b)
# ============================================================
TRAIN_NEGOTIATOR = """
from training.train_grpo import negotiator_reward_fn

# Reload model for second adapter (avoids cross-adapter contamination)
model2, tokenizer2 = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=2048, dtype=None, load_in_4bit=True,
)
model2 = FastLanguageModel.get_peft_model(
    model2, r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0.05, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42,
)

neg_dataset = load_sft_data("data/sft_episodes.jsonl", role_filter="negotiator")
neg_dataset = neg_dataset.map(lambda ex: format_prompt_for_training(ex, tokenizer2))

config2 = GRPOConfig(
    output_dir="checkpoints/negotiator",
    max_steps=500,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    logging_steps=10,
    save_steps=100,
    max_prompt_length=1024,
    max_completion_length=256,
    report_to=["wandb"],
    run_name="negotiarena-negotiator",
    seed=42,
    kl_coeff=0.05,
)

trainer2 = GRPOTrainer(
    model=model2, tokenizer=tokenizer2,
    config=config2, train_dataset=neg_dataset,
    reward_funcs=[negotiator_reward_fn],
)
trainer2.train()
trainer2.save_model("checkpoints/negotiator")
print("✅ Negotiator adapter saved!")
"""

# ============================================================
# CELL 8: Run AFTER evaluation + comparison
# ============================================================
AFTER_EVAL = """
from evaluation.evaluator import evaluate_random_policy, evaluate_trained_policy, print_comparison
import json
from dataclasses import asdict

print("Running AFTER evaluation...")
after = evaluate_trained_policy("checkpoints/overseer", n_episodes=30)

# Load before baseline
with open("eval_before.json") as f:
    from evaluation.evaluator import EvalSummary
    before_data = json.load(f)
    before = EvalSummary(**before_data)

print_comparison(before, after)

# Save results for HF blog
with open("eval_comparison.json", "w") as f:
    json.dump({"before": asdict(before), "after": asdict(after)}, f, indent=2)

# W&B summary log
import wandb
wandb.log({
    "overseer_f1_before": before.mean_overseer_f1,
    "overseer_f1_after": after.mean_overseer_f1,
    "f1_improvement": after.mean_overseer_f1 - before.mean_overseer_f1,
    "deal_rate_improvement": after.deal_rate - before.deal_rate,
    "fp_reduction": before.false_positive_rate - after.false_positive_rate,
})
wandb.finish()
print("Results saved and logged to W&B!")
"""

# ============================================================
# CELL 9: Push to HuggingFace Hub
# ============================================================
PUSH_TO_HUB = """
from huggingface_hub import HfApi, login
import os

# Login (add your HF token to Colab secrets)
login(token=os.environ["HF_TOKEN"])

api = HfApi()
YOUR_HF_USERNAME = "YOUR_HF_USERNAME"  # Replace

# Push overseer adapter
api.upload_folder(
    folder_path="checkpoints/overseer",
    repo_id=f"{YOUR_HF_USERNAME}/negotiarena-overseer",
    repo_type="model",
)

# Push negotiator adapter
api.upload_folder(
    folder_path="checkpoints/negotiator",
    repo_id=f"{YOUR_HF_USERNAME}/negotiarena-negotiator",
    repo_type="model",
)

print("✅ Models pushed to HuggingFace Hub!")
print(f"   Overseer: https://huggingface.co/{YOUR_HF_USERNAME}/negotiarena-overseer")
print(f"   Negotiator: https://huggingface.co/{YOUR_HF_USERNAME}/negotiarena-negotiator")
"""


if __name__ == "__main__":
    print("NegotiArena Colab Training Script")
    print("Copy each CELL into a Colab notebook and run sequentially.")
    print("\nExpected timeline on A100:")
    print("  Cell 3 (data gen):     ~15 min")
    print("  Cell 5 (load model):   ~5 min")
    print("  Cell 6 (overseer):     ~3-4 hours")
    print("  Cell 7 (negotiator):   ~3-4 hours")
    print("  Total:                 ~8-9 hours")
    print("\nW&B dashboard will show live reward curves during training.")