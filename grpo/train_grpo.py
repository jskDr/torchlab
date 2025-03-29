from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")

def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO", logging_steps=10,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,)

trainer = GRPOTrainer(
    model = "Qwen/Qwen2-0.5B-Instruct",
    reward_funcs = reward_len, 
    args = training_args, 
    train_dataset = dataset, 
)

trainer.train()
