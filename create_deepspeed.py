import json

deepspeed_config = {
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 1,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.3
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    "gradient_accumulation_steps": 16,
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 2e-5,
            "warmup_num_steps": 0.03
        }
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "num_checkpoints": None,
        "contiguous_memory_optimization": True,
        "synchronize_checkpoint_boundary": False
    },
    "steps_per_print": 1,
    "wall_clock_breakdown": False
}

with open('deepspeed_config.json', 'w') as f:
    json.dump(deepspeed_config, f, indent=4)

print("Deepspeed configuration saved to deepspeed_config.json")
