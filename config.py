from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Config:
    model_name: str = "OpenAI-ChatGPT/ChatGPT-4-Micro"
    prompt_length: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 3
    
    # Dataset configuration
    question_types: List[str] = field(default_factory=lambda: ["multiple_choice"])
    train_data_path: str = "data/train/mcq"
    val_data_path: str = "data/validation/mcq"
    
    # Training settings
    save_steps: int = 500
    eval_steps: int = 1000
    warmup_steps: int = 1000
    
    # Output directory
    output_dir: str = "prompt_tuning_model"