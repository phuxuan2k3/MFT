# config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # FLAN-T5 large model
    model_name: str = "google/flan-t5-large"
    batch_size: int = 4
    learning_rate: float = 3e-3
    epochs: int = 3
    
    # PEFT-specific configurations
    peft_type: str = "PROMPT_TUNING"
    num_virtual_tokens: int = 20
    prompt_tuning_init: str = "TEXT"
    prompt_tuning_init_text: str = "Generate a technical multiple choice question with context and answer options"
    
    # Dataset configuration
    question_types: List[str] = field(default_factory=lambda: ["multiple_choice"])
    train_data_path: str = "data/train/mcq"
    val_data_path: str = "data/validation/mcq"
    
    # Training settings
    save_steps: int = 500
    eval_steps: int = 250
    warmup_steps: int = 100
    logging_steps: int = 50
    
    # Output directory
    output_dir: str = "output"
    max_source_length: int = 256
    max_target_length: int = 256
    
    # Generation parameters
    num_beams: int = 4
    temperature: float = 0.8
    top_p: float = 0.9