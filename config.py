# config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # FLAN-T5 large model
    model_name: str = "google/flan-t5-large"
    batch_size: int = 4
    learning_rate: float = 1e-3
    epochs: int = 1
    
    # PEFT-specific configurations
    peft_type: str = "PROMPT_TUNING"
    num_virtual_tokens: int = 30
    prompt_tuning_init: str = "TEXT"
    prompt_tuning_init_text: str = (
        "Here are some examples of generating technical questions:\n\n"
        "Example 1 Input:\n"
        "Generate 3 multiple-choice technical questions in English about Databases suitable for Intern level developers. "
        "Context: SQL (Structured Query Language) is a domain-specific language used in programming and designed for managing data held in a relational database management system (RDBMS), or for stream processing in a relational data stream management system (RDSMS). It is particularly useful for handling structured data where there are relations between different entities/tables.\n"
        "Example 1 Output:\n"
        "1. Which SQL clause is used to filter records that satisfy a specified condition?\n"
        "2. What does ACID stand for in the context of database transactions?\n"
        "3. In a relational database, what is a primary key used for?\n\n"

        "Example 2 Input:\n"
        "Generate 4 multiple-choice technical questions in English about Operating Systems suitable for Junior level developers. "
        "Context: An operating system (OS) is system software that manages computer hardware and software resources and provides common services for computer programs. All computer programs, excluding firmware, require an operating system to function.\n"
        "Example 2 Output:\n"
        "1. Which of the following is a primary function of an operating system?\n"
        "2. What is the main purpose of virtual memory in an OS?\n"
        "3. Which of the following is NOT a type of operating system?\n"
        "4. What is the difference between a process and a thread?\n\n"

        "Now, generate exactly {length} {question_type} technical questions in {language} about {topic} for {seniority} developers. "
        "Context: {context}\n"
        "Format: Numbered list (1., 2., 3.) with one question per line.\n"
        "Questions:\n"
    )
    
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
    max_source_length: int = 512
    max_target_length: int = 512
    
    # Enhanced generation parameters
    num_beams: int = 6
    temperature: float = 0.9
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    max_new_tokens: int = 512