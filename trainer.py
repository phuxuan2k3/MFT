# trainer.py
import torch
from transformers import TrainingArguments, Trainer
from model import PEFTPromptTuningModel
from config import Config
import os

class PEFTQuestionTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = PEFTPromptTuningModel(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, train_dataloader, val_dataloader=None):
        """Train using HuggingFace Trainer"""
        
        # Convert dataloaders to datasets
        train_dataset = train_dataloader.dataset
        eval_dataset = val_dataloader.dataset if val_dataloader else None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",  # Changed from evaluation_strategy
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None,  # Disable wandb logging
            gradient_accumulation_steps=2,
            fp16=torch.cuda.is_available(),
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.model.tokenizer,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save the final model
        final_model_path = os.path.join(self.config.output_dir, "final_model")
        self.model.save_pretrained(final_model_path)
        print(f"Model saved to {final_model_path}")
        
        return trainer