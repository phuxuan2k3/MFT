from config import Config
from data import QuestionDataset
from model import PromptTuningModel
from trainer import QuestionTrainer
import torch
from transformers import AutoTokenizer

def main():
    # Initialize configuration
    config = Config()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset and dataloaders (returns both train and val loaders)
    dataset = QuestionDataset(config, tokenizer)
    train_dataloader, val_dataloader = dataset.create_dataloader(config.train_data_path)
    
    # Initialize trainer
    trainer = QuestionTrainer(config)
    
    # Train model
    trainer.train(train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()