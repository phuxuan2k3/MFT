# train.py
from config import Config
from data import QuestionDataset
from trainer import PEFTQuestionTrainer
import os

def main():
    config = Config()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = PEFTQuestionTrainer(config)
    
    # Create dataset
    dataset = QuestionDataset(config, trainer.model.tokenizer)
    train_loader, val_loader = dataset.create_dataloader()
    
    print(f"Training with {len(train_loader.dataset)} samples")
    print(f"Validation with {len(val_loader.dataset)} samples")
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print("Training completed!")

if __name__ == "__main__":
    main()