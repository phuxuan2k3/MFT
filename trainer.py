import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from model import PromptTuningModel
from config import Config


class QuestionTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = PromptTuningModel(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = None
        self.scheduler = None

    def train(self, train_dataloader, val_dataloader=None):
        """Train the prompt-tuned model"""
        self.optimizer = torch.optim.Adam(
            self.model.prompt_embeddings.parameters(),
            lr=self.config.learning_rate
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=len(train_dataloader) * self.config.epochs
        )

        best_loss = float('inf')
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0

            with tqdm(train_dataloader, desc=f'Epoch {epoch+1}') as pbar:
                for batch in pbar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    # Zero gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs.loss

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{total_loss/(pbar.n+1):.4f}'})

            # Save checkpoint
            current_loss = total_loss / len(train_dataloader)
            self.model.save_prompt(f"{self.config.output_dir}/epoch_{epoch+1}.pth")

            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.model.save_prompt(f"{self.config.output_dir}/best.pth")
