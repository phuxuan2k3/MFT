import torch
from transformers import AutoModelForCausalLM
from typing import Optional
from config import Config

class PromptTuningModel(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Load base GPT model
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        
        # Initialize prompt embeddings
        self.prompt_embeddings = torch.nn.Embedding(
            config.prompt_length,
            self.model.config.hidden_size
        )
        
        # Freeze base model weights
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Generate prompt sequence and embeddings
        prompt_sequence = torch.arange(self.config.prompt_length, device=device).unsqueeze(0).expand(batch_size, -1)
        prompt_embeddings = self.prompt_embeddings(prompt_sequence)  # (batch, prompt_length, hidden_size)

        # Get input token embeddings from the model's embedding layer
        input_embeddings = self.model.transformer.wte(input_ids)  # (batch, seq_len, hidden_size)

        # Concatenate prompt and input embeddings
        inputs_embeds = torch.cat([prompt_embeddings, input_embeddings], dim=1)  # (batch, prompt+seq_len, hidden_size)

        # Update attention mask to account for prompt tokens
        prompt_attention = torch.ones((batch_size, self.config.prompt_length), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)  # (batch, prompt+seq_len)

        # Shift labels to the right to match the input (pad prompt positions with -100 so they're ignored in loss)
        if labels is not None:
            ignore_labels = torch.full((batch_size, self.config.prompt_length), -100, device=device, dtype=labels.dtype)
            labels = torch.cat([ignore_labels, labels], dim=1)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
        
    def save_prompt(self, path: str):
        """Save prompt embeddings"""
        torch.save({
            'prompt_embeddings': self.prompt_embeddings.weight,
        }, path)
        
    @classmethod
    def load_prompt(cls, model_path: str, config: Config):
        """Load prompt-tuned model"""
        model = cls(config)
        checkpoint = torch.load(model_path)
        
        model.prompt_embeddings.weight.data.copy_(checkpoint['prompt_embeddings'])
        
        return model