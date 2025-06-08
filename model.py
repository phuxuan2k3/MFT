# model.py
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import get_peft_model, PromptTuningConfig, TaskType, PeftModel
from config import Config

class PEFTPromptTuningModel:
    def __init__(self, config: Config):
        self.config = config
        
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_name)
        self.base_model = T5ForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32
        )
        
        # Configure PEFT for T5
        self.peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,  # T5 is seq2seq, not causal LM
            prompt_tuning_init=config.prompt_tuning_init,
            num_virtual_tokens=config.num_virtual_tokens,
            prompt_tuning_init_text=config.prompt_tuning_init_text,
            tokenizer_name_or_path=config.model_name,
        )
        
        # Create PEFT model
        self.model = get_peft_model(self.base_model, self.peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through the PEFT model"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def save_pretrained(self, save_directory):
        """Save the PEFT adapter"""
        self.model.save_pretrained(save_directory)
        
    @classmethod
    def load_pretrained(cls, config: Config, adapter_path: str):
        """Load a trained PEFT model"""
        # Load tokenizer
        tokenizer = T5Tokenizer.from_pretrained(config.model_name)
        
        # Load base model
        base_model = T5ForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32
        )
        
        # Load PEFT model
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Create wrapper
        instance = cls.__new__(cls)
        instance.config = config
        instance.tokenizer = tokenizer
        instance.base_model = base_model
        instance.model = model
        
        return instance
    
    def generate(self, input_ids, attention_mask, **kwargs):
        """Generate text using the PEFT model"""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        return self