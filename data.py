import json
from typing import List, Tuple, Dict
import random
from transformers import AutoTokenizer
import torch
from pathlib import Path
from config import Config

class QuestionDataset:
    def __init__(self, config: Config, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
    def load_jsonl_file(self, filepath: str) -> List[Dict]:
        """Load questions from a JSONL file"""
        with open(filepath, 'r') as f:
            return [json.loads(line) for line in f]
        
    def get_all_files(self, data_path: str) -> List[str]:
        """Get all JSONL files recursively"""
        return [
            str(p) for p in Path(data_path).rglob('*.jsonl')
            if p.is_file()
        ]
        
    def preprocess_question(self, question: Dict) -> Dict:
        """Preprocess multiple choice question"""
        prompt = f"Generate a multiple choice technical question about {question['topic']}: "
        target = f"{question['question']}\nOptions:\n" + "\n".join(question['options'])
        
        inputs = self.tokenizer(
            prompt,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = self.tokenizer(
            target,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }
        
    def create_dataloader(self, data_path: str):
        """Create training dataloader from organized data"""
        all_files = self.get_all_files(data_path)
        all_questions = []
        
        # Load all questions
        for file_path in all_files:
            questions = self.load_jsonl_file(file_path)
            all_questions.extend(questions)
        
        # Split into train/validation
        random.shuffle(all_questions)
        train_size = int(len(all_questions) * 0.8)
        train_questions, val_questions = (
            all_questions[:train_size],
            all_questions[train_size:]
        )
        
        # Create datasets
        train_dataset = QuestionDatasetWrapper(
            train_questions,
            self.tokenizer,
            self.config
        )
        val_dataset = QuestionDatasetWrapper(
            val_questions,
            self.tokenizer,
            self.config
        )
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    

class QuestionDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, questions, tokenizer, config):
        self.questions = questions
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        prompt = f"Generate a multiple choice technical question about {question['topic']}: "
        target = f"{question['question']}\nOptions:\n" + "\n".join(question['options'])
        inputs = self.tokenizer(
            prompt,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        labels = self.tokenizer(
            target,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }
