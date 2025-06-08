# data.py
import json
from typing import List, Dict, Tuple
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
        questions = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        return questions
        
    def get_all_files(self, data_path: str) -> List[str]:
        """Get all JSONL files recursively"""
        path = Path(data_path)
        if not path.exists():
            return []
        return [str(p) for p in path.rglob('*.jsonl') if p.is_file()]
        
    def format_input_output(self, question: Dict) -> Tuple[str, str]:
        """Create input-output pair for T5 training with consistent formatting"""
        
        # Create varied input formats to make model more robust
        topic = question.get('topic', 'general')
        seniority = question.get('seniority', 'Intern')
        context = question.get('context', '')
        
        # Input variations
        input_variations = [
            f"Generate a technical multiple choice question about {topic} for {seniority} level",
            f"Create a {seniority} level MCQ about {topic}",
            f"Generate a technical multiple choice question about {topic} for {seniority} level with context",
        ]
        
        # Add context to some inputs
        if context and len(context) > 20:
            context_short = context[:150] + "..." if len(context) > 150 else context
            input_text = f"{random.choice(input_variations)} with context: {context_short}"
        else:
            input_text = random.choice(input_variations)
        
        # Create consistent target format
        options_formatted = []
        for i, option in enumerate(question.get('options', [])):
            letter = chr(65 + i)  # A, B, C, D, E
            options_formatted.append(f"{letter}) {option}")
        
        options_text = "\n".join(options_formatted)
        
        # Ensure answer is in correct format
        answer = question.get('answer', 'A')
        if len(answer) > 1:
            # If answer is full text, try to match to option
            for i, option in enumerate(question.get('options', [])):
                if answer.lower() in option.lower() or option.lower() in answer.lower():
                    answer = chr(65 + i)
                    break
        
        target_text = f"""Topic: {topic}
                        Context: {context}
                        Question: {question.get('question', '')}
                        Options:
                        {options_text}
                        Answer: {answer}
                        Type: {question.get('type', 'Multiple Choice')}
                        Seniority: {seniority}"""
        
        return input_text, target_text
        
    def create_synthetic_data(self, count: int = 100) -> List[Dict]:
        """Create synthetic training data if no real data available"""
        synthetic_questions = []
        
        topics = ['Database', 'Python', 'JavaScript', 'Machine Learning', 'Web Development', 
                 'Data Structures', 'Algorithms', 'SQL', 'Cloud Computing', 'API Design']
        
        seniorities = ['Intern', 'Junior', 'Senior']
        
        for i in range(count):
            topic = random.choice(topics)
            seniority = random.choice(seniorities)
            
            # Create a basic question structure
            question_data = {
                'topic': topic,
                'seniority': seniority,
                'context': f'This is about {topic} fundamentals for {seniority} level.',
                'question': f'What is a key concept in {topic}?',
                'options': [
                    f'{topic} concept A',
                    f'{topic} concept B', 
                    f'{topic} concept C',
                    f'{topic} concept D'
                ],
                'answer': 'A',
                'type': 'Multiple Choice'
            }
            synthetic_questions.append(question_data)
        
        return synthetic_questions
        
    def create_dataloader(self, data_path: str) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create training and validation dataloaders"""
        all_files = self.get_all_files(data_path)
        all_questions = []
        
        # Load real data if available
        if all_files:
            for file_path in all_files:
                try:
                    questions = self.load_jsonl_file(file_path)
                    all_questions.extend(questions)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        # If no real data, create synthetic data
        if not all_questions:
            print("No real data found, creating synthetic training data...")
            all_questions = self.create_synthetic_data(200)
        
        print(f"Total questions available: {len(all_questions)}")
        
        # Data augmentation: create variations of existing questions
        augmented_questions = []
        for q in all_questions:
            # Original question
            augmented_questions.append(q)
            
            # Create variation with different seniorities
            for seniority in ['Intern', 'Junior', 'Senior']:
                if seniority != q.get('seniority', 'Intern'):
                    q_variant = q.copy()
                    q_variant['seniority'] = seniority
                    augmented_questions.append(q_variant)
        
        all_questions = augmented_questions
        
        # Split into train/validation
        random.seed(42)
        random.shuffle(all_questions)
        train_size = int(len(all_questions) * 0.8)
        train_questions = all_questions[:train_size]
        val_questions = all_questions[train_size:]
        
        print(f"Training questions: {len(train_questions)}")
        print(f"Validation questions: {len(val_questions)}")
        
        # Create datasets
        train_dataset = QuestionDatasetWrapper(train_questions, self.tokenizer, self.config)
        val_dataset = QuestionDatasetWrapper(val_questions, self.tokenizer, self.config)
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=0  # Avoid multiprocessing issues
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def collate_fn(self, batch):
        """Custom collate function for T5 training"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class QuestionDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, questions: List[Dict], tokenizer: AutoTokenizer, config: Config):
        self.questions = questions
        self.tokenizer = tokenizer
        self.config = config
        self.dataset_processor = QuestionDataset(config, tokenizer)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        input_text, target_text = self.dataset_processor.format_input_output(question)
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.config.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.config.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = input_encoding['input_ids'].squeeze()
        attention_mask = input_encoding['attention_mask'].squeeze()
        labels = target_encoding['input_ids'].squeeze()
        
        # Replace padding token ids in labels with -100 (ignored by loss function)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }