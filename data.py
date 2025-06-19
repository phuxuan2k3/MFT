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
    
    def generate_type_specific_fallbacks(self, topic: str, seniority: str, question_type: str, length: int) -> list:
        """Generate fallback questions based on question type and other parameters"""
        
        fallback_templates = {
            'Multiple Choice': [
                f"What is the primary purpose of {topic}?",
                f"Which of the following best describes {topic}?",
                f"What is a key characteristic of {topic}?",
                f"Which approach is recommended when working with {topic}?",
                f"What is the main advantage of using {topic}?"
            ],
            
            'Multiple Select': [
                f"Which of the following are benefits of {topic}? (Select all that apply)",
                f"Which statements about {topic} are correct?",
                f"Which of these are common use cases for {topic}?",
                f"Which best practices should be followed when using {topic}?",
                f"Which of the following are key components of {topic}?"
            ],
            
            'Short Answer': [
                f"Explain what {topic} is and why it's important.",
                f"Describe how {topic} works in practice.",
                f"What are the main benefits of using {topic}?",
                f"How would you implement {topic} in a project?",
                f"What challenges might you face when working with {topic}?"
            ],
            
            'True/False': [
                f"{topic} is primarily used for data storage.",
                f"{topic} requires significant computational resources.",
                f"{topic} is suitable for beginners to learn.",
                f"{topic} integrates well with modern development frameworks.",
                f"{topic} is an open-source technology."
            ],
            
            'Fill in the Blank': [
                f"The main purpose of {topic} is to ___ data efficiently.",
                f"When implementing {topic}, developers should consider ___ as a key factor.",
                f"The most common use case for {topic} is ___.",
                f"To optimize {topic} performance, you should ___.",
                f"The key difference between {topic} and similar technologies is ___."
            ],
            
            'Code Completion': [
                f"Complete the {topic} implementation code snippet.",
                f"Fix the syntax error in this {topic} code example.",
                f"Write the missing function for {topic} integration.",
                f"Complete the {topic} configuration setup code.",
                f"Implement the error handling for this {topic} operation."
            ],
            
            'Scenario-Based': [
                f"Your team needs to implement {topic} for a high-traffic application. What approach would you take?",
                f"A client reports performance issues with their {topic} implementation. How would you troubleshoot?",
                f"You need to choose between different {topic} solutions for a new project. What factors would you consider?",
                f"Your application using {topic} needs to scale to handle 10x more users. What would you do?",
                f"A security audit found vulnerabilities in your {topic} setup. How would you address them?"
            ]
        }
        
        # Adjust complexity based on seniority level
        if seniority.lower() in ['senior', 'lead', 'principal']:
            # Add complexity indicators for senior levels
            templates = fallback_templates.get(question_type, fallback_templates['Multiple Choice'])
            templates = [t.replace(f"{topic}", f"advanced {topic} concepts") for t in templates]
        elif seniority.lower() in ['intern', 'junior']:
            # Simplify for junior levels
            templates = fallback_templates.get(question_type, fallback_templates['Multiple Choice'])
            templates = [t.replace("implement", "use").replace("optimization", "basic setup") for t in templates]
        else:
            templates = fallback_templates.get(question_type, fallback_templates['Multiple Choice'])
        
        # Return the requested number of questions, cycling through templates if needed
        questions = []
        for i in range(length):
            questions.append(templates[i % len(templates)])
        
        return questions

    def format_input_output(self, question: Dict) -> Tuple[str, str]:
        topic = question.get('topic', 'general')
        seniority = question.get('seniority', 'Intern')
        context = question.get('context', '')
        language = question.get('language', 'English')
        question_type = question.get('question_type', 'Multiple Choice')
        questions = question.get('questions', [])
        length = question.get('length', len(questions) if questions else 5)
        
        # Question type specific instructions and guidelines
        type_specific_info = get_question_type_instructions(question_type)
        
        # Base instruction templates with question type integration
        instruction_templates = [
            f"Your task is to generate {length} {type_specific_info['type_description']} technical questions about {{topic}} suitable for {{seniority}} level developers.",
            f"You are a technical educator. Create {length} {type_specific_info['type_description']} questions about {{topic}} for {{seniority}} level students.",
            f"Design {length} {type_specific_info['type_description']} technical assessment questions about {{topic}} for {{seniority}} level evaluation."
        ]
        
        # Randomly select an instruction template for variety during training
        import random
        base_instruction = random.choice(instruction_templates).format(
            topic=topic, seniority=seniority
        )
        
        # Add language specification if not English
        if language.lower() != 'english':
            base_instruction += f" Generate the questions in {language}."
        
        # Add context with clear instruction
        if context.strip():
            context_short = context[:500] + "..." if len(context) > 500 else context
            base_instruction += f" Use this context as background information: {context_short}"
        
        # Add question type specific formatting and guidelines
        base_instruction += f" {type_specific_info['format_instruction']}"
        base_instruction += f" {type_specific_info['quality_guidelines']}"
        
        input_text = base_instruction
        
        # Target: clean numbered list of questions with type-appropriate fallbacks
        if questions:
            target_text = "\n".join([f"{i+1}. {q.strip()}" for i, q in enumerate(questions)])
        else:
            # Generate type-specific fallback questions
            fallback_questions = self.generate_type_specific_fallbacks(topic, seniority, question_type, length)
            target_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(fallback_questions)])
        
        return input_text, target_text
        
    def create_dataloader(self, train_data_path: str = None, val_data_path: str = None) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Create training and validation dataloaders using separate paths for train and validation data.
        """
        train_data_path = train_data_path or self.config.train_data_path
        val_data_path = val_data_path or self.config.val_data_path

        # Load train data
        train_files = self.get_all_files(train_data_path)
        train_questions = []
        for file_path in train_files:
            try:
                questions = self.load_jsonl_file(file_path)
                train_questions.extend(questions)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        # Load validation data
        val_files = self.get_all_files(val_data_path)
        val_questions = []
        for file_path in val_files:
            try:
                questions = self.load_jsonl_file(file_path)
                val_questions.extend(questions)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

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
            num_workers=0
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
    

def get_question_type_instructions(question_type: str) -> dict:
        """Get question type specific instructions and guidelines"""
        
        type_configs = {
            'Multiple Choice': {
                'type_description': 'multiple choice',
                'format_instruction': 'Present your response as a numbered list where each line contains only one clear, concise question suitable for a multiple choice exam. Do not include answer options or answers.',
                'quality_guidelines': 'Each question should be unambiguous, focused on a single concept, and answerable by selecting from multiple options (not provided).'
            },
            
            'Multiple Select': {
                'type_description': 'multiple select',
                'format_instruction': 'Present your response as a numbered list where each line contains one question that requires selecting multiple correct options from a list.',
                'quality_guidelines': 'Frame questions using phrases like "Which of the following are true about...", "Select all that apply", or "Which options correctly describe...". Ensure multiple correct answers exist.'
            },
            
            'Short Answer': {
                'type_description': 'short answer',
                'format_instruction': 'Present your response as a numbered list where each line contains one question that can be answered with a brief explanation or short phrase.',
                'quality_guidelines': 'Questions should be open-ended but focused, requiring concise explanations rather than yes/no answers. Avoid questions that need lengthy explanations.'
            },
            
            'True/False': {
                'type_description': 'true/false',
                'format_instruction': 'Present your response as a numbered list where each line contains one statement that can be evaluated as either true or false.',
                'quality_guidelines': 'Create clear, unambiguous statements about the topic. Avoid statements that are partially true or context-dependent.'
            },
            
            'Fill in the Blank': {
                'type_description': 'fill in the blank',
                'format_instruction': 'Present your response as a numbered list where each line contains one sentence with a blank space (___) to be filled.',
                'quality_guidelines': 'Create sentences with clear context clues that lead to a specific correct answer. Use underscores (___) to indicate the blank space.'
            },
            
            'Code Completion': {
                'type_description': 'code completion',
                'format_instruction': 'Present your response as a numbered list where each line contains one question about completing or fixing code snippets.',
                'quality_guidelines': 'Focus on practical coding scenarios that require understanding of syntax, logic, or best practices. Questions should be specific to coding tasks.'
            },
            
            'Scenario-Based': {
                'type_description': 'scenario-based',
                'format_instruction': 'Present your response as a numbered list where each line contains one question presenting a real-world scenario that requires problem-solving.',
                'quality_guidelines': 'Create realistic workplace or project scenarios that test practical application of knowledge. Questions should require analytical thinking and decision-making.'
            }
        }
        
        # Default configuration for unknown question types
        default_config = {
            'type_description': 'technical',
            'format_instruction': 'Present your response as a numbered list where each line contains one question.',
            'quality_guidelines': 'Make sure each question is clear, specific, and tests understanding of the topic.'
        }
        
        return type_configs.get(question_type, default_config)