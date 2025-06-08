# main.py
import torch
from model import PEFTPromptTuningModel
from config import Config
import json
import re
import os

def parse_structured_output(text):
    """Parse the structured output format with improved regex patterns"""
    result = {
        "topic": "",
        "context": "",
        "question": "",
        "answer": "",
        "options": [],
        "type": "Multiple Choice",
        "seniority": "Intern"
    }
    
    # More flexible regex patterns
    patterns = {
        "topic": r"Topic:\s*(.+?)(?=\n\w+:|$)",
        "context": r"Context:\s*(.*?)(?=\n\w+:|$)",
        "question": r"Question:\s*(.*?)(?=\n(?:Options:|Answer:|\w+:)|$)",
        "answer": r"Answer:\s*([A-E])\)?",
        "type": r"Type:\s*(.+?)(?=\n\w+:|$)",
        "seniority": r"Seniority:\s*(.+?)(?=\n\w+:|$)"
    }
    
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if match:
            result[field] = match.group(1).strip()
    
    # Extract options with improved pattern matching
    options = []
    
    # First, try to extract from the raw text directly (for cases like your example)
    # Pattern to match: "Options:A SELECT.B INSERT.C UPDATE.D DELETE"
    raw_options_match = re.search(r"Options:\s*([A-E]\s+[^.]+\.?\s*)+", text, re.IGNORECASE)
    if raw_options_match:
        options_text = raw_options_match.group(0)
        # Extract individual options: A SELECT, B INSERT, etc.
        option_matches = re.findall(r"([A-E])\s+([^.]+?)(?=\s*[A-E]\s+|Answer:|$)", options_text)
        if option_matches:
            for letter, option_text in option_matches:
                options.append(option_text.strip())
    
    # If that didn't work, try structured format patterns
    if not options:
        option_patterns = [
            r"([A-E])\)\s*(.+?)(?=\n[A-E]\)|Answer:|$)",  # A) option
            r"([A-E]):\s*(.+?)(?=\n[A-E]:|Answer:|$)",   # A: option
            r"([A-E])\.\s*(.+?)(?=\n[A-E]\.|Answer:|$)"  # A. option
        ]
        
        for pattern in option_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            if matches:
                # Sort by letter and extract text
                sorted_matches = sorted(matches, key=lambda x: x[0])
                for letter, option_text in sorted_matches:
                    options.append(option_text.strip())
                break
    
    result["options"] = options
    return result

def generate_mcq(topic, seniority="Intern", context="", model=None, config=None):
    """Generate MCQ using PEFT model with improved prompting"""
    
    # More specific prompting to avoid content issues
    base_prompts = [
        f"Generate a technical multiple choice question about {topic} for {seniority} level",
        f"Create a {seniority} level MCQ about {topic} concepts",
        f"Generate a {topic} question for {seniority} developers with 4 options"
    ]
    
    if context.strip():
        context_short = context[:150] + "..." if len(context) > 150 else context
        prompt = f"{base_prompts[0]} with context: {context_short}"
    else:
        prompt = base_prompts[0]
    
    # Add instruction for better formatting
    prompt += ". Format: Question, then Options A through D, then Answer."
    
    print(f"Input prompt: {prompt}")
    
    # Tokenize
    inputs = model.tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_source_length
    )
    
    # Move to same device as model
    device = next(model.model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with adjusted parameters for better quality
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=config.max_target_length,
            min_length=80,  # Increased minimum length
            num_beams=4,    # Increased beam search
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            early_stopping=True,
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
            repetition_penalty=1.2,  # Increased to reduce repetition
            length_penalty=1.0,
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
        )
    
    # Decode the generated text
    generated = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated.strip()

def validate_mcq(mcq_data):
    """Enhanced validation for MCQ data"""
    required_fields = ["question", "options", "answer"]
    
    for field in required_fields:
        if not mcq_data.get(field):
            return False, f"Missing {field}"
    
    if len(mcq_data["options"]) < 3:
        return False, "Need at least 3 options"
    
    # Check if answer is valid
    if mcq_data["answer"] not in ['A', 'B', 'C', 'D', 'E']:
        return False, "Invalid answer format"
    
    # Validate answer index
    answer_idx = ord(mcq_data["answer"]) - ord('A')
    if answer_idx >= len(mcq_data["options"]):
        return False, f"Answer {mcq_data['answer']} out of range for {len(mcq_data['options'])} options"
    
    return True, "Valid"

def post_process_generation(text, topic, seniority, context):
    """Post-process generation to ensure proper format while preserving original options"""
    
    # First, try to parse the raw text to extract actual content
    parsed = parse_structured_output(text)
    
    # If we successfully parsed the raw text, use the extracted content
    if parsed["question"] and parsed["options"]:
        # Use the actual parsed options and answer
        options_formatted = []
        for i, option in enumerate(parsed["options"]):
            letter = chr(65 + i)  # A, B, C, D, E
            options_formatted.append(f"{letter}) {option}")
        
        options_text = "\n".join(options_formatted)
        
        # Use parsed answer or default to A if invalid
        answer = parsed["answer"] if parsed["answer"] in ['A', 'B', 'C', 'D', 'E'] else 'A'
        
        processed_text = f"""Topic: {topic}
Context: {context[:100] + '...' if len(context) > 100 else context}
Question: {parsed["question"]}
Options:
{options_text}
Answer: {answer}
Type: Multiple Choice
Seniority: {seniority}"""
    
    # If we have a question but no options, try to extract from raw text differently
    elif parsed["question"]:
        # Try to extract options from the raw text using different patterns
        extracted_options = extract_options_from_raw_text(text)
        
        if extracted_options:
            options_formatted = []
            for i, option in enumerate(extracted_options):
                letter = chr(65 + i)
                options_formatted.append(f"{letter}) {option}")
            
            options_text = "\n".join(options_formatted)
            
            # Try to extract answer
            answer_match = re.search(r"Answer:\s*([A-E])", text, re.IGNORECASE)
            answer = answer_match.group(1).upper() if answer_match else 'A'
            
            processed_text = f"""Topic: {topic}
                Context: {context[:100] + '...' if len(context) > 100 else context}
                Question: {parsed["question"]}
                Options:
                {options_text}
                Answer: {answer}
                Type: Multiple Choice
                Seniority: {seniority}"""
    elif "?" in text:
        question_match = re.search(r"(.+?\?)", text)
        if question_match:
            question = question_match.group(1).strip()
            extracted_options = extract_options_from_raw_text(text)
            
            if extracted_options:
                options_formatted = []
                for i, option in enumerate(extracted_options):
                    letter = chr(65 + i)
                    options_formatted.append(f"{letter}) {option}")
                
                options_text = "\n".join(options_formatted)
                
                # Try to extract answer
                answer_match = re.search(r"Answer:\s*([A-E])", text, re.IGNORECASE)
                answer = answer_match.group(1).upper() if answer_match else 'A'
                
                processed_text = f"""Topic: {topic}
Context: {context[:100] + '...' if len(context) > 100 else context}
Question: {question}
Options:
{options_text}
Answer: {answer}
Type: Multiple Choice
Seniority: {seniority}"""           
    else:
        # No question found, return original text
        processed_text = text
    
    return processed_text

def extract_options_from_raw_text(text):
    """Extract options from raw text using multiple patterns"""
    options = []
    
    # Pattern 1: "Options:A WHERE.B ORDER BY.C GROUP BY.D DELETE"
    inline_match = re.search(r"Options:\s*([A-E][^A-E]*(?:[A-E][^A-E]*)*)", text, re.IGNORECASE)
    if inline_match:
        options_text = inline_match.group(1)
        # Split by letter prefixes
        option_parts = re.split(r'(?=[A-E](?:\s|[\.:\)]))', options_text.strip())
        for part in option_parts:
            if part.strip():
                # Remove the letter prefix and clean up
                clean_option = re.sub(r'^[A-E][\s\.:\)]*', '', part.strip())
                if clean_option:
                    options.append(clean_option)
    
    # Pattern 2: Try to find individual A), B), etc.
    if not options:
        option_matches = re.findall(r'([A-E])[\s\.:\)]+([^A-E\n]+)', text)
        for letter, option_text in option_matches:
            clean_option = option_text.strip().rstrip('.')
            if clean_option:
                options.append(clean_option)
    
    # Pattern 3: Look for options after "Options:" keyword
    if not options:
        options_section = re.search(r'Options:\s*(.+?)(?:Answer:|$)', text, re.DOTALL | re.IGNORECASE)
        if options_section:
            options_text = options_section.group(1)
            # Try to split by common separators
            potential_options = re.split(r'[A-E][\s\.:\)]+', options_text)
            for opt in potential_options[1:]:  # Skip first empty split
                clean_opt = opt.strip().rstrip('.')
                if clean_opt and len(clean_opt) > 1:
                    options.append(clean_opt)
    
    return options[:5]  # Limit to 5 options max

def main():
    config = Config()
    
    # Load trained model
    adapter_path = os.path.join(config.output_dir, "final_model")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Trained model not found at {adapter_path}")
    
    print("Loading PEFT model...")
    model = PEFTPromptTuningModel.load_pretrained(config, adapter_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("Model loaded successfully!")
    
    # Interactive generation
    while True:
        topic = input("\nEnter the topic for MCQ generation (or 'quit' to exit): ")
        if topic.lower() == 'quit':
            break
            
        seniority = input("Enter seniority level (Intern/Junior/Senior): ") or "Intern"
        context = input("Enter context (optional, press Enter to skip): ").strip()
        
        print("Generating MCQ...")
        try:
            # Try multiple generations if first fails
            max_attempts = 5  # Increased attempts
            best_result = None
            
            for attempt in range(max_attempts):
                print(f"Generation attempt {attempt + 1}/{max_attempts}")
                
                generated_text = generate_mcq(topic, seniority, context, model, config)
                print(f"\nRaw generated text:\n{generated_text}")
                
                # Post-process
                processed_text = post_process_generation(generated_text, topic, seniority, context)
                print(f"\nProcessed text:\n{processed_text}")
                
                # Parse to JSON
                mcq_json = parse_structured_output(processed_text)
                print(f"\nParsed JSON:\n{json.dumps(mcq_json, indent=2, ensure_ascii=False)}")
                
                # Validate
                is_valid, message = validate_mcq(mcq_json)
                print(f"Validation: {message}")
                
                if is_valid:
                    best_result = mcq_json
                    break
                else:
                    print(f"Attempt {attempt + 1} failed: {message}")
            
            if best_result:
                print(f"\nFinal MCQ JSON:\n{json.dumps(best_result, indent=2, ensure_ascii=False)}")
            else:
                print("Failed to generate valid MCQ after all attempts")
                # Show partial result anyway
                print(f"\nPartial result:\n{json.dumps(mcq_json, indent=2, ensure_ascii=False)}")
            
        except Exception as e:
            print(f"Error generating MCQ: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()