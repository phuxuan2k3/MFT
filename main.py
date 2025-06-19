# main.py
import torch
from model import PEFTPromptTuningModel
from data import get_question_type_instructions
from config import Config

def parse_question_list_output(text):
    """
    Parse generated text into a list of questions.
    Expected format:
    1. question
    2. question
    """
    import re
    questions = []
    matches = re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|$)", text, re.DOTALL)
    for q in matches:
        q = q.strip()
        if q:
            questions.append(q)
    return questions

def generate_question_list(topic, seniority="Intern", context="", language="English", 
    num_questions=5, question_type="Multiple Choice", model=None, config=None):
    type_info = get_question_type_instructions(question_type)
    type_description = type_info['type_description']
    format_instruction = type_info['format_instruction']
    quality_guidelines = type_info['quality_guidelines']

    base_instruction = f"Generate {num_questions} {type_description} technical questions in {language} about {topic} suitable for {seniority} level developers."

    if language.lower() != 'english':
        base_instruction += f" Generate the questions in {language}."
    if context.strip():
        context_short = context[:500] + "..." if len(context) > 500 else context
        base_instruction += f" Use this context as background information: {context_short}"
    base_instruction += f" {format_instruction} {quality_guidelines}"

    prompt = base_instruction

    print(f"Input instruction: {prompt}")

    inputs = model.tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_source_length
    )
    
    device = next(model.model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=config.max_target_length,
            num_beams=getattr(config, "num_beams", 3),
            temperature=getattr(config, "temperature", 0.8),
            do_sample=True,
            top_p=getattr(config, "top_p", 0.9),
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
            early_stopping=True,
            repetition_penalty=1.1,
            length_penalty=1.0,
        )

    generated = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated}")
    return parse_question_list_output(generated)

def main():
    config = Config()
    adapter_path = f"{config.output_dir}/final_model"
    model = PEFTPromptTuningModel.load_pretrained(config, adapter_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("Model loaded successfully!")
    while True:
        topic = input("\nEnter the topic for Q&A generation (or 'quit' to exit): ").strip()
        if topic.lower() == 'quit':
            break
        seniority = input("Enter seniority level (Intern/Junior/Mid/Senior): ").strip() or "Intern"
        language = input("Enter language (default: English): ").strip() or "English"
        context = input("Enter context (optional, press Enter to skip): ").strip()
        num_questions = input("How many Q&A pairs? (default 5): ").strip()
        try:
            num_questions = int(num_questions) if num_questions else 5
        except Exception:
            num_questions = 5

        print("Generating question list...")
        questions = generate_question_list(
            topic=topic,
            seniority=seniority,
            context=context,
            language=language,
            num_questions=num_questions,
            model=model,
            config=config
        )
        print("\nGenerated questions:")
        for i, qa in enumerate(questions, 1):
            print(f"{i}. {qa['question']}\n")

if __name__ == "__main__":
    main()