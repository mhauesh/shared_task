import os, re
import openai
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from groq import Groq
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")
fanar_api_key = os.getenv("FANAR_API_KEY")
print(os.getenv("GROQ_API_KEY"))
client_groq = Groq(api_key=groq_api_key) if groq_api_key else None
client_fanar = openai.OpenAI(api_key=fanar_api_key, base_url="https://api.fanar.qa/v1/chat/completions") if fanar_api_key else None

allam_tokenizer = None
allam_model = None

def get_valid_responses(choice5, choice6):
    """Generate valid response set."""
    valid = {"A", "B", "C", "D"}
    if choice5: valid.add("E")
    if choice6: valid.add("F")
    return valid


def clean_and_validate_response(raw_response, valid_responses):
    """Clean and extract valid answer letter."""
    if not raw_response:
        return None

    raw_response = raw_response.strip().upper()

    # More robust regex extraction
    match = re.search(r"(?:answer\s*(?:is)?\s*[:\-]?\s*)([A-F])", raw_response, re.IGNORECASE)
    if match:
        candidate = match.group(1).upper()
        if candidate in valid_responses:
            return candidate

    # Fallback simple extraction
    match = re.search(r"\b([A-F])\b", raw_response)
    if match and match.group(1) in valid_responses:
        return match.group(1)

    return None


def generate_mcq_prompt(question, choices):
    """
    Generate MCQ prompt with few-shot Islamic examples.
    """
    options_text = "\n".join([f"{letter}) {text}" for letter, text in choices])
    valid_letters = "/".join([letter for letter, _ in choices])

    few_shot_examples = """
    Example 1:
    Question: ما مدة المسح على الخفين للمقيم؟
    A) يوم وليلة
    B) ثلاثة أيام بلياليهن
    C) يومان وليلتان
    D) أسبوع كامل
    Answer: A
    
    Example 2:
    Question: توفي عن أب، وأخوين شقيقين، وابن أخ شقيق، وعمين شقيقين، وأم، وبنتين، و زوجة، فما نصيب الأم؟
    A) الثلث
    B) الربع
    C) السدس
    D) الثمن
    E) النصف
    F) لا شيء
    Answer: C
    
    Now answer the following question:
"""

    prompt = f"""{few_shot_examples}

You are a specialist in Islamic sciences. Your task is to answer multiple-choice questions by selecting the correct option.

Question: {question}

{options_text}

Please respond using **only one English letter** from the following: {valid_letters}
Do not write any explanation or additional text.
"""
    return prompt


def pack_choices(choice1, choice2, choice3, choice4, choice5=None, choice6=None):
    """Pack MCQ choices as list of (letter, text)."""
    choices = [("A", choice1), ("B", choice2), ("C", choice3), ("D", choice4)]
    if choice5: choices.append(("E", choice5))
    if choice6: choices.append(("F", choice6))
    return choices


def get_prediction_allam(
    question, choice1, choice2, choice3, choice4, choice5=None, choice6=None,
    model_version="ALLaM-AI/ALLaM-7B-Instruct-preview", max_new_tokens=512, max_retries=3
):
    """Inference using local Allam 7B model (HuggingFace)."""
    global allam_model, allam_tokenizer

    if not allam_model or not allam_tokenizer:
        allam_tokenizer = AutoTokenizer.from_pretrained(model_version)
        allam_model = AutoModelForCausalLM.from_pretrained(model_version, torch_dtype=torch.bfloat16, device_map="auto")

    choices = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
    valid_responses = get_valid_responses(choice5, choice6)
    prompt = generate_mcq_prompt(question, choices)

    for attempt in range(1, max_retries + 1):
        try:
            inputs = allam_tokenizer(prompt, return_tensors="pt").to(allam_model.device)
            outputs = allam_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=allam_tokenizer.eos_token_id
            )
            response = allam_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split(prompt)[-1].strip()

            cleaned_result = clean_and_validate_response(response, valid_responses)
            if cleaned_result:
                print(f"✅ {question} | Allam  | Prediction: {cleaned_result}")
                return cleaned_result
            else:
                print(f"⚠️ Attempt {attempt} invalid: {response}")
        except Exception as e:
            print(f"❌ Allam error: {e}")
            return None

    print("❌ Failed after retries.")
    return None


def get_prediction_fanar(
    question, choice1, choice2, choice3, choice4, choice5=None, choice6=None,
    model_version="Islamic-RAG", max_retries=3
):
    """Inference using Fanar API."""
    if not fanar_api_key:
        print("Fanar API key missing.")
        return None

    fanar_url = "https://api.fanar.qa/v1/chat/completions"
    choices = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
    valid_responses = get_valid_responses(choice5, choice6)
    prompt = generate_mcq_prompt(question, choices)

    headers = {"Authorization": f"Bearer {fanar_api_key}", "Content-Type": "application/json"}
    data = {"model": model_version, "messages": [{"role": "user", "content": prompt}]}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(fanar_url, json=data, headers=headers)
            response_json = response.json()

            if response.status_code == 200:
                raw_result = response_json["choices"][0]["message"]["content"].strip().upper()
                cleaned_result = clean_and_validate_response(raw_result, valid_responses)
                if cleaned_result:
                    print(f"✅ {question} | Fanar  | Prediction: {cleaned_result}")
                    return cleaned_result
            else:
                print(f"❌ Fanar API Error: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Fanar Error: {e}")
            return None

    print("❌ Failed after retries.")
    return None


def get_prediction_mistral(
    question, choice1, choice2, choice3, choice4, choice5=None, choice6=None,
    model_version="mistral-saba-24b", max_retries=3
):
    """Inference using Mistral API (via Groq)."""
    if not client_groq:
        print("Groq API key missing.")
        return None

    choices = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
    valid_responses = get_valid_responses(choice5, choice6)
    prompt = generate_mcq_prompt(question, choices)

    for attempt in range(1, max_retries + 1):
        try:
            response = client_groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], model=model_version
            )
            raw_result = response.choices[0].message.content.strip().upper()
            cleaned_result = clean_and_validate_response(raw_result, valid_responses)
            if cleaned_result:
                print(f"✅ {question} | Mistral  | Prediction: {cleaned_result}")
                return cleaned_result
        except Exception as e:
            print(f"❌ Mistral Error: {e}")
            return None

    print("❌ Failed after retries.")
    return None


import sys
import os

try:
    from symbolic_calculator import SymbolicInheritanceCalculator
    
    # Initialize validator
    rules_path = 'results/prediction/enhanced_inheritance_rules.py'
    symbolic_validator = SymbolicInheritanceCalculator(rules_path)
    SYMBOLIC_VALIDATOR_AVAILABLE = True
    print("✅ Symbolic inheritance validator initialized in models.py")
    
except Exception as e:
    print(f"⚠️ Could not initialize symbolic validator in models.py: {e}")
    symbolic_validator = None
    SYMBOLIC_VALIDATOR_AVAILABLE = False


def get_prediction_fanar_validated(question, choice1, choice2, choice3, choice4, choice5=None, choice6=None):
    """
    Fanar LLM with real-time symbolic inheritance validation
    Shows validation decisions as they happen
    """
    
    # Step 1: Get original Fanar prediction
    fanar_prediction = get_prediction_fanar(question, choice1, choice2, choice3, choice4, choice5, choice6)
    
    if not fanar_prediction:
        print("❌ Fanar prediction failed")
        return None
    
    # Step 2: Apply symbolic validation if available
    if not SYMBOLIC_VALIDATOR_AVAILABLE or not symbolic_validator:
        print(f"🔄 {question[:70]}{'...' if len(question) > 70 else ''}")
        print(f"   Fanar: {fanar_prediction} (No symbolic validation available)")
        return fanar_prediction
    
    try:
        # Prepare options for validation
        options = []
        for choice in [choice1, choice2, choice3, choice4, choice5, choice6]:
            if choice is not None:
                options.append(str(choice))
        
        # Parse inheritance scenario
        family_composition = symbolic_validator.extract_family_composition(question)
        question_type = symbolic_validator.classify_question_type(question)
        target_heir = symbolic_validator.identify_target_heir(question)
        
        # Display question analysis
        print(f"🔄 {question[:70]}{'...' if len(question) > 70 else ''}")
        
        if family_composition:
            family_str = ', '.join([f"{heir}:{count}" for heir, count in family_composition.items()])
            print(f"   👥 Family: {family_str}")
            print(f"   🎯 Type: {question_type}" + (f" → {target_heir}" if target_heir else ""))
        
        # Look up expected answer from training patterns
        expected_answer = None
        confidence = 0
        reasoning = ""
        
        if family_composition:
            expected_answer = symbolic_validator.lookup_pattern_answer(
                family_composition, question_type, target_heir
            )
            
            # Estimate confidence from pattern data
            if expected_answer:
                family_key = tuple(sorted(family_composition.items()))
                
                # Check QA patterns for confidence
                if target_heir and hasattr(symbolic_validator, 'qa_patterns'):
                    qa_key = str((family_key, question_type, target_heir))
                    if hasattr(symbolic_validator, 'qa_patterns') and symbolic_validator.qa_patterns:
                        for key, pattern_data in symbolic_validator.qa_patterns.items():
                            if qa_key in key:
                                if isinstance(pattern_data, dict):
                                    confidence = pattern_data.get('confidence', 0)
                                    reasoning = f"QA pattern (conf: {confidence:.2f})"
                                break
                
                # Fallback to scenario patterns
                if confidence == 0 and hasattr(symbolic_validator, 'scenario_patterns'):
                    scenario_key = str((family_key, question_type))
                    if hasattr(symbolic_validator, 'scenario_patterns') and symbolic_validator.scenario_patterns:
                        for key, pattern_data in symbolic_validator.scenario_patterns.items():
                            if scenario_key in key:
                                if isinstance(pattern_data, dict):
                                    confidence = pattern_data.get('confidence', 0.5)
                                    reasoning = f"Scenario pattern (conf: {confidence:.2f})"
                                break
        
        # Make validation decision and display result
        print(f"   🤖 Fanar: {fanar_prediction}")
        
        if expected_answer and expected_answer != fanar_prediction:
            if confidence > 0.75:  # High confidence correction
                print(f"   ✅ CORRECTED -> {expected_answer}")
                print(f"   📊 Reason: {reasoning} - High confidence override")
                return expected_answer
            elif confidence > 0.6:  # Medium confidence - show warning but keep original
                print(f"   ⚠️  Pattern suggests {expected_answer} ({reasoning})")
                print(f"   ✅ KEEPING original (medium confidence)")
                return fanar_prediction
            else:  # Low confidence - keep original
                print(f"   ✅ VALIDATED (pattern suggests {expected_answer} but low confidence)")
                return fanar_prediction
        elif expected_answer and expected_answer == fanar_prediction:
            print(f"   ✅ VALIDATED ✓ ({reasoning})")
            return fanar_prediction
        else:
            # No pattern found
            if family_composition:
                print(f"   ✅ VALIDATED (no pattern available for this scenario)")
            else:
                print(f"   ✅ VALIDATED (non-inheritance question)")
            return fanar_prediction

    except Exception as e:
        print(f"   🤖 Fanar: {fanar_prediction}")
        print(f"   ❌ Validation error: {str(e)[:60]}...")
        print(f"   ✅ KEEPING original")
        return fanar_prediction

def get_prediction_fanar_validated_detailed(question, choice1, choice2, choice3, choice4, choice5=None, choice6=None):
    """
    Detailed version with more verbose output for debugging
    """
    
    print("=" * 80)
    print(f"QUESTION: {question}")
    
    # Get Fanar prediction
    fanar_prediction = get_prediction_fanar(question, choice1, choice2, choice3, choice4, choice5, choice6)
    
    if not fanar_prediction:
        print("❌ FANAR FAILED")
        print("=" * 80)
        return None
    
    print(f"🤖 FANAR PREDICTION: {fanar_prediction}")
    
    if not SYMBOLIC_VALIDATOR_AVAILABLE or not symbolic_validator:
        print("⚠️ SYMBOLIC VALIDATION: Not available")
        print("=" * 80)
        return fanar_prediction
    
    try:
        # Parse scenario
        family_composition = symbolic_validator.extract_family_composition(question)
        question_type = symbolic_validator.classify_question_type(question)
        target_heir = symbolic_validator.identify_target_heir(question)
        
        print(f"👥 FAMILY COMPOSITION: {family_composition}")
        print(f"🎯 QUESTION TYPE: {question_type}")
        print(f"🔍 TARGET HEIR: {target_heir}")
        
        # Get pattern prediction
        expected_answer = symbolic_validator.lookup_pattern_answer(
            family_composition, question_type, target_heir if target_heir is not None else ""
        )
        
        print(f"📊 PATTERN PREDICTION: {expected_answer}")
        
        if expected_answer and expected_answer != fanar_prediction:
            print("⚠️ MISMATCH DETECTED!")
            print(f"✅ FINAL DECISION: Using pattern -> {expected_answer}")
            final_answer = expected_answer
        elif expected_answer:
            print("✅ VALIDATION: Pattern confirms Fanar")
            final_answer = fanar_prediction
        else:
            print("ℹ️ NO PATTERN: Keeping Fanar prediction")
            final_answer = fanar_prediction
        
        print(f"🎯 FINAL ANSWER: {final_answer}")
        print("=" * 80)
        return final_answer
        
    except Exception as e:
        print(f"❌ VALIDATION ERROR: {e}")
        print(f"✅ FALLBACK: Using Fanar -> {fanar_prediction}")
        print("=" * 80)
        return fanar_prediction

# Statistics tracking
class ValidationStats:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_questions = 0
        self.corrections_made = 0
        self.validations_confirmed = 0
        self.no_pattern_available = 0
        self.validation_errors = 0
    
    def log_correction(self):
        self.total_questions += 1
        self.corrections_made += 1
    
    def log_validation(self):
        self.total_questions += 1
        self.validations_confirmed += 1
    
    def log_no_pattern(self):
        self.total_questions += 1
        self.no_pattern_available += 1
    
    def log_error(self):
        self.total_questions += 1
        self.validation_errors += 1
    
    def print_summary(self):
        if self.total_questions > 0:
            print("\n" + "="*60)
            print("📊 SYMBOLIC VALIDATION SUMMARY")
            print("="*60)
            print(f"Total questions processed: {self.total_questions}")
            print(f"Corrections made: {self.corrections_made} ({self.corrections_made/self.total_questions*100:.1f}%)")
            print(f"Validations confirmed: {self.validations_confirmed} ({self.validations_confirmed/self.total_questions*100:.1f}%)")
            print(f"No pattern available: {self.no_pattern_available} ({self.no_pattern_available/self.total_questions*100:.1f}%)")
            print(f"Validation errors: {self.validation_errors} ({self.validation_errors/self.total_questions*100:.1f}%)")
            print("="*60)

# Global stats instance
validation_stats = ValidationStats()

def get_validation_summary():
    """Get validation statistics summary"""
    return validation_stats.print_summary()