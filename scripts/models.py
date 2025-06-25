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
from scripts.validation import validate_fanar_answer

# Load environment variables
load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")
fanar_api_key = os.getenv("FANAR_API_KEY")
print(os.getenv("GROQ_API_KEY"))
client_groq = Groq(api_key=groq_api_key) if groq_api_key else None
client_fanar = openai.OpenAI(api_key=fanar_api_key, base_url="https://api.fanar.qa/v1/chat/completions") if fanar_api_key else None

allam_tokenizer = None
allam_model = None

try:
    from symbolic_calculator import SymbolicInheritanceCalculator
    VALIDATOR_AVAILABLE = True
    
    # Initialize the validator (only once)
    try:
        validator = SymbolicInheritanceCalculator('results/prediction/enhanced_inheritance_rules.py')
        print("✅ Symbolic validator loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load validator: {e}")
        validator = None
        VALIDATOR_AVAILABLE = False

except Exception as e:
    print(f"Error: {e}")


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


def get_prediction_fanar_validated(question, choice1, choice2, choice3, choice4, choice5=None, choice6=None):
    """
    Fanar API with real-time symbolic validation display
    Shows original prediction vs validated prediction as it runs
    """
    
    # Step 1: Get original Fanar prediction (but suppress its print for cleaner output)
    original_prediction = get_prediction_fanar(question, choice1, choice2, choice3, choice4, choice5, choice6)
    
    if not original_prediction:
        print("❌ Fanar prediction failed")
        return None
    
    # Step 2: Apply symbolic validation if available
    if not VALIDATOR_AVAILABLE or not validator:
        print(f"🔄 {question[:60]}{'...' if len(question) > 60 else ''}")
        print(f"   Original: {original_prediction} (No validation available)")
        return original_prediction
    
    try:
        # Prepare options for validation
        options = []
        for choice in [choice1, choice2, choice3, choice4, choice5, choice6]:
            if choice is not None:
                options.append(str(choice))
        
        # Parse the question and family composition
        family_composition = validator.extract_family_composition(question)
        question_type = validator.classify_question_type(question)
        target_heir = validator.identify_target_heir(question)
        
        # Try to get expected answer from patterns
        expected_answer = None
        confidence = 0
        
        if family_composition:
            expected_answer = validator.lookup_pattern_answer(family_composition, question_type, target_heir)
            
            # If we have a pattern match, get confidence
            if expected_answer:
                # Try to estimate confidence from the pattern data
                family_key = tuple(sorted(family_composition.items()))
                
                # Check QA patterns first
                if target_heir:
                    qa_key = str((family_key, question_type, target_heir))
                    if hasattr(validator, 'qa_patterns') and validator.qa_patterns and qa_key in validator.qa_patterns:
                        pattern_data = validator.qa_patterns[qa_key]
                        if isinstance(pattern_data, dict):
                            confidence = pattern_data.get('confidence', 0)
                
                # Check scenario patterns as fallback
                if confidence == 0:
                    scenario_key = str((family_key, question_type))
                    if hasattr(validator, 'scenario_patterns') and validator.scenario_patterns and scenario_key in validator.scenario_patterns:
                        pattern_data = validator.scenario_patterns[scenario_key]
                        if isinstance(pattern_data, dict):
                            confidence = pattern_data.get('confidence', 0)
        
        # Display validation results in real-time
        print(f"🔄 {question[:60]}{'...' if len(question) > 60 else ''}")
        
        if family_composition:
            family_str = ', '.join([f"{heir}:{count}" for heir, count in family_composition.items()])
            print(f"   Family: {family_str}")
            print(f"   Type: {question_type}" + (f" (asking about {target_heir})" if target_heir else ""))
        
        # Decision logic and real-time display
        if expected_answer and expected_answer != original_prediction:
            if confidence > 0.7:  # High confidence correction
                print(f"   Original: {original_prediction}")
                print(f"   ✅ CORRECTED → {expected_answer} (confidence: {confidence:.2f})")
                print(f"   Reason: High-confidence pattern match")
                return expected_answer
            elif confidence > 0.5:  # Medium confidence - show but don't correct
                print(f"   Original: {original_prediction}")
                print(f"   ⚠️ Pattern suggests {expected_answer} (confidence: {confidence:.2f}) - keeping original")
                return original_prediction
            else:
                print(f"   Original: {original_prediction} ✓ (low confidence pattern)")
                return original_prediction
        elif expected_answer and expected_answer == original_prediction:
            print(f"   Original: {original_prediction}")
            print(f"   ✅ VALIDATED ✓ (pattern confirms)")
            return original_prediction
        else:
            # No pattern found or family composition unclear
            if family_composition:
                print(f"   Original: {original_prediction} ✓ (no pattern available)")
            else:
                print(f"   Original: {original_prediction} ✓ (family unclear)")
            return original_prediction
    
    except Exception as e:
        print(f"🔄 {question[:60]}{'...' if len(question) > 60 else ''}")
        print(f"   Original: {original_prediction} ✓ (validation error: {str(e)[:50]})")
        return original_prediction

def get_prediction_fanar_validated_detailed(question, choice1, choice2, choice3, choice4, choice5=None, choice6=None):
    """
    More detailed validation output - use this if you want even more information
    """
    
    # Get original prediction
    original_prediction = get_prediction_fanar(question, choice1, choice2, choice3, choice4, choice5, choice6)
    
    if not original_prediction:
        return None
    
    if not VALIDATOR_AVAILABLE or not validator:
        print(f"📝 QUESTION: {question}")
        print(f"🤖 FANAR: {original_prediction} (No validation)")
        print("─" * 80)
        return original_prediction
    
    try:
        # Parse question
        family_composition = validator.extract_family_composition(question)
        question_type = validator.classify_question_type(question)
        target_heir = validator.identify_target_heir(question)
        
        print(f"📝 QUESTION: {question}")
        print(f"👥 FAMILY: {family_composition}")
        print(f"🎯 TYPE: {question_type}" + (f" → {target_heir}" if target_heir else ""))
        print(f"🤖 FANAR: {original_prediction}")
        
        # Get expected answer
        expected_answer = validator.lookup_pattern_answer(family_composition, question_type, target_heir)
        
        if expected_answer:
            if expected_answer != original_prediction:
                print(f"🔍 PATTERN: {expected_answer}")
                print(f"⚠️ MISMATCH DETECTED!")
                
                print(f"✅ FINAL: {expected_answer} (pattern overrides)")
                print("─" * 80)
                return expected_answer
            else:
                print(f"✅ VALIDATED: Pattern confirms {original_prediction}")
                print("─" * 80)
                return original_prediction
        else:
            print(f"🔍 PATTERN: None found")
            print(f"✅ FINAL: {original_prediction} (no pattern available)")
            print("─" * 80)
            return original_prediction
    
    except Exception as e:
        print(f"🤖 FANAR: {original_prediction}")
        print(f"❌ VALIDATION ERROR: {e}")
        print("─" * 80)
        return original_prediction

# Statistics tracking (optional)
class ValidationStats:
    def __init__(self):
        self.total_questions = 0
        self.corrections_made = 0
        self.validations_confirmed = 0
        self.no_pattern_available = 0
        
    def log_result(self, result_type):
        self.total_questions += 1
        if result_type == "corrected":
            self.corrections_made += 1
        elif result_type == "validated":
            self.validations_confirmed += 1
        elif result_type == "no_pattern":
            self.no_pattern_available += 1
    
    def print_summary(self):
        if self.total_questions > 0:
            print("\n" + "="*50)
            print("📊 VALIDATION SUMMARY")
            print("="*50)
            print(f"Total questions: {self.total_questions}")
            print(f"Corrections made: {self.corrections_made} ({self.corrections_made/self.total_questions*100:.1f}%)")
            print(f"Validations confirmed: {self.validations_confirmed} ({self.validations_confirmed/self.total_questions*100:.1f}%)")
            print(f"No pattern available: {self.no_pattern_available} ({self.no_pattern_available/self.total_questions*100:.1f}%)")

# Global stats tracker
validation_stats = ValidationStats()

def get_prediction_fanar_validated_with_stats(question, choice1, choice2, choice3, choice4, choice5=None, choice6=None):
    """
    Validation with statistics tracking
    """
    result = get_prediction_fanar_validated(question, choice1, choice2, choice3, choice4, choice5, choice6)
    
    # Log statistics based on the output
    # This is a simple heuristic - you can make it more sophisticated
    if "CORRECTED" in str(result):
        validation_stats.log_result("corrected")
    elif "VALIDATED" in str(result):
        validation_stats.log_result("validated")
    else:
        validation_stats.log_result("no_pattern")
    
    return result