import os, re
import openai
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from groq import Groq
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


def get_prediction_fanar_validated(
    question, choice1, choice2, choice3, choice4, choice5=None, choice6=None,
    model_version="Islamic-RAG", max_retries=3
):
    """Inference using Fanar API, then validate the answer."""
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
                    # Build options list for validation
                    options = [c[1] for c in choices]
                    validated, reason = validate_fanar_answer(question, options, cleaned_result)
                    print(f"🔎 Fanar validated: {validated} ({reason})")
                    return validated
            else:
                print(f"❌ Fanar API Error: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Fanar Error: {e}")
            return None

    print("❌ Failed after retries.")
    return None
