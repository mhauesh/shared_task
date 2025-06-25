import json
import re
import os
from collections import Counter

# Load rules from JSON (path can be changed as needed)
RULES_PATH = os.path.join(os.path.dirname(__file__), '../results/prediction/enhanced_inheritance_rules.json')
with open(RULES_PATH, encoding='utf-8') as f:
    rules = json.load(f)

# Heir patterns (should match those in symbolic_calculator.py)
HEIR_PATTERNS = {
    'wife': r'زوجة',
    'husband': r'زوج(?!ة)',
    'mother': r'أم',
    'father': r'أب',
    'son': r'ابن',
    'daughter': r'بنت',
    'full_brother': r'أخ شقيق',
    'full_sister': r'أخت شقيقة',
    'paternal_brother': r'أخ من الأب',
    'paternal_sister': r'أخت من الأب',
    'maternal_brother': r'أخ من الأم',
    'maternal_sister': r'أخت من الأم',
}

# Helper: parse family composition
def extract_family_composition(question):
    family_members = {}
    for heir, pattern in HEIR_PATTERNS.items():
        numbered_matches = re.findall(rf'(\d+)\s*{pattern}', question)
        simple_matches = re.findall(rf'(?<!\d)\s*{pattern}(?!\s*\d)', question)
        count = 0
        for match in numbered_matches:
            count += int(match)
        if simple_matches and count == 0:
            count = 1
        if heir in ['husband', 'father', 'mother'] and count > 1:
            count = 1
        elif heir == 'wife' and count > 4:
            count = 4
        elif count > 10:
            count = 1
        if count > 0:
            family_members[heir] = count
    return family_members

# Helper: classify question type
def classify_question_type(question):
    q_lower = question.lower()
    if 'نصيب' in q_lower or 'حصة' in q_lower:
        return 'share_calculation'
    elif 'أسهم' in q_lower or 'سهم' in q_lower:
        return 'total_shares'
    elif 'يرث' in q_lower or 'ترث' in q_lower or 'الوارث' in q_lower:
        return 'heir_identification'
    elif 'حجب' in q_lower or 'محجوب' in q_lower:
        return 'blocking'
    elif 'أصل المسألة' in q_lower:
        return 'problem_base'
    else:
        return 'general'

# Main validation function
def validate_fanar_answer(question, options, fanar_answer):
    """
    Validate Fanar answer using symbolic rules.
    Returns: (validated_answer, decision_reason)
    """
    family = extract_family_composition(question)
    qtype = classify_question_type(question)
    family_key = tuple(sorted(family.items()))
    scenario_key = str((family_key, qtype))
    # Try scenario_patterns first
    scenario_patterns = rules.get('scenario_patterns', {})
    if scenario_key in scenario_patterns:
        expected = scenario_patterns[scenario_key]['most_common_answer']
        if fanar_answer == expected:
            return fanar_answer, 'accepted (matches symbolic rule)'
        else:
            # If confidence is high, override
            if scenario_patterns[scenario_key]['confidence'] > 0.8:
                return expected, 'overridden (symbolic rule high confidence)'
            else:
                return fanar_answer, 'ambiguous (symbolic rule low confidence)'
    # Fallback: accept original
    return fanar_answer, 'no rule found (accepted)' 