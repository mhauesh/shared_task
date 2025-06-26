# ==============================================================================
# STEP 4: Create a quick rule generator script
# Save this as generate_rules_quick.py in your project root
# ==============================================================================

import pandas as pd
import re
from collections import defaultdict, Counter
import os

def generate_inheritance_rules_quick():
    """
    Quick rule generation from your training data
    Creates the enhanced_inheritance_rules.py file needed for validation
    """
    
    # Paths
    data_path = 'data/Task1_QCM_Dev.csv'  # Adjust if your path is different
    output_path = 'results/prediction/enhanced_inheritance_rules.py'
    
    print("🔍 Quick rule extraction starting...")
    
    # Check if data exists
    if not os.path.exists(data_path):
        # Try alternative paths
        alt_paths = [
            '../data/Task1_QCM_Dev.csv',
            'data/Task1/MCQs/Task1_MCQs_Dev.csv',
            '../data/Task1/MCQs/Task1_MCQs_Dev.csv'
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                data_path = alt_path
                break
        else:
            print(f"❌ Could not find training data file")
            print(f"   Looked in: {[data_path] + alt_paths}")
            return False
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} training examples from {data_path}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Basic heir patterns
    heir_patterns = {
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
    }
    
    # Extract basic family patterns and their answers
    family_patterns = defaultdict(list)
    qa_patterns = {}
    inheritance_rules = defaultdict(dict)
    
    for _, row in df.iterrows():
        question = str(row['question'])
        label = row['label']
        
        # Extract family composition
        family_composition = {}
        for heir, pattern in heir_patterns.items():
            # Look for numbers before heir mentions
            numbered_matches = re.findall(rf'(\d+)\s*{pattern}', question)
            simple_matches = re.findall(rf'(?<!\d)\s*{pattern}(?!\s*\d)', question)
            
            count = 0
            for match in numbered_matches:
                count += int(match)
            if simple_matches and count == 0:
                count = 1
            
            # Sanity checks
            if heir in ['husband', 'father', 'mother'] and count > 1:
                count = 1
            elif heir == 'wife' and count > 4:
                count = 4
            elif count > 10:
                count = 1
            
            if count > 0:
                family_composition[heir] = count
        
        # Classify question type
        question_type = 'general'
        if 'نصيب' in question.lower() or 'حصة' in question.lower():
            question_type = 'share_calculation'
        elif 'أسهم' in question.lower() or 'سهم' in question.lower():
            question_type = 'total_shares'
        elif 'يرث' in question.lower() or 'ترث' in question.lower():
            question_type = 'heir_identification'
        
        # Store patterns
        if family_composition:
            family_key = tuple(sorted(family_composition.items()))
            scenario_key = (family_key, question_type)
            family_patterns[str(scenario_key)].append(label)
            
            # Store QA patterns for specific heir questions
            target_heir = None
            for heir, pattern in heir_patterns.items():
                if re.search(pattern, question) and 'نصيب' in question:
                    target_heir = heir
                    break
            
            if target_heir:
                qa_key = str((family_key, question_type, target_heir))
                if qa_key not in qa_patterns:
                    qa_patterns[qa_key] = []
                qa_patterns[qa_key].append(label)
    
    # Process patterns to get most common answers
    scenario_patterns = {}
    for scenario_key, answers in family_patterns.items():
        answer_counter = Counter(answers)
        most_common_answer, count = answer_counter.most_common(1)[0]
        confidence = count / len(answers)
        
        scenario_patterns[scenario_key] = {
            'most_common_answer': most_common_answer,
            'confidence': confidence,
            'frequency': count,
            'total_occurrences': len(answers)
        }
    
    # Process QA patterns
    processed_qa_patterns = {}
    for qa_key, answers in qa_patterns.items():
        answer_counter = Counter(answers)
        most_common_answer, count = answer_counter.most_common(1)[0]
        confidence = count / len(answers)
        
        processed_qa_patterns[qa_key] = {
            'expected_answer': most_common_answer,
            'confidence': confidence,
            'frequency': count
        }
    
    # Create basic inheritance rules (simplified)
    inheritance_rules = {
        'husband': {
            'with_children': {'share': '1/4', 'confidence': 0.9},
            'no_children': {'share': '1/2', 'confidence': 0.9}
        },
        'wife': {
            'with_children': {'share': '1/8', 'confidence': 0.9},
            'no_children': {'share': '1/4', 'confidence': 0.9}
        },
        'mother': {
            'with_children': {'share': '1/6', 'confidence': 0.9},
            'no_children': {'share': '1/3', 'confidence': 0.9}
        },
        'daughter': {
            'single': {'share': '1/2', 'confidence': 0.9},
            'multiple': {'share': '2/3', 'confidence': 0.9}
        }
    }
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the rules file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('# Auto-generated inheritance rules for symbolic validation\n')
        f.write('# Generated from training data\n\n')
        
        f.write('# Basic inheritance rules\n')
        f.write('inheritance_rules = ')
        f.write(repr(inheritance_rules))
        f.write('\n\n')
        
        f.write('# Family scenario patterns\n')
        f.write('scenario_patterns = ')
        f.write(repr(scenario_patterns))
        f.write('\n\n')
        
        f.write('# Question-answer patterns\n')
        f.write('qa_patterns = ')
        f.write(repr(processed_qa_patterns))
        f.write('\n\n')
        
        f.write('# Ambiguous cases (empty for now)\n')
        f.write('ambiguous_answers = []\n')
    
    print(f"✅ Rules generated successfully!")
    print(f"   📊 Scenario patterns: {len(scenario_patterns)}")
    print(f"   📊 QA patterns: {len(processed_qa_patterns)}")
    print(f"   📁 Saved to: {output_path}")
    
    return True

if __name__ == "__main__":
    generate_inheritance_rules_quick()