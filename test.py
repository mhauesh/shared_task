import pandas as pd
import re
from collections import defaultdict, Counter
import os

# Create rules in the EXACT location where the calculator is looking
def fix_rules_location():
    # The calculator is looking here (from notebooks folder):
    target_path = 'results/prediction/enhanced_inheritance_rules.py'
    
    # Load training data
    data_file = '../data/Task1_QCM_Dev.csv'
    if not os.path.exists(data_file):
        print("❌ Training data not found")
        return False
    
    df = pd.read_csv(data_file)
    print(f"✅ Processing {len(df)} training examples")
    
    # Family extraction (same logic as your calculator)
    def extract_family_simple(question):
        family = {}
        patterns = {
            'wife': r'زوجة', 'husband': r'زوج(?!ة)', 'mother': r'أم', 'father': r'أب',
            'son': r'ابن', 'daughter': r'بنت', 'full_brother': r'أخ شقيق', 
            'full_sister': r'أخت شقيقة', 'paternal_brother': r'أخ من الأب',
            'paternal_sister': r'أخت من الأب'
        }
        
        for heir, pattern in patterns.items():
            matches = re.findall(rf'(\d+)?\s*{pattern}', question)
            count = 0
            for match in matches:
                count += int(match) if match and match.isdigit() else 1
            
            if heir in ['husband', 'father', 'mother'] and count > 1:
                count = 1
            elif heir == 'wife' and count > 4:
                count = 4
            elif count > 10:
                count = 1
                
            if count > 0:
                family[heir] = count
        return family
    
    def classify_question_simple(question):
        if 'نصيب' in question.lower(): return 'share_calculation'
        elif 'أسهم' in question.lower(): return 'total_shares'
        elif 'يرث' in question.lower(): return 'heir_identification'
        else: return 'general'
    
    # Extract patterns
    scenario_patterns = defaultdict(list)
    successful = 0
    
    for idx, row in df.iterrows():
        question = str(row['question'])
        label = row['label']
        family = extract_family_simple(question)
        question_type = classify_question_simple(question)
        
        if family:
            family_tuple = tuple(sorted(family.items()))
            scenario_key = str((family_tuple, question_type))
            scenario_patterns[scenario_key].append(label)
            successful += 1
    
    print(f"✅ Extracted {successful} patterns from {len(scenario_patterns)} unique scenarios")
    
    # Process patterns
    final_patterns = {}
    for key, answers in scenario_patterns.items():
        counter = Counter(answers)
        most_common, freq = counter.most_common(1)[0]
        final_patterns[key] = {
            'most_common_answer': most_common,
            'confidence': freq / len(answers),
            'frequency': freq
        }
    
    # Create directory and file in the RIGHT location (from notebooks perspective)
    os.makedirs('results/prediction', exist_ok=True)
    
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write('# Working inheritance rules - generated in correct location\n\n')
        
        f.write('inheritance_rules = {\n')
        f.write('    "husband": {"with_children": {"share": "1/4"}, "no_children": {"share": "1/2"}},\n')
        f.write('    "wife": {"with_children": {"share": "1/8"}, "no_children": {"share": "1/4"}},\n')
        f.write