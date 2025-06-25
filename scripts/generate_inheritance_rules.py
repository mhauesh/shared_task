import pandas as pd
import re
from collections import defaultdict, Counter
import json
import argparse
import sys

class EnhancedInheritanceRuleGenerator:
    """
    Extracts, generalizes, and outputs Islamic inheritance rules from MCQ data.
    Includes robust family composition parsing, synonym handling, ambiguous answer flagging,
    and outputs both string and float shares for calculator use.
    """
    def __init__(self, data_path):
        """
        Initialize the rule generator with the path to the MCQ data.
        """
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        # Expanded heir patterns with synonyms
        self.heir_patterns = {
            'wife': r'زوجة|حرم',
            'husband': r'زوج(?!ة)',
            'mother': r'أم|والدة',
            'father': r'أب|والد',
            'son': r'ابن|ولد',
            'daughter': r'بنت|ابنة',
            'full_brother': r'أخ شقيق',
            'full_sister': r'أخت شقيقة',
            'paternal_brother': r'أخ من الأب',
            'paternal_sister': r'أخت من الأب',
            'maternal_brother': r'أخ من الأم',
            'maternal_sister': r'أخت من الأم',
        }
        self.share_patterns = {
            'half': r'النصف|1/2',
            'third': r'الثلث|1/3',
            'quarter': r'الربع|1/4',
            'eighth': r'الثمن|1/8',
            'sixth': r'السدس|1/6',
            'two_thirds': r'الثلثان|2/3',
            'remainder': r'الباقي|تعصيبا',
        }
        self.share_to_fraction = {
            'half': '1/2',
            'third': '1/3',
            'quarter': '1/4',
            'eighth': '1/8',
            'sixth': '1/6',
            'two_thirds': '2/3',
            'remainder': 'residual',
        }
        self.ambiguous_patterns = [
            r'تعصيبا', r'الباقي', r'للذكر مثل حظ الأنثيين', r'محجوب', r'لا يرث', r'لا شيء', r'blocked', r'رد', r'عول'
        ]
        # Initialize rule containers
        self.individual_heir_rules = defaultdict(lambda: defaultdict(Counter))
        self.family_scenario_rules = defaultdict(list)
        self.question_answer_patterns = defaultdict(list)
        self.mathematical_relationships = []
        self.ambiguous_answers = []

    def fraction_to_float(self, frac_str):
        """
        Convert a fraction string (e.g., '1/2') to a float. Returns None if not possible.
        """
        if isinstance(frac_str, (int, float)):
            return float(frac_str)
        if isinstance(frac_str, str):
            if '/' in frac_str:
                try:
                    num, denom = frac_str.split('/')
                    return float(num) / float(denom)
                except Exception:
                    return None
            try:
                return float(frac_str)
            except Exception:
                return None
        return None

    def extract_family_composition(self, question):
        """
        Extract full family composition with counts from the question text.
        Handles synonyms and number mentions.
        """
        family_members = {}
        for heir, pattern in self.heir_patterns.items():
            numbered_matches = re.findall(rf'(\d+)\s*{pattern}', question)
            simple_matches = re.findall(rf'(?<!\d)\s*{pattern}(?!\s*\d)', question)
            count = 0
            for match in numbered_matches:
                if match and match.isdigit():
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

    def classify_question_type(self, question):
        """
        Classify the type of inheritance question for better rule extraction.
        """
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

    def identify_target_heir(self, question):
        """
        Identify which heir the question is asking about.
        """
        for heir, patterns in self.heir_patterns.items():
            if re.search(patterns, question) and 'نصيب' in question:
                return heir
        return None

    def detect_inheritance_conditions(self, family_composition):
        """
        Detect specific inheritance conditions from family composition.
        """
        conditions = []
        if any(heir in family_composition for heir in ['son', 'daughter']):
            conditions.append('has_children')
        else:
            conditions.append('no_children')
        if 'son' in family_composition and 'daughter' in family_composition:
            conditions.append('mixed_children')
        elif 'son' in family_composition:
            conditions.append('sons_only')
        elif 'daughter' in family_composition:
            conditions.append('daughters_only')
        sibling_count = sum(family_composition.get(heir, 0) for heir in 
                          ['full_brother', 'full_sister', 'paternal_brother', 'paternal_sister'])
        if sibling_count > 0:
            conditions.append('has_siblings')
        if family_composition.get('wife', 0) > 1:
            conditions.append('multiple_wives')
        return conditions

    def is_ambiguous_answer(self, answer):
        """
        Check if the answer contains ambiguous or composite inheritance terms.
        """
        for pat in self.ambiguous_patterns:
            if re.search(pat, answer):
                return True
        return False

    def extract_rules(self):
        """
        Main extraction method. Parses each question, extracts rules, and flags ambiguous answers.
        """
        print(f"Processing {len(self.df)} questions...")
        for idx, row in self.df.iterrows():
            question = str(row['question'])
            label = row['label']
            options = [str(row.get(f'option{i}', '')) for i in range(1, 7)]
            answer = options[ord(label) - ord('A')] if label in 'ABCDEF' else ''
            family_composition = self.extract_family_composition(question)
            question_type = self.classify_question_type(question)
            conditions = self.detect_inheritance_conditions(family_composition)
            ambiguous = self.is_ambiguous_answer(answer)
            if ambiguous:
                self.ambiguous_answers.append({
                    'question': question,
                    'answer': answer,
                    'label': label,
                    'family_composition': family_composition,
                    'conditions': conditions
                })
            for heir, heir_pat in self.heir_patterns.items():
                if re.search(heir_pat, question):
                    main_condition = 'with_children' if 'has_children' in conditions else 'no_children'
                    found_share = None
                    for share, share_pat in self.share_patterns.items():
                        if re.search(share_pat, answer):
                            found_share = share
                            break
                    if found_share:
                        self.individual_heir_rules[heir][main_condition][found_share] += 1
            if family_composition:
                family_key = tuple(sorted(family_composition.items()))
                scenario_key = (family_key, question_type)
                self.family_scenario_rules[scenario_key].append(label)
                target_heir = self.identify_target_heir(question)
                if target_heir:
                    pattern_key = (family_key, question_type, target_heir)
                    self.question_answer_patterns[pattern_key].append(label)
                if question_type == 'share_calculation' and target_heir:
                    share_fraction = self.extract_fraction_from_answer(answer)
                    if share_fraction:
                        self.mathematical_relationships.append({
                            'family_composition': dict(family_composition),
                            'target_heir': target_heir,
                            'expected_share': share_fraction,
                            'correct_option': label,
                            'conditions': conditions
                        })

    def extract_fraction_from_answer(self, answer):
        """
        Extract numerical fraction from answer text, or Arabic fraction term.
        """
        fraction_match = re.search(r'(\d+/\d+)', answer)
        if fraction_match:
            return fraction_match.group(1)
        for share, pattern in self.share_patterns.items():
            if re.search(pattern, answer):
                return self.share_to_fraction[share]
        return None

    def generate_validation_rules(self):
        """
        Generate rules for symbolic validation, including all observed shares with frequencies and confidence.
        """
        validation_rules = {}
        generalized_rules = {}
        for heir, cond_dict in self.individual_heir_rules.items():
            generalized_rules[heir] = {}
            for cond, shares in cond_dict.items():
                total = sum(shares.values())
                all_shares = []
                for share, count in shares.items():
                    share_str = self.share_to_fraction.get(share, share)
                    share_float = self.fraction_to_float(share_str)
                    confidence = count / total if total else 0
                    all_shares.append({
                        'share': share_str,
                        'share_float': share_float,
                        'frequency': count,
                        'confidence': confidence
                    })
                # Sort by frequency descending
                all_shares = sorted(all_shares, key=lambda x: -x['frequency'])
                generalized_rules[heir][cond] = all_shares
        validation_rules['individual_heir_rules'] = generalized_rules
        scenario_patterns = {}
        for scenario_key, answers in self.family_scenario_rules.items():
            answer_counter = Counter(answers)
            most_common_answer, count = answer_counter.most_common(1)[0]
            confidence = count / len(answers)
            scenario_patterns[str(scenario_key)] = {
                'most_common_answer': most_common_answer,
                'confidence': confidence,
                'frequency': count,
                'total_occurrences': len(answers)
            }
        validation_rules['scenario_patterns'] = scenario_patterns
        qa_patterns = {}
        for pattern_key, answers in self.question_answer_patterns.items():
            answer_counter = Counter(answers)
            most_common_answer, count = answer_counter.most_common(1)[0]
            confidence = count / len(answers)
            qa_patterns[str(pattern_key)] = {
                'expected_answer': most_common_answer,
                'confidence': confidence,
                'frequency': count
            }
        validation_rules['qa_patterns'] = qa_patterns
        validation_rules['mathematical_relationships'] = self.mathematical_relationships
        validation_rules['ambiguous_answers'] = self.ambiguous_answers
        return validation_rules

    def save_rules(self, output_path):
        """
        Save comprehensive rules for validation as both JSON and Python files.
        """
        validation_rules = self.generate_validation_rules()
        json_path = output_path.replace('.py', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(validation_rules, f, ensure_ascii=False, indent=2)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('# Auto-generated inheritance rules from MCQ data\n')
            f.write('# Enhanced for symbolic validation\n\n')
            f.write('inheritance_rules = ')
            f.write(repr(validation_rules['individual_heir_rules']))
            f.write('\n\n')
            f.write('scenario_patterns = ')
            f.write(repr(validation_rules['scenario_patterns']))
            f.write('\n\n')
            f.write('qa_patterns = ')
            f.write(repr(validation_rules['qa_patterns']))
            f.write('\n\n')
            f.write('ambiguous_answers = ')
            f.write(repr(validation_rules['ambiguous_answers']))
            f.write('\n')
        print(f'Enhanced rules saved to:')
        print(f'  JSON: {json_path}')
        print(f'  Python: {output_path}')
        return validation_rules

    def print_summary(self):
        """
        Print extraction summary, including ambiguous/composite rules.
        """
        print("\n" + "="*50)
        print("ENHANCED RULE EXTRACTION SUMMARY")
        print("="*50)
        print(f"📊 Individual heir rules: {len(self.individual_heir_rules)}")
        print(f"🏠 Family scenario patterns: {len(self.family_scenario_rules)}")
        print(f"❓ Question-answer patterns: {len(self.question_answer_patterns)}")
        print(f"🧮 Mathematical relationships: {len(self.mathematical_relationships)}")
        print(f"⚠️ Ambiguous/composite answers: {len(self.ambiguous_answers)}")
        print("\n🏆 Most common family compositions:")
        family_compositions = [family_key for (family_key, _) in self.family_scenario_rules.keys()]
        family_counter = Counter(family_compositions)
        for i, (family, count) in enumerate(family_counter.most_common(5)):
            print(f"  {i+1}. {dict(family)} - {count} scenarios")
        if self.ambiguous_answers:
            print("\n⚠️ Ambiguous/composite answer examples:")
            for ex in self.ambiguous_answers[:5]:
                print(f"  Q: {ex['question']}")
                print(f"  A: {ex['answer']} | Family: {ex['family_composition']} | Conditions: {ex['conditions']}")
                print("  ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and generalize Islamic inheritance rules from MCQ data.")
    parser.add_argument('--input', type=str, default='../data/Task1/MCQs/Task1_MCQs_Dev.csv', help='Path to MCQ CSV file')
    parser.add_argument('--output', type=str, default='../results/prediction/enhanced_inheritance_rules.py', help='Path to output Python file')
    args = parser.parse_args()
    generator = EnhancedInheritanceRuleGenerator(args.input)
    generator.extract_rules()
    validation_rules = generator.save_rules(args.output)
    generator.print_summary()