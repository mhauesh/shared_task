import pandas as pd
import re
from collections import defaultdict, Counter
import json
import logging
import argparse
import sys
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class EnhancedInheritanceRuleGenerator:
    def __init__(self, data_path):
        """Initialize the rule generator and load the dataset."""
        self.data_path = data_path
        try:
        self.df = pd.read_csv(data_path)
        except FileNotFoundError:
            logging.error(f"File not found: {data_path}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error reading file {data_path}: {e}")
            sys.exit(1)
        # Validate required columns
        required_columns = {'question', 'label'}
        option_columns = {f'option{i}' for i in range(1, 7)}
        missing = required_columns - set(self.df.columns)
        if missing:
            logging.error(f"Missing required columns: {missing}")
            sys.exit(1)
        # Your existing patterns (keeping them - they're good!)
        self.heir_patterns = {
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
        # Initialize rule containers
        self.individual_heir_rules = defaultdict(lambda: defaultdict(Counter))
        self.family_scenario_rules = defaultdict(list)
        self.question_answer_patterns = defaultdict(list)
        self.mathematical_relationships = []
        self.ambiguous_cases = []
    
    def extract_family_composition(self, question: str) -> dict:
        """Extract full family composition with counts from a question string."""
        family_members = {}
        for heir, pattern in self.heir_patterns.items():
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
    
    def classify_question_type(self, question: str) -> str:
        """Classify the type of inheritance question."""
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
    
    def identify_target_heir(self, question: str) -> Optional[str]:
        """Identify which heir the question is asking about."""
        for heir, patterns in self.heir_patterns.items():
            if re.search(patterns, question) and 'نصيب' in question:
                return heir
        return None
    
    def detect_inheritance_conditions(self, family_composition: dict) -> list:
        """Detect specific inheritance conditions from family composition."""
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
        if len(family_composition) > 5:
            conditions.append('complex_family')
        return conditions
    
    def extract_rules(self):
        """Main extraction method for inheritance rules from the dataset."""
        logging.info(f"Processing {len(self.df)} questions...")
        for idx, row in self.df.iterrows():
            question = str(row.get('question', ''))
            label = row.get('label', '')
            # Ensure label is a string and a single character
            if isinstance(label, str) and len(label) == 1 and label in 'ABCDEF':
                label_idx = ord(label) - ord('A')
            else:
                label_idx = None
            # Extract options robustly
            options = [str(row.get(f'option{i}', '')) for i in range(1, 7)]
            answer = options[label_idx] if label_idx is not None and 0 <= label_idx < len(options) else ''
            # Extract family composition
            family_composition = self.extract_family_composition(question)
            # Classify question
            question_type = self.classify_question_type(question)
            # Detect conditions
            conditions = self.detect_inheritance_conditions(family_composition)
            # Individual heir rules (your original logic, enhanced)
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
            # Family scenario patterns (for validation)
            if family_composition:
                family_key = tuple(sorted(family_composition.items()))
                scenario_key = (family_key, question_type)
                self.family_scenario_rules[scenario_key].append(label)
                # Question-Answer patterns for direct validation
                target_heir = self.identify_target_heir(question)
                if target_heir:
                    pattern_key = (family_key, question_type, target_heir)
                    self.question_answer_patterns[pattern_key].append(label)
                # Mathematical relationships
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
                # Detect ambiguous cases
                if 'complex_family' in conditions or len(set(self.family_scenario_rules[scenario_key])) > 1:
                    self.ambiguous_cases.append({
                        'family_composition': dict(family_composition),
                        'question_type': question_type,
                        'question': question,
                        'conditions': conditions,
                        'answers_seen': list(set(self.family_scenario_rules[scenario_key]))
                    })
    
    def extract_fraction_from_answer(self, answer: str) -> Optional[str]:
        """Extract numerical fraction from answer text."""
        fraction_match = re.search(r'(\d+/\d+)', answer)
        if fraction_match:
            return fraction_match.group(1)
        for share, pattern in self.share_patterns.items():
            if re.search(pattern, answer):
                return self.share_to_fraction[share]
        return None
    
    def generate_validation_rules(self):
        """Generate rules in the format expected by your calculator."""
        inheritance_rules = {}
        for heir, cond_dict in self.individual_heir_rules.items():
            inheritance_rules[heir] = {}
            for cond, shares in cond_dict.items():
                if shares:
                    most_common_share, count = shares.most_common(1)[0]
                    confidence = count / sum(shares.values())
                    inheritance_rules[heir][cond] = {
                        'share': self.share_to_fraction[most_common_share],
                        'confidence': confidence,
                        'frequency': count
                    }
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
        ambiguous_answers = self.ambiguous_cases
        return inheritance_rules, scenario_patterns, qa_patterns, ambiguous_answers
    
    def save_rules(self, output_path: str):
        """Save rules in the format your calculator expects."""
        inheritance_rules, scenario_patterns, qa_patterns, ambiguous_answers = self.generate_validation_rules()
        # Save as Python file (your format)
        try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('# Auto-generated inheritance rules from MCQ data\n')
            f.write('# Enhanced for symbolic validation\n')
            f.write('# Compatible with SymbolicInheritanceCalculator\n\n')
            f.write('# Individual heir rules\n')
            f.write('inheritance_rules = ')
            f.write(repr(inheritance_rules))
            f.write('\n\n')
            f.write('# Family scenario patterns\n')
            f.write('scenario_patterns = ')
            f.write(repr(scenario_patterns))
            f.write('\n\n')
            f.write('# Question-answer patterns\n')
            f.write('qa_patterns = ')
            f.write(repr(qa_patterns))
            f.write('\n\n')
            f.write('# Ambiguous cases requiring special handling\n')
            f.write('ambiguous_answers = ')
            f.write(repr(ambiguous_answers))
            f.write('\n')
        except Exception as e:
            logging.error(f"Failed to write Python rules file: {e}")
        # Also save as JSON for backup
        json_path = output_path.replace('.py', '.json')
        try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'inheritance_rules': inheritance_rules,
                'scenario_patterns': scenario_patterns,
                'qa_patterns': qa_patterns,
                'ambiguous_answers': ambiguous_answers
            }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to write JSON rules file: {e}")
        logging.info(f'✅ Rules saved to:')
        logging.info(f'   Python: {output_path}')
        logging.info(f'   JSON: {json_path}')
        return inheritance_rules, scenario_patterns, qa_patterns, ambiguous_answers
    
    def print_summary(self):
        """Print extraction summary to the log."""
        logging.info("\n" + "="*50)
        logging.info("ENHANCED RULE EXTRACTION SUMMARY")
        logging.info("="*50)
        logging.info(f"📊 Individual heir rules: {len(self.individual_heir_rules)}")
        logging.info(f"🏠 Family scenario patterns: {len(self.family_scenario_rules)}")
        logging.info(f"❓ Question-answer patterns: {len(self.question_answer_patterns)}")
        logging.info(f"🧮 Mathematical relationships: {len(self.mathematical_relationships)}")
        logging.info(f"⚠️  Ambiguous cases detected: {len(self.ambiguous_cases)}")
        logging.info("\n🏆 Most common family compositions:")
        family_compositions = [family_key for (family_key, _) in self.family_scenario_rules.keys()]
        family_counter = Counter(family_compositions)
        for i, (family, count) in enumerate(family_counter.most_common(5)):
            logging.info(f"  {i+1}. {dict(family)} - {count} scenarios")
        if self.ambiguous_cases:
            logging.info(f"\n⚠️  Sample ambiguous cases:")
            for i, case in enumerate(self.ambiguous_cases[:3], 1):
                logging.info(f"  {i}. {case['family_composition']} - {len(case['answers_seen'])} different answers")

def main():
    """Main CLI entry point for EnhancedInheritanceRuleGenerator."""
    parser = argparse.ArgumentParser(description="Enhanced Inheritance Rule Generator")
    parser.add_argument('--data', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to output Python file')
    args = parser.parse_args()
    generator = EnhancedInheritanceRuleGenerator(args.data)
    generator.extract_rules()
    generator.save_rules(args.output)
    generator.print_summary()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # Basic test scaffolding for development
        logging.info("Running basic test mode...")
        # Use a small sample or mock data if available
        # Example: generator = EnhancedInheritanceRuleGenerator('sample.csv')
        # generator.extract_rules()
        # generator.print_summary()
        logging.info("No CLI arguments provided. Please use --data and --output for full run.")