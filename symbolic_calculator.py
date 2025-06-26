import os
import sys
import importlib.util
import re
from typing import Dict, Optional

class SymbolicInheritanceCalculator:
    """
    Complete symbolic calculator for Islamic inheritance validation
    """
    
    def __init__(self, rules_path='results/prediction/enhanced_inheritance_rules.py'):
        """Initialize calculator and load rules"""
        self.rules_path = os.path.abspath(rules_path)
        self.inheritance_rules = {}
        self.scenario_patterns = {}
        self.qa_patterns = {}
        self.ambiguous_answers = []
        
        # Heir patterns for family composition extraction
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
        
        # Load rules from file
        self._load_rules()
    
    def _load_rules(self):
        """Load rules from the Python file"""
        try:
            if not os.path.exists(self.rules_path):
                print(f"⚠️ Rules file not found: {self.rules_path}")
                print("   Using fallback basic rules")
                self._create_fallback_rules()
                return
            
            # Import the rules module
            spec = importlib.util.spec_from_file_location('rules_module', self.rules_path)
            rules_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rules_module)
            
            # Load rules
            self.inheritance_rules = getattr(rules_module, 'inheritance_rules', {})
            self.scenario_patterns = getattr(rules_module, 'scenario_patterns', {})
            self.qa_patterns = getattr(rules_module, 'qa_patterns', {})
            self.ambiguous_answers = getattr(rules_module, 'ambiguous_answers', [])
            
            print(f"✅ Loaded rules: {len(self.scenario_patterns)} scenario patterns")
            
        except Exception as e:
            print(f"⚠️ Error loading rules: {e}")
            print("   Using fallback basic rules")
            self._create_fallback_rules()
    
    def _create_fallback_rules(self):
        """Create basic fallback rules if file loading fails"""
        self.inheritance_rules = {
            "husband": {"with_children": {"share": "1/4"}, "no_children": {"share": "1/2"}},
            "wife": {"with_children": {"share": "1/8"}, "no_children": {"share": "1/4"}},
            "mother": {"with_children": {"share": "1/6"}, "no_children": {"share": "1/3"}}
        }
        self.scenario_patterns = {}
        self.qa_patterns = {}
        self.ambiguous_answers = []
    
    def extract_family_composition(self, question: str) -> Dict[str, int]:
        """Extract family composition from Arabic question"""
        family_composition = {}
        
        for heir, pattern in self.heir_patterns.items():
            # Look for explicit numbers before heir mentions
            numbered_matches = re.findall(rf'(\d+)\s*{pattern}', question)
            simple_matches = re.findall(rf'(?<!\d)\s*{pattern}(?!\s*\d)', question)
            
            count = 0
            
            # Count numbered mentions
            for match in numbered_matches:
                if match.isdigit():
                    count += int(match)
            
            # Count simple mentions (without numbers)
            if simple_matches and count == 0:
                count = 1
            
            # Apply sanity checks
            if heir in ['husband', 'father', 'mother'] and count > 1:
                count = 1  # These can only be 1 in Islamic inheritance
            elif heir == 'wife' and count > 4:
                count = 4  # Max 4 wives in Islam
            elif count > 10:
                count = 1  # Probably a parsing error
            
            if count > 0:
                family_composition[heir] = count
        
        return family_composition
    
    def classify_question_type(self, question: str) -> str:
        """Classify the type of inheritance question"""
        question_lower = question.lower()
        
        if 'نصيب' in question_lower or 'حصة' in question_lower:
            return 'share_calculation'
        elif 'أسهم' in question_lower or 'سهم' in question_lower:
            return 'total_shares'
        elif 'يرث' in question_lower or 'ترث' in question_lower or 'الوارث' in question_lower:
            return 'heir_identification'
        elif 'حجب' in question_lower or 'محجوب' in question_lower:
            return 'blocking'
        elif 'أصل المسألة' in question_lower:
            return 'problem_base'
        else:
            return 'general'
    
    def identify_target_heir(self, question: str) -> Optional[str]:
        """Identify which specific heir the question is asking about"""
        for heir, pattern in self.heir_patterns.items():
            if re.search(pattern, question) and 'نصيب' in question:
                return heir
        return None
    
    def lookup_pattern_answer(self, family_composition: Dict[str, int], 
                            question_type: str, target_heir: str = None) -> Optional[str]:
        """
        Look up expected answer from training patterns
        """
        if not family_composition:
            return None
        
        # Create pattern keys
        family_key = tuple(sorted(family_composition.items()))
        
        # Try QA patterns first (most specific)
        if target_heir and self.qa_patterns:
            qa_key = str((family_key, question_type, target_heir))
            for key, pattern_data in self.qa_patterns.items():
                if qa_key in key:
                    if isinstance(pattern_data, dict):
                        confidence = pattern_data.get('confidence', 0)
                        if confidence > 0.5:  # Minimum confidence threshold
                            return pattern_data.get('expected_answer')
                    elif isinstance(pattern_data, str):
                        return pattern_data
        
        # Try scenario patterns (less specific)
        if self.scenario_patterns:
            scenario_key = str((family_key, question_type))
            for key, pattern_data in self.scenario_patterns.items():
                if scenario_key in key:
                    if isinstance(pattern_data, dict):
                        confidence = pattern_data.get('confidence', 0)
                        if confidence > 0.4:  # Lower threshold for scenario patterns
                            return pattern_data.get('most_common_answer')
                    elif isinstance(pattern_data, str):
                        return pattern_data
        
        return None
    
    def calculate_shares(self, family_composition: Dict[str, int]) -> Dict[str, str]:
        """
        Calculate expected inheritance shares (basic implementation)
        Returns shares as fractions or percentages
        """
        if not family_composition:
            return {}
        
        shares = {}
        
        # Determine conditions
        has_children = any(heir in family_composition for heir in ['son', 'daughter'])
        
        # Calculate basic shares using inheritance rules
        for heir, count in family_composition.items():
            if heir in self.inheritance_rules:
                heir_rules = self.inheritance_rules[heir]
                
                if heir == 'husband':
                    condition = 'with_children' if has_children else 'no_children'
                    if condition in heir_rules:
                        shares[heir] = heir_rules[condition].get('share', 'unknown')
                
                elif heir == 'wife':
                    condition = 'with_children' if has_children else 'no_children'
                    if condition in heir_rules:
                        base_share = heir_rules[condition].get('share', 'unknown')
                        # Divide among multiple wives if needed
                        if count > 1 and base_share != 'unknown':
                            shares[heir] = f"{base_share}//{count}"
                        else:
                            shares[heir] = base_share
                
                elif heir == 'mother':
                    condition = 'with_children' if has_children else 'no_children'
                    if condition in heir_rules:
                        shares[heir] = heir_rules[condition].get('share', 'unknown')
        
        return shares
    
    def validate_answer(self, question: str, llm_answer: str, options: list) -> dict:
        """
        Complete validation of LLM answer against Islamic inheritance rules
        """
        family_composition = self.extract_family_composition(question)
        question_type = self.classify_question_type(question)
        target_heir = self.identify_target_heir(question)
        
        # Look up expected answer
        expected_answer = self.lookup_pattern_answer(family_composition, question_type, target_heir)
        
        # Determine validation result
        is_valid = True
        confidence = 0.5
        reasoning = "No specific validation rule available"
        
        if expected_answer:
            if expected_answer == llm_answer:
                is_valid = True
                confidence = 0.8
                reasoning = "Answer matches training pattern"
            else:
                is_valid = False
                confidence = 0.7
                reasoning = f"Pattern suggests {expected_answer}, but LLM predicted {llm_answer}"
        
        return {
            'is_valid': is_valid,
            'confidence': confidence,
            'expected_answer': expected_answer,
            'family_composition': family_composition,
            'question_type': question_type,
            'target_heir': target_heir,
            'reasoning': reasoning
        }
    
    def handle_ambiguous_case(self, scenario: dict) -> str:
        """Handle ambiguous inheritance cases"""
        family_composition = scenario.get('family_composition', {})
        
        # Simple ambiguity detection
        ambiguity_indicators = []
        
        if len(family_composition) > 5:
            ambiguity_indicators.append("Complex family structure")
        
        if family_composition.get('wife', 0) > 1:
            ambiguity_indicators.append("Multiple wives")
        
        mixed_siblings = sum(1 for heir in family_composition.keys() 
                           if 'brother' in heir or 'sister' in heir)
        if mixed_siblings > 2:
            ambiguity_indicators.append("Multiple sibling types")
        
        if ambiguity_indicators:
            return f"Ambiguity detected: {'; '.join(ambiguity_indicators)}"
        
        return "No significant ambiguity detected"

# Factory function for easy creation
def create_calculator(rules_path: str = 'results/prediction/enhanced_inheritance_rules.py'):
    """Create and return a symbolic calculator instance"""
    return SymbolicInheritanceCalculator(rules_path)

# Test function
def test_calculator():
    """Test the calculator with sample questions"""
    calc = SymbolicInheritanceCalculator()
    
    test_questions = [
        "توفي عن زوجة وابن واحد. ما نصيب الزوجة؟",
        "توفيت عن زوج وبنتين. ما نصيب الزوج؟"
    ]
    
    for question in test_questions:
        family = calc.extract_family_composition(question)
        q_type = calc.classify_question_type(question)
        target = calc.identify_target_heir(question)
        
        print(f"Question: {question}")
        print(f"Family: {family}")
        print(f"Type: {q_type}, Target: {target}")
        print("-" * 50)

if __name__ == "__main__":
    test_calculator()