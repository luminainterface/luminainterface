#!/usr/bin/env python3
"""
Constraint Mask - Actionable Feedback for Logic Puzzles
=======================================================

Provides boolean constraint masks and actionable feedback for zebra-type puzzles
instead of vague "mostly correct" critiques.
"""

import re
from typing import Dict, List, Any, Optional, Tuple

class ConstraintMask:
    """Generate actionable constraint feedback for logic puzzles"""
    
    def __init__(self):
        self.colors = ['red', 'blue', 'green', 'yellow', 'white']
        self.pets = ['cat', 'dog', 'bird', 'fish', 'horse']
        self.drinks = ['tea', 'coffee', 'milk', 'juice', 'water']
        self.nationalities = ['british', 'swedish', 'danish', 'norwegian', 'german']
        self.cigarettes = ['dunhill', 'blend', 'pall mall', 'prince', 'blue master']
    
    def parse_zebra_solution(self, answer: str) -> Dict[str, Dict[str, str]]:
        """Parse zebra puzzle solution from answer text"""
        houses = {}
        
        # Initialize houses
        for i in range(1, 6):  # Support up to 5 houses
            houses[f'house_{i}'] = {
                'color': None,
                'pet': None,
                'drink': None,
                'nationality': None,
                'cigarette': None
            }
        
        answer_lower = answer.lower()
        
        # Extract house assignments using various patterns
        patterns = [
            r'house\s*(\d+)[:\s]*([a-z\s]+)',
            r'(\d+)[:\s]*([a-z\s]+)',
            r'house\s*(\d+).*?([a-z]+).*?([a-z]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, answer_lower)
            for match in matches:
                if len(match) >= 2:
                    house_num = int(match[0])
                    if 1 <= house_num <= 5:
                        # Parse attributes from the match
                        text = ' '.join(match[1:]).strip()
                        self._extract_attributes(houses[f'house_{house_num}'], text)
        
        return houses
    
    def _extract_attributes(self, house: Dict[str, str], text: str) -> None:
        """Extract attributes from text and assign to house"""
        words = text.split()
        
        for word in words:
            word = word.strip(',.')
            if word in self.colors and not house['color']:
                house['color'] = word
            elif word in self.pets and not house['pet']:
                house['pet'] = word
            elif word in self.drinks and not house['drink']:
                house['drink'] = word
            elif word in self.nationalities and not house['nationality']:
                house['nationality'] = word
            elif word in self.cigarettes and not house['cigarette']:
                house['cigarette'] = word
    
    def check_mini_zebra_constraints(self, houses: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """
        Check mini zebra puzzle constraints (3 houses version)
        
        Constraints:
        1. The Red house is immediately to the left of the Blue house
        2. The Dog lives in house 2
        3. The Bird does not live in the Green house
        """
        
        constraint_results = {
            'constraint_1': False,  # Red immediately left of Blue
            'constraint_2': False,  # Dog in house 2
            'constraint_3': False,  # Bird not in Green house
            'mask': [0, 0, 0],      # Boolean mask for constraints
            'violations': [],       # Specific violations
            'actionable_feedback': []  # What to fix
        }
        
        # Check constraint 1: Red immediately to the left of Blue
        red_blue_valid = False
        if (houses.get('house_1', {}).get('color') == 'red' and 
            houses.get('house_2', {}).get('color') == 'blue'):
            red_blue_valid = True
        elif (houses.get('house_2', {}).get('color') == 'red' and 
              houses.get('house_3', {}).get('color') == 'blue'):
            red_blue_valid = True
        
        constraint_results['constraint_1'] = red_blue_valid
        constraint_results['mask'][0] = 1 if red_blue_valid else 0
        
        if not red_blue_valid:
            constraint_results['violations'].append("Red house is not immediately to the left of Blue house")
            constraint_results['actionable_feedback'].append("CONSTRAINT 1 VIOLATION: Place Red house immediately left of Blue house (Red in house 1→Blue in house 2, OR Red in house 2→Blue in house 3)")
        
        # Check constraint 2: Dog in house 2
        dog_in_house_2 = houses.get('house_2', {}).get('pet') == 'dog'
        constraint_results['constraint_2'] = dog_in_house_2
        constraint_results['mask'][1] = 1 if dog_in_house_2 else 0
        
        if not dog_in_house_2:
            current_pet = houses.get('house_2', {}).get('pet', 'unknown')
            constraint_results['violations'].append(f"Dog is not in house 2 (currently: {current_pet})")
            constraint_results['actionable_feedback'].append(f"CONSTRAINT 2 VIOLATION: Move Dog to house 2 (currently has {current_pet})")
        
        # Check constraint 3: Bird not in Green house
        bird_in_green = False
        green_house = None
        bird_house = None
        
        for house_id, house_data in houses.items():
            if house_data.get('color') == 'green':
                green_house = house_id
            if house_data.get('pet') == 'bird':
                bird_house = house_id
        
        if green_house and bird_house and green_house == bird_house:
            bird_in_green = True
        
        constraint_results['constraint_3'] = not bird_in_green
        constraint_results['mask'][2] = 1 if not bird_in_green else 0
        
        if bird_in_green:
            constraint_results['violations'].append("Bird is living in the Green house")
            constraint_results['actionable_feedback'].append(f"CONSTRAINT 3 VIOLATION: Move Bird out of Green house ({green_house}) to a different house")
        
        # Overall assessment
        constraint_results['all_satisfied'] = all(constraint_results['mask'])
        constraint_results['satisfaction_count'] = sum(constraint_results['mask'])
        constraint_results['satisfaction_rate'] = sum(constraint_results['mask']) / 3
        
        return constraint_results
    
    def generate_actionable_critique(self, houses: Dict[str, Dict[str, str]], 
                                   constraint_type: str = "mini_zebra") -> str:
        """Generate actionable critique with specific fixes"""
        
        if constraint_type == "mini_zebra":
            results = self.check_mini_zebra_constraints(houses)
        else:
            return "Unknown constraint type"
        
        if results['all_satisfied']:
            return "OK - All constraints satisfied correctly."
        
        critique_parts = [
            f"CONSTRAINT VIOLATIONS DETECTED: {3 - results['satisfaction_count']}/3 constraints violated",
            f"Constraint Mask: {results['mask']} (1=satisfied, 0=violated)",
            "",
            "SPECIFIC VIOLATIONS:"
        ]
        
        for violation in results['violations']:
            critique_parts.append(f"❌ {violation}")
        
        critique_parts.extend([
            "",
            "ACTIONABLE FIXES REQUIRED:"
        ])
        
        for fix in results['actionable_feedback']:
            critique_parts.append(f"🔧 {fix}")
        
        critique_parts.extend([
            "",
            f"NEXT STEP: Revise the solution to address the {len(results['violations'])} violations above.",
            "Provide a new house arrangement that satisfies ALL constraints."
        ])
        
        return "\n".join(critique_parts)
    
    def check_full_zebra_constraints(self, houses: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """Check full Einstein's zebra puzzle constraints (15 constraints)"""
        
        # This would implement all 15 constraints of the full zebra puzzle
        # For now, implementing a subset as example
        
        constraints = {
            'brit_red': False,      # The Brit lives in the red house
            'swede_dog': False,     # The Swede keeps dogs as pets
            'dane_tea': False,      # The Dane drinks tea
            'green_left_white': False,  # Green house is left of white house
            'green_coffee': False,  # Green house owner drinks coffee
            # ... more constraints
        }
        
        # Check Brit in red house
        for house_id, house_data in houses.items():
            if (house_data.get('nationality') == 'british' and 
                house_data.get('color') == 'red'):
                constraints['brit_red'] = True
                break
        
        # Check Swede with dog
        for house_id, house_data in houses.items():
            if (house_data.get('nationality') == 'swedish' and 
                house_data.get('pet') == 'dog'):
                constraints['swede_dog'] = True
                break
        
        # Check Dane drinks tea
        for house_id, house_data in houses.items():
            if (house_data.get('nationality') == 'danish' and 
                house_data.get('drink') == 'tea'):
                constraints['dane_tea'] = True
                break
        
        # More constraint checks would go here...
        
        return {
            'constraints': constraints,
            'mask': [1 if satisfied else 0 for satisfied in constraints.values()],
            'satisfaction_rate': sum(constraints.values()) / len(constraints)
        }

def test_constraint_mask():
    """Test the constraint mask with a sample zebra solution"""
    mask = ConstraintMask()
    
    # Test with a violating solution
    test_answer = """
    House 1: Green, Cat
    House 2: Red, Bird  
    House 3: Blue, Dog
    """
    
    houses = mask.parse_zebra_solution(test_answer)
    print("Parsed houses:", houses)
    
    critique = mask.generate_actionable_critique(houses, "mini_zebra")
    print("\nActionable Critique:")
    print(critique)
    
    return houses, critique

if __name__ == "__main__":
    test_constraint_mask() 