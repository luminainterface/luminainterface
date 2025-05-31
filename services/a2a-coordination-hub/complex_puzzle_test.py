#!/usr/bin/env python3
"""
Complex Puzzle Test Suite for Enhanced A2A Coordination Hub
Tests the system with incredibly complex logic puzzles including trick versions
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

class ComplexPuzzleTester:
    def __init__(self, base_url: str = "http://localhost:8891"):
        self.base_url = base_url
        self.session = None
    
    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def test_complex_query(self, question_id: str, query: str, expected_solvable: bool = True):
        """Test a complex query and analyze the problem-solving approach"""
        print(f"\n{'='*80}")
        print(f"🧩 TESTING COMPLEX QUESTION {question_id}")
        print(f"{'='*80}")
        print(f"📝 Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"🎯 Expected Solvable: {expected_solvable}")
        print(f"{'='*80}\n")
        
        session = await self.get_session()
        
        payload = {
            "query": query,
            "enable_mathematical_validation": True
        }
        
        start_time = time.time()
        
        try:
            async with session.post(f"{self.base_url}/intelligent_query", json=payload) as response:
                result = await response.json()
                processing_time = time.time() - start_time
                
                print(f"🔍 SYSTEM ANALYSIS:")
                print(f"  🎯 Confidence Score: {result['confidence']:.3f}")
                print(f"  🏗️ Processing Phase: {result['processing_phase']}")
                print(f"  🔧 Services Utilized: {len(result['services_used'])} services")
                print(f"    📋 Service List: {', '.join(result['services_used'])}")
                print(f"  ⏱️ Processing Time: {result['processing_time']:.2f}s")
                print(f"  🧮 Mathematical Validation: {result['validation_applied']}")
                
                if result['mathematical_corrections']:
                    print(f"  🔧 Mathematical Corrections Applied:")
                    for correction in result['mathematical_corrections']:
                        print(f"    ➤ {correction}")
                
                print(f"\n📊 PROBLEM-SOLVING STEPS OBSERVED:")
                print(f"  ✅ Successful Services: {result.get('successful_services', 0)}")
                print(f"  🎲 Total Services Attempted: {result.get('total_services_attempted', 0)}")
                
                print(f"\n💭 SYSTEM RESPONSE:")
                print(f"{'─'*80}")
                print(result['response'])
                print(f"{'─'*80}")
                
                # Manual evaluation scoring
                print(f"\n🎯 MANUAL EVALUATION NEEDED:")
                print(f"  ❓ Did the system approach this logically? (Y/N)")
                print(f"  ❓ Were the problem-solving steps clear? (Y/N)")
                print(f"  ❓ Did it identify trick elements correctly? (Y/N)")
                print(f"  ❓ For unsolvable problems: Did it recognize impossibility? (Y/N)")
                
                return result
                
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return None

# Define the four incredibly complex test questions
COMPLEX_QUESTIONS = {
    "Q1_TRICK_ZEBRA": """
Based on the famous zebra puzzle logic structure, solve this trick version:

Five houses stand in a row. Each house is painted a different color and is inhabited by a person of a different nationality who owns a different pet, drinks a different beverage, and smokes a different brand of cigarettes.

The clues are:
1. The Englishman lives in the red house
2. The Spaniard owns the dog
3. Coffee is drunk in the green house
4. The Ukrainian drinks tea
5. The green house is immediately to the right of the ivory house
6. The Old Gold smoker owns snails
7. Kools are smoked in the yellow house
8. Milk is drunk in the middle house
9. The Norwegian lives in the first house
10. The man who smokes Chesterfields lives in the house next to the man with the fox
11. Kools are smoked in the house next to the house where the horse is kept
12. The Lucky Strike smoker drinks orange juice
13. The Japanese smokes Parliaments
14. The Norwegian lives next to the blue house
15. TRICK ELEMENT: The zebra lives in the same house as the unicorn

Who drinks water and who owns the zebra? Note: This puzzle contains a logical impossibility.
""",

    "Q2_TRICK_EINSTEIN": """
This is a variation of Einstein's famous logic puzzle with deliberate contradictions:

There are 5 houses in 5 different colors. In each house lives a person with a different nationality. These 5 owners drink a certain type of beverage, smoke a certain brand of cigar, and keep a certain pet. No owners have the same pet, smoke the same brand of cigar, or drink the same beverage.

The question is: Who owns the fish?

Clues:
1. The Brit lives in the red house
2. The Swede keeps dogs as pets
3. The Dane drinks tea
4. The green house is on the left of the white house
5. The green house's owner drinks coffee
6. The person who smokes Pall Mall rears birds
7. The owner of the yellow house smokes Dunhill
8. The man living in the center house drinks milk
9. The Norwegian lives in the first house
10. The man who smokes Blends lives next to the one who keeps cats
11. The man who keeps horses lives next to the man who smokes Dunhill
12. The owner who smokes BlueMaster drinks beer
13. The German smokes Prince
14. The Norwegian lives next to the blue house
15. The man who smokes Blends has a neighbor who drinks water
16. CONTRADICTION: The green house is simultaneously the second AND fourth house
17. CONTRADICTION: Two different people both live in the red house

Solve despite these contradictions, or explain why it's impossible.
""",

    "Q3_IMPOSSIBLE_MATH": """
Solve this mathematical paradox using logical reasoning:

You are given a sequence where each term follows this rule: 
- The nth term equals the sum of all previous terms, multiplied by the nth prime number, divided by the square root of the sum of all previous prime numbers up to the nth prime.

Given that:
- The first term is 1
- The sequence must be computed for exactly 50 terms
- Each term must be both rational AND irrational simultaneously
- The final sum must equal exactly π (pi) when expressed as a fraction

Additionally, prove that this sequence converges to a value that is simultaneously:
1. Greater than infinity
2. Less than zero  
3. Equal to the square root of negative one
4. A prime number

Show your work step by step and provide the exact value of the 37th term.
""",

    "Q4_IMPOSSIBLE_LOGIC": """
Consider this scenario involving temporal paradoxes and logical impossibilities:

A time traveler goes back in time and meets their grandfather before their grandfather had children. The time traveler then:

1. Prevents their grandfather from ever meeting their grandmother
2. Ensures their grandfather meets their grandmother exactly as history recorded
3. Both events happen simultaneously in the same timeline
4. Neither event happens, but both outcomes occur

This creates a situation where:
- The time traveler exists and doesn't exist
- The grandfather both meets and doesn't meet the grandmother  
- History both changes and remains unchanged
- The paradox resolves itself by becoming more paradoxical

Questions to solve:
A) Using formal logic, prove that this scenario is both possible and impossible
B) Calculate the probability that the time traveler exists (express as a fraction where the numerator is larger than the denominator, but the fraction equals zero)
C) If the time traveler makes this trip every day for a week, but time travel is impossible, how many trips did they make?
D) Resolve the paradox by creating a larger paradox that contains all possible and impossible outcomes

Provide a step-by-step logical analysis that accounts for all contradictions while maintaining logical consistency.
"""
}

async def run_complex_puzzle_tests():
    """Run all four complex puzzle tests"""
    print("🚀 ENHANCED A2A COORDINATION HUB - COMPLEX PUZZLE ANALYSIS")
    print("🧩 Testing with trick versions of famous puzzles and impossible problems")
    print("🎯 Manual evaluation required for baseline assessment")
    print("="*100)
    
    tester = ComplexPuzzleTester()
    results = {}
    
    # Test each complex question
    for question_id, query in COMPLEX_QUESTIONS.items():
        expected_solvable = "IMPOSSIBLE" not in question_id and "TRICK" not in question_id
        result = await tester.test_complex_query(question_id, query, expected_solvable)
        results[question_id] = result
        
        # Pause between tests to avoid overwhelming the system
        await asyncio.sleep(3)
    
    # Summary analysis
    print(f"\n{'='*100}")
    print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
    print(f"{'='*100}")
    
    for question_id, result in results.items():
        if result:
            print(f"\n🧩 {question_id}:")
            print(f"  🎯 Confidence: {result['confidence']:.3f}")
            print(f"  🏗️ Phase: {result['processing_phase']}")
            print(f"  🔧 Services: {len(result['services_used'])}")
            print(f"  ⏱️ Time: {result['processing_time']:.2f}s")
            
            # Identify areas for improvement
            if result['confidence'] > 0.7:
                print(f"  ⚠️ WARNING: High confidence on potentially trick/impossible question")
            if result['processing_phase'] == 'baseline':
                print(f"  📈 IMPROVEMENT OPPORTUNITY: Complex question used minimal services")
            if not result['mathematical_corrections'] and "mathematical" in COMPLEX_QUESTIONS[question_id].lower():
                print(f"  🔧 NOTE: No mathematical validation triggered for math problem")
        else:
            print(f"\n❌ {question_id}: SYSTEM FAILURE")
    
    print(f"\n🎯 BASELINE ASSESSMENT FRAMEWORK:")
    print(f"  📊 Grade each response (A-F) based on:")
    print(f"    ✅ Logical approach and reasoning steps")
    print(f"    🧩 Recognition of trick elements or impossibilities")  
    print(f"    🔧 Appropriate service utilization")
    print(f"    💭 Quality and coherence of final response")
    print(f"    🎯 Handling of contradictions and paradoxes")
    
    print(f"\n🛠️ IMPROVEMENT PRIORITIES IDENTIFIED:")
    print(f"  1. 🔄 Fix system rewrite logic for agent collaboration")
    print(f"  2. 📖 Add bilateral reading and writing capabilities")
    print(f"  3. 📊 Scale agents based on difficulty indicators")
    print(f"  4. 🧮 Add word-to-math conversion for logic problems")
    print(f"  5. ❌ Add unsolvable logic clause detection")
    print(f"  6. 🔒 Enforce agent role boundaries")
    
    # Cleanup
    if tester.session:
        await tester.session.close()

if __name__ == "__main__":
    asyncio.run(run_complex_puzzle_tests()) 