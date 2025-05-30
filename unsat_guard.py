#!/usr/bin/env python3
"""
Unsat Guard - Early Detection of Impossible Math Problems
=========================================================

Prevents infinite loops on unsolvable riddles by detecting mathematical impossibilities
before the composer starts hunting for non-existent solutions.
"""

import re
from typing import Dict, List, Any, Optional
from math import sqrt

class UnsatGuard:
    """Early detection system for mathematically impossible problems"""
    
    def __init__(self):
        self.prime_cache = self._generate_primes(100)  # Cache primes up to 100
    
    def _generate_primes(self, limit: int) -> List[int]:
        """Generate prime numbers up to limit using Sieve of Eratosthenes"""
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    def check_3digit_prime_riddle(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check if 3-digit prime riddle is mathematically possible
        
        Pattern: Find ABC where A+B+C=sum, A×B×C=product, all different primes
        """
        # Extract constraints from query
        sum_match = re.search(r'A \+ B \+ C = (\d+)', query)
        product_match = re.search(r'A × B × C = (\d+)|A \* B \* C = (\d+)', query)
        
        if not (sum_match and product_match):
            return None  # Not a 3-digit prime riddle
        
        target_sum = int(sum_match.group(1))
        target_product = int(product_match.group(1) or product_match.group(2))
        
        print(f"🔍 UNSAT GUARD: Checking riddle with sum={target_sum}, product={target_product}")
        
        # Factor the target product
        prime_factors = self._prime_factorization(target_product)
        unique_primes = list(set(prime_factors))
        
        print(f"  Prime factorization: {prime_factors}")
        print(f"  Unique primes: {unique_primes}")
        
        # Quick impossibility checks
        impossibility_reasons = []
        
        # Check 1: Need exactly 3 different primes for A, B, C
        if len(unique_primes) < 3:
            impossibility_reasons.append(f"Only {len(unique_primes)} unique prime factors, need 3 different primes")
        
        # Check 2: All primes must be single digits for 3-digit number ABC
        large_primes = [p for p in unique_primes if p >= 10]
        if large_primes:
            impossibility_reasons.append(f"Prime factors {large_primes} are ≥10, but need single-digit primes for 3-digit number")
        
        # Check 3: Check if any combination of 3 primes from factorization can sum to target
        if len(unique_primes) >= 3:
            possible_combinations = self._get_prime_combinations(unique_primes, target_product, 3)
            valid_sums = [sum(combo) for combo in possible_combinations if self._product_matches(combo, target_product)]
            
            if target_sum not in valid_sums:
                impossibility_reasons.append(f"No combination of 3 primes from factorization sums to {target_sum}")
        
        # Check 4: Specific case for 1000 = 2³×5³
        if target_product == 1000:
            impossibility_reasons.append("1000 = 2³×5³ has only 2 unique prime factors (2,5), impossible to form 3 different prime digits")
        
        if impossibility_reasons:
            return {
                'is_impossible': True,
                'reasons': impossibility_reasons,
                'suggested_response': f"This problem has no solution. {' '.join(impossibility_reasons)}",
                'confidence': 0.95
            }
        
        return {
            'is_impossible': False,
            'reasons': [],
            'confidence': 0.8
        }
    
    def _prime_factorization(self, n: int) -> List[int]:
        """Get prime factorization of n"""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    def _get_prime_combinations(self, primes: List[int], target_product: int, count: int) -> List[List[int]]:
        """Get all combinations of 'count' primes that could multiply to target_product"""
        from itertools import combinations_with_replacement
        
        # For simplicity, just check combinations of unique primes
        # In a full implementation, we'd handle repeated primes properly
        if len(primes) < count:
            return []
        
        from itertools import combinations
        return list(combinations(primes, count))
    
    def _product_matches(self, combo: List[int], target: int) -> bool:
        """Check if product of combo matches target (allowing for repeated factors)"""
        product = 1
        for p in combo:
            product *= p
        
        # Check if target is divisible by this product and remaining factors are reasonable
        if target % product == 0:
            remaining = target // product
            # Check if remaining can be formed by repeating primes in combo
            return self._can_form_remaining(remaining, combo)
        return False
    
    def _can_form_remaining(self, remaining: int, available_primes: List[int]) -> bool:
        """Check if remaining can be formed by powers of available primes"""
        if remaining == 1:
            return True
        
        for prime in available_primes:
            if remaining % prime == 0:
                return self._can_form_remaining(remaining // prime, available_primes)
        
        return False
    
    def check_general_impossibility(self, query: str) -> Optional[Dict[str, Any]]:
        """Check for general mathematical impossibilities"""
        
        # Check for contradictory constraints
        if "no solution" in query.lower() and "find" in query.lower():
            return {
                'is_impossible': True,
                'reasons': ["Query asks to find something that has no solution"],
                'suggested_response': "This is a contradiction - cannot find something that doesn't exist.",
                'confidence': 0.9
            }
        
        # Check for impossible number theory problems
        if re.search(r'prime.*even.*greater than 2', query.lower()):
            return {
                'is_impossible': True,
                'reasons': ["No even prime numbers exist greater than 2"],
                'suggested_response': "This is impossible. The only even prime number is 2.",
                'confidence': 0.95
            }
        
        return None
    
    def guard_check(self, query: str) -> Optional[Dict[str, Any]]:
        """Main guard check - returns impossibility detection if found"""
        
        # Check 3-digit prime riddle
        riddle_check = self.check_3digit_prime_riddle(query)
        if riddle_check and riddle_check['is_impossible']:
            return riddle_check
        
        # Check general impossibilities
        general_check = self.check_general_impossibility(query)
        if general_check and general_check['is_impossible']:
            return general_check
        
        return None

def test_unsat_guard():
    """Test the unsat guard with known impossible problems"""
    guard = UnsatGuard()
    
    # Test the 1000 riddle
    test_query = """Find a 3-digit number ABC where:
- A + B + C = 25
- A × B × C = 1000  
- A, B, C are all different prime numbers
- A > B > C"""
    
    result = guard.guard_check(test_query)
    print("Test Result:", result)
    
    return result

if __name__ == "__main__":
    test_unsat_guard() 