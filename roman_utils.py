import re

class RomanNumeral:
    # Roman numeral values and their combinations
    ROMAN_VALUES = {
        'I': 1, 'V': 5, 'X': 10,
        'L': 50, 'C': 100, 'D': 500, 'M': 1000
    }
    
    # Valid Roman numeral patterns
    VALID_PATTERNS = [
        r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$',
        r'^[IVXLCDM]+$'
    ]
    
    @staticmethod
    def is_valid(roman):
        """Check if a string is a valid Roman numeral."""
        if not isinstance(roman, str):
            return False
        roman = roman.strip().upper()
        return any(re.match(pattern, roman) for pattern in RomanNumeral.VALID_PATTERNS)
    
    @staticmethod
    def to_int(roman):
        """Convert Roman numeral to integer."""
        if not RomanNumeral.is_valid(roman):
            raise ValueError(f"Invalid Roman numeral: {roman}")
            
        roman = roman.strip().upper()
        total = 0
        prev_value = 0
        
        for char in reversed(roman):
            curr_value = RomanNumeral.ROMAN_VALUES[char]
            if curr_value >= prev_value:
                total += curr_value
            else:
                total -= curr_value
            prev_value = curr_value
            
        return total
    
    @staticmethod
    def from_int(num):
        """Convert integer to Roman numeral."""
        if not isinstance(num, int) or num <= 0 or num > 3999:
            raise ValueError(f"Number must be between 1 and 3999: {num}")
            
        roman_symbols = [
            ('M', 1000), ('CM', 900), ('D', 500), ('CD', 400),
            ('C', 100), ('XC', 90), ('L', 50), ('XL', 40),
            ('X', 10), ('IX', 9), ('V', 5), ('IV', 4), ('I', 1)
        ]
        
        result = ''
        for symbol, value in roman_symbols:
            while num >= value:
                result += symbol
                num -= value
                
        return result
    
    @staticmethod
    def add(roman1, roman2):
        """Add two Roman numerals."""
        num1 = RomanNumeral.to_int(roman1)
        num2 = RomanNumeral.to_int(roman2)
        return RomanNumeral.from_int(num1 + num2)
    
    @staticmethod
    def subtract(roman1, roman2):
        """Subtract two Roman numerals."""
        num1 = RomanNumeral.to_int(roman1)
        num2 = RomanNumeral.to_int(roman2)
        if num1 < num2:
            raise ValueError(f"Cannot subtract {roman2} from {roman1}")
        return RomanNumeral.from_int(num1 - num2)
    
    @staticmethod
    def multiply(roman1, roman2):
        """Multiply two Roman numerals."""
        num1 = RomanNumeral.to_int(roman1)
        num2 = RomanNumeral.to_int(roman2)
        return RomanNumeral.from_int(num1 * num2)
    
    @staticmethod
    def divide(roman1, roman2):
        """Divide two Roman numerals."""
        num1 = RomanNumeral.to_int(roman1)
        num2 = RomanNumeral.to_int(roman2)
        if num2 == 0:
            raise ValueError("Cannot divide by zero")
        return RomanNumeral.from_int(num1 // num2)
    
    @staticmethod
    def parse_expression(expr):
        """Parse a Roman numeral arithmetic expression."""
        expr = expr.strip()
        if '+' in expr:
            parts = expr.split('+')
            if len(parts) != 2:
                raise ValueError("Invalid addition expression")
            return RomanNumeral.add(parts[0].strip(), parts[1].strip())
        elif '-' in expr:
            parts = expr.split('-')
            if len(parts) != 2:
                raise ValueError("Invalid subtraction expression")
            return RomanNumeral.subtract(parts[0].strip(), parts[1].strip())
        elif '*' in expr:
            parts = expr.split('*')
            if len(parts) != 2:
                raise ValueError("Invalid multiplication expression")
            return RomanNumeral.multiply(parts[0].strip(), parts[1].strip())
        elif '/' in expr:
            parts = expr.split('/')
            if len(parts) != 2:
                raise ValueError("Invalid division expression")
            return RomanNumeral.divide(parts[0].strip(), parts[1].strip())
        else:
            raise ValueError("Invalid arithmetic expression") 