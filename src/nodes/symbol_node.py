from typing import Dict, Any, List
from .base_node import BaseNode

class SymbolNode(BaseNode):
    """Node for processing and managing symbolic representations"""
    
    def __init__(self, node_id: str = None):
        super().__init__(node_id)
        self.symbols = {
            "fire": {"symbol": "🜂", "activation": 0.0, "meaning": "transformation"},
            "water": {"symbol": "🜄", "activation": 0.0, "meaning": "flow"},
            "air": {"symbol": "🜁", "activation": 0.0, "meaning": "thought"},
            "earth": {"symbol": "🜃", "activation": 0.0, "meaning": "foundation"},
            "infinity": {"symbol": "∞", "activation": 0.0, "meaning": "limitless"},
            "void": {"symbol": "⦾", "activation": 0.0, "meaning": "potential"},
            "unity": {"symbol": "☯", "activation": 0.0, "meaning": "harmony"},
            "consciousness": {"symbol": "◉", "activation": 0.0, "meaning": "awareness"}
        }
        
        self.active_symbols = []
        self.state.update({
            "active_symbols": self.active_symbols,
            "total_symbols": len(self.symbols)
        })
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process symbolic data"""
        try:
            # Extract symbol information from input
            input_symbol = data.get("symbol", "")
            input_text = data.get("text", "")
            
            # Process symbol activation
            if input_symbol in self.symbols:
                self.symbols[input_symbol]["activation"] = 1.0
                if input_symbol not in self.active_symbols:
                    self.active_symbols.append(input_symbol)
            
            # Analyze text for symbolic content
            detected_symbols = self._detect_symbols(input_text)
            
            # Update state
            self.state["active_symbols"] = self.active_symbols
            self.last_update = datetime.now()
            
            return {
                "status": "success",
                "processed_symbol": input_symbol,
                "detected_symbols": detected_symbols,
                "active_symbols": self.active_symbols,
                "symbol_info": self.symbols.get(input_symbol, {})
            }
            
        except Exception as e:
            self.logger.error(f"Error processing symbol data: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def _detect_symbols(self, text: str) -> List[str]:
        """Detect symbols in text"""
        detected = []
        for symbol_name, info in self.symbols.items():
            if symbol_name.lower() in text.lower() or info["symbol"] in text:
                detected.append(symbol_name)
                if symbol_name not in self.active_symbols:
                    self.active_symbols.append(symbol_name)
                    self.symbols[symbol_name]["activation"] = 0.5
        return detected
        
    def get_symbol_info(self, symbol_name: str) -> Dict[str, Any]:
        """Get information about a specific symbol"""
        return self.symbols.get(symbol_name, {}) 