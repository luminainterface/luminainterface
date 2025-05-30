#!/usr/bin/env python3
"""
Phi-2 Ultrafast Engine Service with NPU Support
High-performance inference with phi-2 model optimized for NPU/CPU/GPU
"""

import asyncio
import time
import torch
import gc
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7
    stream: bool = False

class Phi2Engine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = self._detect_best_device()
        self.model_name = "microsoft/phi-2"
        self.ready = False
        
    def _detect_best_device(self) -> str:
        """Detect the best available device (NPU > CUDA > CPU)"""
        # Check for NPU support (Intel/AMD/Qualcomm NPU)
        try:
            # Try Intel NPU
            if hasattr(torch.backends, 'intel_npu') and torch.backends.intel_npu.is_available():
                logger.info("🚀 Intel NPU detected!")
                return "npu"
        except:
            pass
        
        try:
            # Try DirectML for Windows NPU/GPU acceleration
            import torch_directml
            if torch_directml.is_available():
                logger.info("🚀 DirectML NPU/GPU detected!")
                return torch_directml.device()
        except ImportError:
            pass
        
        # Check CUDA
        if torch.cuda.is_available():
            logger.info(f"🎮 CUDA GPU detected: {torch.cuda.get_device_name()}")
            return "cuda"
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🍎 Apple MPS detected!")
            return "mps"
        
        # Fallback to CPU with optimizations
        logger.info("💻 Using optimized CPU")
        return "cpu"
        
    async def initialize(self):
        """Initialize phi-2 model with optimizations"""
        try:
            logger.info(f"Loading phi-2 model on {self.device}...")
            start_time = time.time()
            
            # Memory optimization before loading
            if self.device == "cpu":
                self._optimize_cpu_memory()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                cache_dir="./model_cache"
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with device-specific optimizations
            model_kwargs = {
                "trust_remote_code": True,
                "cache_dir": "./model_cache",
                "low_cpu_mem_usage": True
            }
            
            if self.device == "cpu":
                # CPU optimizations
                model_kwargs.update({
                    "torch_dtype": torch.float32,
                    "device_map": None
                })
            elif self.device == "cuda":
                # CUDA optimizations
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": "auto"
                })
            elif "directml" in str(self.device):
                # DirectML optimizations
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": None
                })
            else:
                # NPU/MPS optimizations
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": None
                })
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move model to device
            if self.device not in ["auto"] and not ("directml" in str(self.device)):
                self.model = self.model.to(self.device)
            
            # CPU-specific optimizations
            if self.device == "cpu":
                self._apply_cpu_optimizations()
            
            self.ready = True
            load_time = time.time() - start_time
            logger.info(f"✅ Phi-2 model loaded in {load_time:.2f}s on {self.device}")
            
            # Memory usage info
            self._log_memory_usage()
            
        except Exception as e:
            logger.error(f"❌ Failed to load phi-2: {e}")
            raise
    
    def _optimize_cpu_memory(self):
        """Optimize CPU memory before model loading"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Set CPU thread count for optimal performance
        torch.set_num_threads(min(8, psutil.cpu_count()))
        logger.info(f"Set PyTorch threads to {torch.get_num_threads()}")
    
    def _apply_cpu_optimizations(self):
        """Apply CPU-specific optimizations"""
        try:
            # Enable CPU optimizations
            self.model.eval()
            
            # Try to compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("✅ Model compiled for faster CPU inference")
                except:
                    logger.info("ℹ️ Model compilation not available or failed")
        except Exception as e:
            logger.warning(f"⚠️ CPU optimization failed: {e}")
    
    def _log_memory_usage(self):
        """Log current memory usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            logger.info(f"📊 Memory usage: {memory_mb:.1f} MB")
            
            if torch.cuda.is_available() and self.device == "cuda":
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                logger.info(f"🎮 GPU memory: {gpu_memory:.1f} MB")
        except:
            pass
    
    async def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> dict:
        """Generate text using phi-2 with optimizations"""
        if not self.ready:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        try:
            start_time = time.time()
            
            # Tokenize input with padding
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=1024
            )
            
            # Move to device
            if self.device not in ["auto"] and not ("directml" in str(self.device)):
                inputs = inputs.to(self.device)
            
            # Generate with optimized parameters
            generation_config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            }
            
            # Device-specific generation optimizations
            if self.device == "cpu":
                generation_config.update({
                    "num_beams": 1,  # Faster on CPU
                    "early_stopping": True
                })
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generation_config
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = generated_text[len(prompt):].strip()
            
            generation_time = time.time() - start_time
            
            # Memory cleanup
            del inputs, outputs
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return {
                "response": response_text,
                "prompt": prompt,
                "generation_time_ms": round(generation_time * 1000, 2),
                "tokens_generated": len(self.tokenizer.encode(response_text)),
                "model": self.model_name,
                "device": str(self.device),
                "performance_stats": {
                    "tokens_per_second": round(len(self.tokenizer.encode(response_text)) / generation_time, 2) if generation_time > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

# Initialize FastAPI and phi-2 engine
app = FastAPI(title="Phi-2 Ultrafast Engine with NPU Support", version="2.0.0")
phi2_engine = Phi2Engine()

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    await phi2_engine.initialize()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if phi2_engine.ready else "loading",
        "model": phi2_engine.model_name,
        "device": str(phi2_engine.device),
        "ready": phi2_engine.ready,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.post("/generate")
async def generate_text(request: QueryRequest):
    """Generate text using phi-2"""
    result = await phi2_engine.generate(
        request.prompt,
        request.max_tokens,
        request.temperature
    )
    return result

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_name": phi2_engine.model_name,
        "device": str(phi2_engine.device),
        "ready": phi2_engine.ready,
        "parameters": "2.7B",
        "architecture": "Transformer",
        "capabilities": ["text-generation", "reasoning", "coding"],
        "optimizations": ["npu-support", "memory-optimized", "cpu-compiled"],
        "version": "2.0.0"
    }

@app.get("/performance")
async def get_performance():
    """Get performance metrics"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        stats = {
            "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 1),
            "cpu_percent": process.cpu_percent(),
            "device": str(phi2_engine.device),
            "model_ready": phi2_engine.ready
        }
        
        if torch.cuda.is_available() and phi2_engine.device == "cuda":
            stats["gpu_memory_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
            stats["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 1)
        
        return stats
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8892)
