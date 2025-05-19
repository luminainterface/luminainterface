import os
import logging
import asyncio
import torch
from typing import Dict, Any, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from peft.utils import get_peft_model_state_dict
import json
from datetime import datetime

logger = logging.getLogger("output_engine.lora")

class Phi2LoRAAdapter:
    def __init__(
        self,
        base_model_path: str = "microsoft/phi-2",
        adapter_path: str = "/app/models/phi-2-lora",
        device: str = "cpu",
        r: int = 16,  # LoRA rank
        alpha: int = 32,  # LoRA alpha
        dropout: float = 0.1,
        target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "dense"],
        save_interval: int = 10,
        max_length: int = 512
    ):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.device = device
        self.save_interval = save_interval
        self.max_length = max_length
        self.update_count = 0
        self.lock = asyncio.Lock()
        
        # Initialize model and tokenizer
        logger.info(f"Loading base model from {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # Configure LoRA
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Load or initialize model
        self._initialize_model()
        
        # Create adapter directory if it doesn't exist
        os.makedirs(adapter_path, exist_ok=True)
        
        # Load adapter weights if they exist
        self._load_adapter_if_exists()
        
        logger.info(f"LoRA adapter initialized on {device}")
    
    def _initialize_model(self):
        """Initialize the base model and prepare it for LoRA training."""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
            
            # Prepare model for LoRA training
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, self.lora_config)
            
            # Handle meta tensors correctly
            if any(p.device.type == "meta" for p in self.model.parameters()):
                self.model = self.model.to_empty(device=self.device)
            else:
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Start in eval mode
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def _load_adapter_if_exists(self):
        """Load adapter weights if they exist."""
        adapter_weights_path = os.path.join(self.adapter_path, "adapter_model.bin")
        if os.path.exists(adapter_weights_path):
            try:
                self.model.load_state_dict(
                    torch.load(adapter_weights_path, map_location=self.device),
                    strict=False
                )
                logger.info("Loaded existing adapter weights")
            except Exception as e:
                logger.error(f"Error loading adapter weights: {e}")
    
    async def finetune(
        self,
        embedding: List[float],
        concept: Dict[str, Any],
        learning_rate: float = 1e-4,
        num_epochs: int = 3
    ) -> Dict[str, Any]:
        """
        Fine-tune the LoRA adapter on a new concept.
        
        Args:
            embedding: The concept embedding vector
            concept: Dictionary containing concept data (text, metadata, etc.)
            learning_rate: Learning rate for fine-tuning
            num_epochs: Number of training epochs
        
        Returns:
            Dict containing training metrics and status
        """
        async with self.lock:
            try:
                # Switch to training mode
                self.model.train()
                
                # Prepare optimizer
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=learning_rate
                )
                
                # Get concept text and prepare input
                text = concept.get("text", "")
                if not text:
                    raise ValueError("Concept text is required for fine-tuning")
                
                # Tokenize input
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # Training loop
                total_loss = 0
                for epoch in range(num_epochs):
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / num_epochs
                
                # Switch back to eval mode
                self.model.eval()
                
                # Increment update counter
                self.update_count += 1
                
                # Save adapter if needed
                if self.update_count % self.save_interval == 0:
                    await self.save_adapter()
                
                # Log training results
                training_metrics = {
                    "concept_id": concept.get("id", "unknown"),
                    "avg_loss": avg_loss,
                    "epochs": num_epochs,
                    "update_count": self.update_count,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"Fine-tuning completed: {json.dumps(training_metrics)}")
                return training_metrics
                
            except Exception as e:
                logger.error(f"Error during fine-tuning: {e}")
                raise
            finally:
                self.model.eval()  # Ensure we're in eval mode
    
    async def save_adapter(self) -> bool:
        """Save the current adapter state."""
        try:
            # Save adapter weights
            weights_path = os.path.join(self.adapter_path, "adapter_model.bin")
            torch.save(
                get_peft_model_state_dict(self.model),
                weights_path
            )
            
            # Save config
            config_path = os.path.join(self.adapter_path, "adapter_config.json")
            self.lora_config.save_pretrained(config_path)
            
            logger.info(f"Adapter saved to {self.adapter_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving adapter: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.85,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3
    ) -> Dict[str, Any]:
        """
        Generate text using the fine-tuned model.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower = more focused)
            top_p: Top-p sampling parameter (lower = more focused)
            repetition_penalty: Penalty for repeating tokens
            no_repeat_ngram_size: Size of n-grams to prevent repeating
        
        Returns:
            Dict containing generated text and metadata
        """
        try:
            # Ensure model is in eval mode
            self.model.eval()
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            ).to(self.device)
            
            # Generate with more conservative parameters
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        early_stopping=True
                    )
                except RuntimeError as e:
                    if "probability tensor" in str(e):
                        # Fallback to more conservative parameters
                        logger.warning("Falling back to more conservative generation parameters")
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=0.5,  # Even more conservative
                            top_p=0.8,
                            do_sample=True,
                            repetition_penalty=1.5,
                            no_repeat_ngram_size=3,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            num_return_sequences=1,
                            early_stopping=True
                        )
                    else:
                        raise
            
            # Decode and return
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            return {
                "text": generated_text,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """
        Get information about the current state of the LoRA adapter.
        
        Returns:
            Dict containing adapter information including:
            - update_count: Number of times the adapter has been updated
            - config: Current LoRA configuration
            - device: Current device
            - has_weights: Whether adapter weights exist
        """
        adapter_weights_path = os.path.join(self.adapter_path, "adapter_model.bin")
        return {
            "update_count": self.update_count,
            "config": {
                "r": self.lora_config.r,
                "alpha": self.lora_config.lora_alpha,
                "dropout": self.lora_config.lora_dropout,
                "target_modules": self.lora_config.target_modules
            },
            "device": self.device,
            "has_weights": os.path.exists(adapter_weights_path)
        } 