"""
QAT Trainer for Vision-Language Models
Implements block-wise and end-to-end quantization-aware training
"""

import torch
import torch.nn as nn
import torch.quantization as tq
import logging
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from tqdm import tqdm

from .config import QATConfig

logger = logging.getLogger(__name__)


class QATTrainer:
    """
    Quantization-Aware Training trainer for VLMs

    Implements fake quantization during training to simulate quantization effects
    """

    def __init__(
        self,
        model_name_or_path: str,
        config: QATConfig,
        device_map: str = "auto",
    ):
        """
        Initialize QAT trainer

        Args:
            model_name_or_path: HuggingFace model path or local path
            config: QAT configuration
            device_map: Device mapping strategy
        """
        self.model_name_or_path = model_name_or_path
        self.config = config
        self.device_map = device_map

        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None

        logger.info(f"Initialized QAT trainer for {model_name_or_path}")
        logger.info(f"Config: {config.to_dict()}")

    def load_model(self):
        """Load the base model"""
        logger.info(f"Loading model from {self.model_name_or_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        logger.info("Model loaded successfully")
        return self.model

    def prepare_for_qat(self):
        """
        Prepare model for quantization-aware training
        Inserts fake quantization modules
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info("Preparing model for QAT")

        # Set quantization config
        if self.config.use_fake_quantization:
            # Use fake quantization during training
            self.model.qconfig = tq.get_default_qat_qconfig('fbgemm')

            # Prepare model for QAT
            self.model = tq.prepare_qat(self.model, inplace=True)

            logger.info("Model prepared with fake quantization")

        # Freeze non-quantized layers if requested
        if self.config.freeze_non_quantized:
            self._freeze_non_quantized_layers()

        return self.model

    def _freeze_non_quantized_layers(self):
        """Freeze layers that should not be quantized"""
        modules_to_not_quantize = self.config.modules_to_not_quantize or []

        for name, param in self.model.named_parameters():
            # Check if this parameter is in a module we should not quantize
            should_freeze = any(
                module_name in name for module_name in modules_to_not_quantize
            )

            if should_freeze:
                param.requires_grad = False
                logger.debug(f"Frozen parameter: {name}")

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Filter parameters that require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Linear warmup scheduler
        total_steps = self.config.num_epochs * 1000  # Placeholder
        warmup_steps = self.config.warmup_steps

        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )

        logger.info("Optimizer and scheduler setup complete")

    def train(
        self,
        train_dataset,
        eval_dataset=None,
    ):
        """
        Train the model with quantization-aware training

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        if self.model is None:
            raise ValueError("Model not prepared. Call prepare_for_qat() first.")

        logger.info("Starting QAT training")

        # Setup optimizer if not already done
        if self.optimizer is None:
            self.setup_optimizer()

        # Training loop
        self.model.train()
        global_step = 0

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            epoch_loss = 0
            num_batches = 0

            # Simple training loop (in practice, use DataLoader)
            for batch_idx, batch in enumerate(tqdm(train_dataset)):
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    global_step += 1

                epoch_loss += loss.item()
                num_batches += 1

                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(f"Step {global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")

                # Evaluation
                if eval_dataset and global_step % self.config.eval_steps == 0:
                    self._evaluate(eval_dataset)

                # Saving
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(global_step)

            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        logger.info("QAT training completed")

    def _evaluate(self, eval_dataset):
        """Evaluate the model"""
        self.model.eval()

        eval_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataset:
                outputs = self.model(**batch)
                eval_loss += outputs.loss.item()
                num_batches += 1

        avg_eval_loss = eval_loss / num_batches
        logger.info(f"Evaluation loss: {avg_eval_loss:.4f}")

        self.model.train()

    def _save_checkpoint(self, step: int):
        """Save model checkpoint"""
        output_path = f"{self.config.output_dir}/checkpoint-{step}"
        logger.info(f"Saving checkpoint to {output_path}")

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

    def convert_to_quantized(self):
        """
        Convert the QAT model to actual quantized model
        This converts fake quantization to real quantization
        """
        if self.model is None:
            raise ValueError("No model to convert")

        logger.info("Converting QAT model to quantized model")

        self.model.eval()
        self.model = tq.convert(self.model, inplace=True)

        logger.info("Model converted to quantized format")
        return self.model

    def save_quantized(self, output_dir: str):
        """Save the final quantized model"""
        logger.info(f"Saving quantized model to {output_dir}")

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save config
        import json
        config_path = f"{output_dir}/qat_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info("Quantized model saved successfully")

    def load_quantized(self, model_dir: str):
        """Load a QAT-trained quantized model"""
        logger.info(f"Loading quantized model from {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=self.device_map,
            trust_remote_code=True
        )

        logger.info("Quantized model loaded successfully")
        return self.model

    def get_model_size(self):
        """Calculate model size in MB"""
        if self.model is None:
            return 0

        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)

        return size_mb
