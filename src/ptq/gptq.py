"""
GPTQ (Gradient-based Post-Training Quantization) implementation
Layer-wise quantization using Hessian matrix optimization
"""

import torch
import logging
from typing import Dict, List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

logger = logging.getLogger(__name__)


class GPTQQuantizer:
    """
    GPTQ quantizer for Vision-Language Models

    Reference: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
    """

    def __init__(
        self,
        model_name_or_path: str,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        device_map: str = "auto",
    ):
        """
        Initialize GPTQ quantizer

        Args:
            model_name_or_path: HuggingFace model path or local path
            bits: Number of bits for quantization (4, 8)
            group_size: Group size for quantization (32, 64, 128)
            desc_act: Whether to quantize activation in descending order
            device_map: Device mapping strategy
        """
        self.model_name_or_path = model_name_or_path
        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.device_map = device_map

        self.quantize_config = None
        self.model = None
        self.tokenizer = None

        logger.info(f"Initialized GPTQ quantizer for {model_name_or_path}")
        logger.info(f"Config: bits={bits}, group_size={group_size}")

    def setup_quantize_config(self, **kwargs):
        """Setup quantization configuration"""
        self.quantize_config = BaseQuantizeConfig(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=self.desc_act,
            **kwargs
        )
        logger.info(f"Quantization config: {self.quantize_config}")
        return self.quantize_config

    def load_model(self):
        """Load the base model"""
        logger.info(f"Loading model from {self.model_name_or_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )

        # Load model
        self.model = AutoGPTQForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantize_config=self.quantize_config,
            device_map=self.device_map,
            trust_remote_code=True
        )

        logger.info("Model loaded successfully")
        return self.model

    def quantize(
        self,
        calibration_dataset: List[Dict],
        batch_size: int = 1,
        use_triton: bool = False,
    ):
        """
        Quantize the model using GPTQ

        Args:
            calibration_dataset: List of calibration samples
            batch_size: Batch size for calibration
            use_triton: Whether to use Triton kernels
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info(f"Starting GPTQ quantization with {len(calibration_dataset)} samples")

        # Prepare calibration data
        calibration_data = self._prepare_calibration_data(
            calibration_dataset,
            batch_size
        )

        # Quantize
        self.model.quantize(
            calibration_data,
            batch_size=batch_size,
            use_triton=use_triton,
        )

        logger.info("GPTQ quantization completed")
        return self.model

    def _prepare_calibration_data(self, dataset: List[Dict], batch_size: int):
        """Prepare calibration data for quantization"""
        calibration_data = []

        for sample in dataset:
            # Handle different data formats
            if isinstance(sample, dict):
                if "text" in sample:
                    text = sample["text"]
                elif "prompt" in sample:
                    text = sample["prompt"]
                else:
                    text = str(sample)
            else:
                text = str(sample)

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )

            calibration_data.append(inputs)

        logger.info(f"Prepared {len(calibration_data)} calibration samples")
        return calibration_data

    def save_quantized(self, output_dir: str):
        """Save quantized model"""
        if self.model is None:
            raise ValueError("No model to save. Run quantize() first.")

        logger.info(f"Saving quantized model to {output_dir}")
        self.model.save_quantized(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully")

    def load_quantized(self, model_dir: str):
        """Load a pre-quantized model"""
        logger.info(f"Loading quantized model from {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True
        )

        self.model = AutoGPTQForCausalLM.from_quantized(
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

    def benchmark_speed(self, num_samples: int = 100, max_length: int = 512):
        """Benchmark inference speed"""
        import time

        if self.model is None:
            raise ValueError("Model not loaded")

        logger.info(f"Benchmarking speed with {num_samples} samples")

        # Generate dummy inputs
        dummy_text = "This is a test prompt for benchmarking. " * 20
        inputs = self.tokenizer(dummy_text, return_tensors="pt").to(self.model.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model.generate(**inputs, max_length=max_length)

        # Benchmark
        start_time = time.time()
        total_tokens = 0

        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.model.generate(**inputs, max_length=max_length)
                total_tokens += outputs.shape[1]

        end_time = time.time()
        elapsed = end_time - start_time
        tokens_per_sec = total_tokens / elapsed

        logger.info(f"Speed: {tokens_per_sec:.2f} tokens/sec")

        return {
            "tokens_per_sec": tokens_per_sec,
            "elapsed_time": elapsed,
            "total_tokens": total_tokens
        }
