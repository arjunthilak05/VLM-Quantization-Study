"""
Model loading utilities for VLMs
"""

import logging
from typing import Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

logger = logging.getLogger(__name__)


# Model registry with HuggingFace paths
MODEL_REGISTRY = {
    "llava-phi-3b": "xtuner/llava-phi-3-mini-hf",
    "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
    "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",
    "tinygptv": "TinyGPT-V/TinyGPT-V",  # Placeholder
    "qwen-vl": "Qwen/Qwen-VL",
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
}


def load_model(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
    **kwargs
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a VLM model and tokenizer

    Args:
        model_name: Model name from MODEL_REGISTRY or HuggingFace path
        device_map: Device mapping strategy
        torch_dtype: Torch dtype for model weights
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments for model loading

    Returns:
        Tuple of (model, tokenizer)
    """
    # Get model path from registry
    if model_name in MODEL_REGISTRY:
        model_path = MODEL_REGISTRY[model_name]
        logger.info(f"Loading model '{model_name}' from {model_path}")
    else:
        model_path = model_name
        logger.info(f"Loading model from {model_path}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            **kwargs
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs
        )

        logger.info(f"Model loaded successfully: {model_path}")
        logger.info(f"Model size: {get_model_size(model):.2f} MB")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_quantized_model(
    model_path: str,
    quantization_method: str = "gptq",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    **kwargs
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a quantized VLM model

    Args:
        model_path: Path to quantized model
        quantization_method: Quantization method used ('gptq', 'awq', 'qat')
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading quantized model from {model_path}")
    logger.info(f"Quantization method: {quantization_method}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )

        # Load quantized model based on method
        if quantization_method.lower() == "gptq":
            from auto_gptq import AutoGPTQForCausalLM

            model = AutoGPTQForCausalLM.from_quantized(
                model_path,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                **kwargs
            )

        elif quantization_method.lower() == "awq":
            # AWQ loading
            logger.warning("AWQ loading not fully implemented. Using standard loading.")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                **kwargs
            )

        elif quantization_method.lower() == "qat":
            # QAT model loading
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                **kwargs
            )

        else:
            raise ValueError(f"Unknown quantization method: {quantization_method}")

        logger.info(f"Quantized model loaded successfully")
        logger.info(f"Model size: {get_model_size(model):.2f} MB")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading quantized model: {e}")
        raise


def get_model_size(model) -> float:
    """
    Calculate model size in MB

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)

    return size_mb


def get_model_info(model) -> dict:
    """
    Get information about a model

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": get_model_size(model),
        "dtype": str(next(model.parameters()).dtype),
    }


def print_model_info(model):
    """Print model information"""
    info = get_model_info(model)

    print("=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"Model size: {info['size_mb']:.2f} MB")
    print(f"Data type: {info['dtype']}")
    print("=" * 50)
