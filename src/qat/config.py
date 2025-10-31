"""
Configuration for Quantization-Aware Training
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class QATConfig:
    """
    Configuration for Quantization-Aware Training

    Args:
        bits: Number of bits for quantization (4, 8)
        weight_bits: Bits for weight quantization
        activation_bits: Bits for activation quantization
        quantization_scheme: Quantization scheme (symmetric, asymmetric)
        per_channel: Whether to use per-channel quantization
        learning_rate: Learning rate for QAT
        num_epochs: Number of training epochs
        batch_size: Training batch size
        gradient_accumulation_steps: Steps for gradient accumulation
        warmup_steps: Number of warmup steps
        block_wise: Whether to use block-wise QAT
        freeze_non_quantized: Whether to freeze non-quantized layers
    """

    # Quantization parameters
    bits: int = 4
    weight_bits: Optional[int] = None
    activation_bits: Optional[int] = None
    quantization_scheme: str = "symmetric"  # or "asymmetric"
    per_channel: bool = True

    # Training parameters
    learning_rate: float = 1e-5
    num_epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # QAT-specific parameters
    block_wise: bool = True
    freeze_non_quantized: bool = False
    use_fake_quantization: bool = True

    # Data parameters
    max_length: int = 2048
    num_workers: int = 4

    # Logging and saving
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = "experiments/checkpoints/qat"

    # Hardware
    device: str = "cuda"
    mixed_precision: str = "fp16"  # or "bf16", "no"

    # Advanced options
    modules_to_quantize: Optional[List[str]] = None
    modules_to_not_quantize: Optional[List[str]] = field(default_factory=lambda: ["lm_head"])

    def __post_init__(self):
        """Set default values after initialization"""
        if self.weight_bits is None:
            self.weight_bits = self.bits
        if self.activation_bits is None:
            self.activation_bits = 16  # Typically keep activations at higher precision

    def to_dict(self):
        """Convert config to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)
