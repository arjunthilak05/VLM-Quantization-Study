"""
Quantization-Aware Training (QAT) implementations
Supports block-wise and end-to-end QAT for VLMs
"""

from .trainer import QATTrainer
from .config import QATConfig

__all__ = ["QATTrainer", "QATConfig"]
