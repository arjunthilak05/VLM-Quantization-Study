#!/usr/bin/env python3
"""
Script to run Post-Training Quantization (PTQ) experiments
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ptq import GPTQQuantizer, AWQQuantizer
from models import load_model
from evaluation import VLMEvaluator
from utils import setup_logging, load_config

import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run PTQ experiments")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override model path from config"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Setup logging
    log_config = config.get("logging", {})
    setup_logging(
        log_dir=log_config.get("log_dir", "experiments/logs"),
        log_level=getattr(logging, log_config.get("level", "INFO")),
        log_to_file=log_config.get("log_to_file", True),
        log_to_console=log_config.get("log_to_console", True),
    )

    logger.info("=" * 50)
    logger.info("PTQ Experiment")
    logger.info("=" * 50)

    # Get model config
    model_config = config["model"]
    model_path = args.model_path or model_config.get("path") or model_config["name"]

    # Get quantization config
    quant_config = config["quantization"]
    method = quant_config["method"]
    bits = quant_config["bits"]

    logger.info(f"Model: {model_path}")
    logger.info(f"Quantization method: {method}")
    logger.info(f"Bits: {bits}")

    # Initialize quantizer
    if method.lower() == "gptq":
        logger.info("Initializing GPTQ quantizer")
        quantizer = GPTQQuantizer(
            model_name_or_path=model_path,
            bits=bits,
            group_size=quant_config.get("group_size", 128),
            desc_act=quant_config.get("desc_act", False),
            device_map=model_config.get("device_map", "auto"),
        )
    elif method.lower() == "awq":
        logger.info("Initializing AWQ quantizer")
        quantizer = AWQQuantizer(
            model_name_or_path=model_path,
            bits=bits,
            group_size=quant_config.get("group_size", 128),
            zero_point=quant_config.get("zero_point", True),
            device_map=model_config.get("device_map", "auto"),
        )
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    # Load model
    logger.info("Loading model...")
    if method.lower() == "gptq":
        quantizer.setup_quantize_config()
    quantizer.load_model()

    # Prepare calibration data
    calib_config = config.get("calibration", {})
    num_samples = calib_config.get("num_samples", 512)

    logger.info(f"Preparing calibration data ({num_samples} samples)...")
    # Placeholder: Load actual calibration dataset
    calibration_data = [
        {"text": f"Sample calibration text {i}"} for i in range(num_samples)
    ]

    # Quantize
    logger.info("Starting quantization...")
    quantizer.quantize(
        calibration_dataset=calibration_data,
        batch_size=calib_config.get("batch_size", 1),
    )

    # Save quantized model
    output_dir = args.output_dir or config["output"]["save_dir"]
    logger.info(f"Saving quantized model to {output_dir}")
    quantizer.save_quantized(output_dir)

    # Get model size
    model_size = quantizer.get_model_size()
    logger.info(f"Quantized model size: {model_size:.2f} MB")

    # Evaluate
    eval_config = config.get("evaluation", {})
    if eval_config:
        logger.info("Running evaluation...")

        evaluator = VLMEvaluator(
            model=quantizer.model,
            tokenizer=quantizer.tokenizer,
            benchmarks=eval_config.get("benchmarks", ["VQAv2"]),
            output_dir=eval_config.get("output_dir", "experiments/results"),
        )

        results = evaluator.evaluate()
        logger.info(f"Evaluation results: {results}")

        # Benchmark efficiency
        if config.get("efficiency", {}).get("benchmark_speed", False):
            logger.info("Benchmarking efficiency...")
            efficiency = evaluator.benchmark_efficiency()
            logger.info(f"Efficiency metrics: {efficiency}")

    logger.info("=" * 50)
    logger.info("PTQ Experiment Complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
