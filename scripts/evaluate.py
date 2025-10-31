#!/usr/bin/env python3
"""
Script to evaluate quantized VLM models
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import load_quantized_model
from evaluation import VLMEvaluator
from utils import setup_logging

import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized VLM models")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to quantized model"
    )
    parser.add_argument(
        "--quantization-method",
        type=str,
        default="gptq",
        choices=["gptq", "awq", "qat"],
        help="Quantization method used"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["VQAv2"],
        help="Benchmarks to evaluate on"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--benchmark-efficiency",
        action="store_true",
        help="Benchmark efficiency metrics"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    logger.info("=" * 50)
    logger.info("Model Evaluation")
    logger.info("=" * 50)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Quantization method: {args.quantization_method}")
    logger.info(f"Benchmarks: {args.benchmarks}")

    # Load model
    logger.info("Loading quantized model...")
    model, tokenizer = load_quantized_model(
        model_path=args.model_path,
        quantization_method=args.quantization_method,
    )

    # Create evaluator
    logger.info("Creating evaluator...")
    evaluator = VLMEvaluator(
        model=model,
        tokenizer=tokenizer,
        benchmarks=args.benchmarks,
        output_dir=args.output_dir,
    )

    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate()

    # Print results
    report = evaluator.generate_report(results)
    print(report)

    # Benchmark efficiency
    if args.benchmark_efficiency:
        logger.info("Benchmarking efficiency...")
        efficiency = evaluator.benchmark_efficiency()

        print("\n" + "=" * 50)
        print("EFFICIENCY METRICS")
        print("=" * 50)
        for metric, value in efficiency.items():
            print(f"{metric}: {value:.2f}")

    logger.info("=" * 50)
    logger.info("Evaluation Complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
