# Research Log: PTQ vs QAT for Small VLMs

## Project Overview

**Research Question**: How do Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) compare in accuracy-efficiency trade-offs for small Vision-Language Models (1-3B parameters)?

**Timeline**: 12-13 weeks
**Target**: Submittable research paper

---

## Week 1-2: Foundation & Setup

### 2024-10-31 - Project Initialization

**Completed:**
- [x] Created repository structure
- [x] Set up Python environment configuration
- [x] Implemented PTQ scaffolding (GPTQ, AWQ)
- [x] Implemented QAT scaffolding
- [x] Created evaluation framework structure
- [x] Set up configuration files for experiments
- [x] Created executable scripts for running experiments

**Project Structure:**
```
Qvlm/
├── src/                 # Source code
│   ├── ptq/            # PTQ implementations (GPTQ, AWQ)
│   ├── qat/            # QAT implementation
│   ├── evaluation/     # Evaluation framework
│   ├── models/         # Model loading utilities
│   ├── data/           # Data loading (to be implemented)
│   └── utils/          # Utility functions
├── configs/            # YAML configuration files
├── scripts/            # Executable scripts
├── experiments/        # Experiment outputs
└── docs/               # Documentation
```

**Next Steps:**
- [ ] Download and test base VLM model (LLaVA-Phi-3B)
- [ ] Implement calibration dataset loading
- [ ] Run FP16 baseline evaluation
- [ ] Test PTQ implementation (GPTQ 4-bit)
- [ ] Review and refine evaluation metrics

---

## Week 3: Literature Review

### Key Papers to Read

**PTQ for VLMs:**
- [ ] MBQ: Modality-Balanced Quantization
- [ ] VLMQ: Efficient PTQ via Hessian Augmentation
- [ ] Q-VLM: Post-training Quantization for Large VLMs (NeurIPS 2024)

**QAT Techniques:**
- [ ] EfficientQAT: Block-wise quantization-aware training
- [ ] Progressive Quantization for 2-bit models

**Comparative Analysis:**
- [ ] Comprehensive PTQ vs QAT analysis papers
- [ ] NVIDIA's QAT guide

### Notes and Insights

*Add notes here as you read papers...*

---

## Week 4: Experimental Design

### Target Models

1. **LLaVA-Phi-3B** (Primary)
   - 3B parameters
   - HF path: `xtuner/llava-phi-3-mini-hf`
   - Status: Not yet downloaded

2. **Qwen2-VL-2B** (Secondary)
   - 2B parameters
   - HF path: `Qwen/Qwen2-VL-2B-Instruct`
   - Status: Not yet evaluated

### Quantization Configuration Matrix

| Method | Bits | Group Size | Calibration Samples | Status |
|--------|------|------------|---------------------|--------|
| GPTQ   | 4    | 128        | 512                 | Config ready |
| GPTQ   | 8    | 128        | 512                 | Config ready |
| AWQ    | 4    | 128        | 128                 | Config ready |
| QAT    | 4    | -          | Full training       | Config ready |
| QAT    | 8    | -          | Full training       | Config ready |

### Evaluation Benchmarks

- [ ] VQAv2: General VQA
- [ ] POPE: Hallucination detection
- [ ] MME: Perception and cognition
- [ ] MMBench: Multi-domain evaluation
- [ ] TextVQA: Text-oriented VQA
- [ ] GQA: Compositional reasoning

---

## Experiments Log

### Experiment 1: Baseline FP16 Evaluation

**Date:**
**Status:** Not started

**Configuration:**
- Model: LLaVA-Phi-3B
- Precision: FP16
- Benchmarks: VQAv2, POPE, MME, MMBench

**Results:**
- TBD

**Notes:**
- TBD

---

### Experiment 2: PTQ GPTQ 4-bit

**Date:**
**Status:** Not started

**Configuration:**
- Model: LLaVA-Phi-3B
- Method: GPTQ
- Bits: 4
- Group size: 128
- Calibration samples: 512

**Results:**
- TBD

**Notes:**
- TBD

---

### Experiment 3: PTQ GPTQ 8-bit

**Date:**
**Status:** Not started

---

### Experiment 4: PTQ AWQ 4-bit

**Date:**
**Status:** Not started

---

### Experiment 5: QAT 4-bit

**Date:**
**Status:** Not started

---

### Experiment 6: QAT 8-bit

**Date:**
**Status:** Not started

---

## Analysis & Insights

### Observations

*Add observations here as experiments progress...*

### Challenges

*Document challenges encountered...*

### Novel Contributions

*Ideas for novel contributions:*
- Modality-Aware Hybrid Approach
- Progressive Quantization
- Efficiency-Accuracy Pareto Analysis

---

## Paper Writing Progress

### Outline Status

- [ ] Abstract
- [ ] Introduction
- [ ] Related Work
- [ ] Methodology
- [ ] Experiments
- [ ] Analysis & Discussion
- [ ] Conclusion
- [ ] References

---

## Meeting Notes

*Add meeting notes here...*

---

## TODO List

### Immediate (Week 1-2)
- [ ] Set up GPU environment
- [ ] Download LLaVA-Phi-3B model
- [ ] Prepare calibration dataset (COCO)
- [ ] Run baseline FP16 evaluation
- [ ] Test GPTQ implementation

### Short-term (Week 3-4)
- [ ] Complete literature review
- [ ] Finalize experimental protocol
- [ ] Set up experiment tracking (W&B or MLflow)
- [ ] Implement benchmark data loaders

### Medium-term (Week 5-8)
- [ ] Run all PTQ experiments
- [ ] Run all QAT experiments
- [ ] Conduct ablation studies
- [ ] Analyze results

### Long-term (Week 9-12)
- [ ] Write paper draft
- [ ] Create figures and tables
- [ ] Prepare for submission
- [ ] Code cleanup and release

---

## Resources

### Useful Links

- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [AWQ](https://github.com/mit-han-lab/llm-awq)
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)

### Hardware

- Required: 1x RTX 4090 (24GB) or A100 (40GB)
- Available: TBD

---

## Version History

- **v0.1.0** (2024-10-31): Initial project setup
