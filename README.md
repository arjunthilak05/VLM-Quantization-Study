# PTQ vs QAT for Small Vision-Language Models

A comprehensive comparative study of Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) methods for small Vision-Language Models (1-3B parameters).

## 🎯 Research Objectives

This research aims to provide a systematic comparison of PTQ and QAT techniques applied to small VLMs, addressing:

1. **Primary Research Question**: How do PTQ and QAT compare in accuracy-efficiency trade-offs for small VLMs (1-3B parameters)?
2. **Secondary Questions**:
   - Which bit-widths (4-bit, 8-bit) offer optimal trade-offs?
   - How do modality-specific sensitivities affect quantization?
   - What is the computational cost difference between methods?

## 📋 Project Structure

```
Qvlm/
├── src/                          # Source code
│   ├── ptq/                      # Post-Training Quantization implementations
│   ├── qat/                      # Quantization-Aware Training implementations
│   ├── evaluation/               # Evaluation and benchmarking code
│   ├── models/                   # Model loading and configuration
│   ├── data/                     # Data loading and preprocessing
│   └── utils/                    # Utility functions
├── configs/                      # Configuration files
├── experiments/                  # Experiment outputs
│   ├── results/                  # Evaluation results
│   ├── logs/                     # Training/quantization logs
│   └── checkpoints/              # Model checkpoints
├── notebooks/                    # Jupyter notebooks for analysis
├── docs/                         # Documentation
│   ├── papers/                   # Related papers and notes
│   └── notes/                    # Research notes and logs
├── scripts/                      # Utility scripts
└── requirements.txt              # Python dependencies
```

## 🔬 Target Models

- **LLaVA-Phi-3B**: 3B parameters, achieves performance comparable to 7B models
- **Instella-VL-1B**: 1B parameters, recently released with full training details
- **TinyGPT-V**: 2.8B parameters, trainable on 24GB GPU

## 🧪 Quantization Methods

### Post-Training Quantization (PTQ)
- **GPTQ**: Layer-wise optimization using Hessian matrix
- **AWQ**: Activation-aware weight quantization

### Quantization-Aware Training (QAT)
- Block-wise quantization with end-to-end step size training
- Progressive quantization combining PTQ and QAT

## 📊 Evaluation Benchmarks

Using VLMEvalKit with the following benchmarks:
- **VQAv2**: General visual question answering
- **POPE**: Hallucination detection
- **MME**: Perception and cognition evaluation
- **MMBench**: Multi-domain evaluation
- **TextVQA**: Text-oriented VQA
- **GQA**: Compositional visual reasoning

## 📈 Evaluation Metrics

- **Accuracy** (primary metric)
- **Inference Speed** (tokens/sec)
- **Memory Usage** (GB)
- **Model Size** (GB)
- **Quantization Time** (hours)
- **Perplexity** (for calibration datasets)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/arjunthilak05/Qvlm.git
cd Qvlm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running PTQ

```bash
python scripts/run_ptq.py --config configs/ptq_gptq_4bit.yaml
```

### Running QAT

```bash
python scripts/run_qat.py --config configs/qat_4bit.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --model_path experiments/checkpoints/model_ptq_4bit --benchmarks VQAv2 POPE MME
```

## 📅 Research Timeline

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| Setup | 2 weeks | Working environment, baseline results |
| Literature Review | 1 week | Theoretical framework |
| Experimental Design | 1 week | Experimental protocol |
| Evaluation Framework | 1 week | Benchmark pipeline |
| Experimentation | 4 weeks | Complete results |
| Analysis | 2 weeks | Novel insights |
| Writing | 2 weeks | Draft paper |
| **Total** | **12-13 weeks** | **Submittable paper** |

## 🔑 Key Features

- **Modular Design**: Easy to extend with new quantization methods
- **Comprehensive Evaluation**: Multiple benchmarks and metrics
- **Experiment Tracking**: Integration with Weights & Biases / MLflow
- **Reproducibility**: Fixed seeds, detailed logging, configuration management

## 📚 References

### Key Papers

**PTQ for VLMs:**
- MBQ: Modality-Balanced Quantization
- VLMQ: Efficient PTQ via Hessian Augmentation
- Q-VLM: Post-training Quantization for Large VLMs (NeurIPS 2024)

**QAT Techniques:**
- EfficientQAT: Block-wise quantization-aware training
- Progressive Quantization for 2-bit models

## 🤝 Contributing

This is a research project. For questions or collaboration inquiries, please open an issue.

## 📄 License

[Add your license here]

## 📧 Contact

[Add your contact information here]

## 🏗️ Development Status

- [x] Initial setup
- [ ] PTQ implementation (GPTQ)
- [ ] PTQ implementation (AWQ)
- [ ] QAT implementation
- [ ] Evaluation framework
- [ ] Baseline experiments
- [ ] Full experimental suite
- [ ] Paper writing

## 💡 Research Notes

See `docs/notes/research_log.md` for detailed research progress and insights.
