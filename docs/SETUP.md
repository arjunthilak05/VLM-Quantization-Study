# Setup Guide

This guide will help you set up the development environment for the PTQ vs QAT research project.

## Prerequisites

- Python 3.8 or higher
- CUDA 11.8 or higher (for GPU support)
- At least 24GB GPU memory (RTX 4090 or A100 recommended)
- 50GB+ free disk space

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/arjunthilak05/Qvlm.git
cd Qvlm
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Setup Script (Optional)

```bash
bash setup.sh
```

This will:
- Create necessary directories
- Set up Python package structure
- Initialize log files

## Downloading Models

### LLaVA-Phi-3B (Primary Model)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "xtuner/llava-phi-3-mini-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save locally
model.save_pretrained("models/weights/llava-phi-3b")
tokenizer.save_pretrained("models/weights/llava-phi-3b")
```

Or use the HuggingFace CLI:

```bash
huggingface-cli download xtuner/llava-phi-3-mini-hf --local-dir models/weights/llava-phi-3b
```

## Preparing Datasets

### Calibration Dataset (COCO)

For PTQ calibration, you'll need a subset of the COCO dataset:

```bash
# Create data directory
mkdir -p data/calibration

# Download COCO (you can use a subset)
# Instructions: http://cocodataset.org/#download
```

### Evaluation Datasets

Follow instructions for each benchmark:

- **VQAv2**: https://visualqa.org/download.html
- **POPE**: https://github.com/AoiDragon/POPE
- **MME**: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models
- **MMBench**: https://github.com/open-compass/MMBench

## Verifying Installation

Run a quick test to verify everything is installed correctly:

```python
import torch
import transformers
from auto_gptq import AutoGPTQForCausalLM

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers version: {transformers.__version__}")
print("All imports successful!")
```

## Configuration

### Experiment Tracking

#### Weights & Biases

```bash
pip install wandb
wandb login
```

Configure in your config file:
```yaml
logging:
  use_wandb: true
  wandb_project: "qvlm-research"
  wandb_entity: "your-username"
```

#### MLflow

```bash
pip install mlflow
```

Start MLflow server:
```bash
mlflow ui
```

## Running Your First Experiment

### 1. Baseline Evaluation (FP16)

```bash
python scripts/evaluate.py \
  --model-path models/weights/llava-phi-3b \
  --benchmarks VQAv2 POPE \
  --output-dir experiments/results/baseline
```

### 2. PTQ GPTQ 4-bit

```bash
python scripts/run_ptq.py --config configs/ptq_gptq_4bit.yaml
```

### 3. QAT 4-bit

```bash
python scripts/run_qat.py --config configs/qat_4bit.yaml
```

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size in config files
- Use gradient checkpointing
- Try model parallelism with `device_map="auto"`

### Model Download Issues

- Use HuggingFace CLI with authentication:
  ```bash
  huggingface-cli login
  ```

### Import Errors

- Ensure virtual environment is activated
- Reinstall requirements:
  ```bash
  pip install -r requirements.txt --force-reinstall
  ```

## Next Steps

1. Read the [Research Log](notes/research_log.md) to understand the project timeline
2. Review the [README.md](../README.md) for project overview
3. Check configuration files in `configs/` directory
4. Start with baseline experiments

## Additional Resources

- [LLaVA Documentation](https://github.com/haotian-liu/LLaVA)
- [AutoGPTQ Guide](https://github.com/PanQiWei/AutoGPTQ)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
