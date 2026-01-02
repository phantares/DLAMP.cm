# Project Name

> A generative AI model for high fidelity 3D cloud structure downscaling.

## 🛠 Tech Stack

* **Package Manager:** [uv](https://github.com/astral-sh/uv)
* **Configuration:** [Hydra](https://hydra.cc/)
* **Logging:** [Weights & Biases (WandB)](https://wandb.ai/)
* **Training:** [PyTorch Lightning](https://lightning.ai/)

---

## 🚀 Installation & Setup

### 1. Install `uv`
This project uses `uv` for fast dependency management. If you don't have it installed, run:

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

### 2. Install Dependencies
Sync the project environment to install all required packages:

```bash
uv sync
```

### 3. Configuration
Set up your environment variables. Copy the example file to a new `.env` file:

```bash
cp .env.example .env
```

Edit paths in `.env` using your preferred editor:

```bash
vi .env
```

### 4. Logging 
Experiments are tracked using Weights & Biases.

Before running training for the first time, ensure you are logged in:

```bash
uv run wandb login
```

## 🧠 Training
To start training with the default configuration, run:
```bash
uv run python train.py
```

### Customizing the Run
You can override any configuration setting directly from the command line without editing the config.yaml files. Simply append the parameters you want to change using the key=value syntax:

```bash
uv run python train.py parameter=value
```