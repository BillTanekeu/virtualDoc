# virtualDoc 🩺

`virtualDoc` is an advanced medical diagnostic system that combines **Reinforcement Learning (RL)** and **Large Language Models (LLMs)** to provide an interactive, conversational symptom-checking experience.

The system uses a **Policy Gradient (REINFORCE)** agent to intelligently select the most informative symptoms to inquire about, minimizing the number of questions while maximizing diagnostic accuracy. An LLM (phi4-mini via Ollama) handles Natural Language Understanding (NLU) and Generation (NLG) to make the interaction feel natural and empathetic.

---

##  Key Features

- **Intelligent Symptom Acquisition**: Uses an AARLC (Adaptive Alignment Reinforcement Learning for Medical Consultation) architecture to decide which symptom to ask for next.
- **Conversational Interface**: Powered by **Ollama (phi4-mini)** for natural language interaction, capable of understanding free-text responses and providing empathetic clarifications.
- **Adaptive Confidence Thresholding**: Automatically determines when enough information has been gathered to provide a high-confidence diagnosis.
- **Data-Driven Preprocessing**: Built-in pipeline for discretizing continuous medical data (like severity and duration) to optimize RL state-space.
- **French Language Support**: Configured for medical consultations in French.

---

##  Project Structure

```text
virtualDoc/
├── configs/            # JSON configuration files (disease list, thresholds, etc.)
├── data/               # Raw and processed medical datasets (DDXPlus format)
├── env/                # Custom Reinforcement Learning environment (environment.py)
├── models/             # RL agent architecture (agent.py)
├── scripts/            # Core pipeline scripts (01-05)
│   ├── 01_preprocess.py     # Data cleaning and discretization
│   ├── 02_validate_dataset.py # Dataset integrity checks
│   ├── 03_train.py          # Training the RL and Classifier networks
│   ├── 04_evaluate.py       # Performance evaluation on test set
│   ├── 05_chatbot.py        # Interactive conversational chatbot
│   └── llm_interface.py     # Ollama API wrapper for NLU/NLG
├── output/             # Saved models, logs, and evaluation reports
└── requirements.txt    # Python dependencies
```

---

##  Getting Started

### 1. Prerequisites

- **Python 3.10+**
- **Ollama**: Installed and running locally.
- **LLM Model**: Pull the requested model (default: `phi4-mini`).
  ```bash
  ollama pull phi4-mini
  ```

### 2. Installation

Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Pipeline Usage

#### Step 1: Preprocessing
Clean and discretize the raw data in `data/raw/`:
```bash
python3 scripts/01_preprocess.py
```

#### Step 2: Training
Train the RL agent and the dual-classifier:
```bash
python3 scripts/03_train.py
```
*Note: This will save the best model and adaptive thresholds to `output/models/`.*

#### Step 3: Evaluation
Run benchmarks on the test set:
```bash
python3 scripts/04_evaluate.py
```

#### Step 4: Chat with virtualDoc
Start the interactive medical consultation:
```bash
python3 scripts/05_chatbot.py
```

---

## Technical Overview

- **RL Agent**: Policy Gradient with a dual-stream architecture (Policy vs. Diagnosis).
- **Environment**: Based on the DDXPlus simulator, adapted for discretized attributes.
- **LLM Strategy**: 
  - **NLU**: Uses hybrid Regex + LLM-streaming for ultra-fast intent detection and symptom extraction.
  - **NLG**: Streaming-based empathetic question generation and pedagogical explanations.

---

## License
This project is licensed under the terms of the LICENSE file included in the repository.
