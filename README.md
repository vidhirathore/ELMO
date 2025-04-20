# ELMo Embedding and Text Classification Project

This project implements an ELMo-like model for generating contextualized word embeddings, pre-trains it on the Brown Corpus, and subsequently uses these embeddings for a downstream text classification task on the AG News dataset. It also explores and compares different strategies for combining the internal layer representations of the ELMo model for the classification task.

## Overview

The project consists of several components:

1. **ELMo Pre-training (`elmo.py`):** Defines and trains a Bidirectional LSTM model (similar to ELMo's architecture) on the Brown Corpus to learn contextual word embeddings. It saves the trained model weights and the vocabulary.
2. **AG News Classification (`classification.py`):** Uses the pre-trained ELMo model to generate embeddings for AG News descriptions and trains a downstream classifier (LSTM-based) to predict news categories. Uses a fixed embedding combination strategy (concatenation).
3. **Hyperparameter Tuning (`calc.py`):** Compares three different methods for combining the simulated ELMo layer embeddings for the AG News classification task:
   * `trainable_lambdas`: Learns scalar weights (lambdas) for combining embeddings.
   * `frozen_lambdas`: Uses fixed, randomly initialized lambdas.
   * `learnable_function`: Uses a small neural network to learn a combination function.
   
   It trains, evaluates, generates plots (loss, accuracy, confusion matrices), and saves results for each method.
4. **Inference (`inference.py`):** Loads a trained classifier (either from `classification.py` or `calc.py`), the pre-trained ELMo model, and the vocabulary to predict the category of a new, unseen text description provided via the command line.
5. **Report (`report.md`):** Summarizes the findings from the hyperparameter tuning experiments (`calc.py`) and compares ELMo's performance against other traditional embedding techniques.

## Requirements

* Python 3.7+
* PyTorch (`torch`)
* Pandas (`pandas`)
* NumPy (`numpy`)
* Scikit-learn (`scikit-learn`)
* Matplotlib (`matplotlib`)
* Seaborn (`seaborn`)
* NLTK (`nltk`)
* Brown Corpus (via NLTK or downloaded separately)
* AG News Dataset (specifically `train.csv`)

You can install the Python dependencies using pip:

```bash
pip install -r requirements.txt
```

### requirements.txt:
```
torch
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
```

## Setup

Clone the repository:
```bash
git clone [<your-repo-url>](https://github.com/vidhirathore/ELMO.git)
cd ELMO
```

Create a virtual environment (Recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download NLTK data:
Run Python and execute:
```python
import nltk
nltk.download('punkt')
nltk.download('brown') # If you want NLTK to manage the corpus
```

### Prepare Brown Corpus:
The script elmo.py expects the Brown Corpus files to be located in `./brown/brown/brown/`.
If using NLTK's download, you might need to adjust the `corpus_path` in elmo.py or copy/link the NLTK data directory accordingly. Ensure the directory structure matches what `load_brown_corpus` expects (a directory containing the individual corpus files like ca01, cb01, etc.).

### Prepare AG News Dataset:
Place the AG News `train.csv` file inside a directory named `news` within the ELMO directory. The expected path is `ELMO/news/train.csv`. You can typically find this dataset on Kaggle or other machine learning dataset repositories.

## Usage

**Important**: Run the scripts from the parent directory containing the ELMO folder, not from within the ELMO folder itself, unless you adjust the file paths within the scripts (like model/vocab loading/saving locations and data paths).

### 1. Pre-train the ELMo Model (elmo.py)
Purpose: Trains the biLSTM model on the Brown Corpus and saves the weights (bilstm.pt) and vocabulary (vocab.json). This must be run first.

Command:
```bash
python ELMO/elmo.py
```

Output: Creates `bilstm.pt` and `vocab.json` in the root project directory. This step can take a significant amount of time depending on your hardware and the number of epochs.

### 2. Train the Basic AG News Classifier (classification.py)
Purpose: Trains a classifier using concatenated ELMo embeddings.

Prerequisites: `bilstm.pt`, `vocab.json`, `ELMO/news/train.csv`.

Command:
```bash
python ELMO/classification.py
```

Output: Prints training progress (loss, accuracy per epoch) and saves the trained classifier model as `ag_news_classifier.pt` in the root project directory.

### 3. Run Hyperparameter Tuning (calc.py)
Purpose: Trains and compares the three different embedding combination strategies.

Prerequisites: `bilstm.pt`, `vocab.json`, `ELMO/news/train.csv`.

Command:
```bash
python ELMO/calc.py
```

Output:
- Creates a `results/` directory in the root project directory.
- Inside `results/`, saves:
  - Training history plots (`training_history_*.png`)
  - Confusion matrices (`confusion_matrix_*.png`)
  - Saved models for each mode (`model_*.pt`)
  - A comparison plot (`methods_comparison.png`)
- Prints training progress and final comparison results to the console.

### 4. Run Inference (inference.py)
Purpose: Predict the category for a given text description using a pre-trained classifier.

Prerequisites: `bilstm.pt`, `vocab.json`, and a saved classifier model (e.g., `ag_news_classifier.pt` from step 2 or `results/model_trainable_lambdas.pt` from step 3).

Command:
```bash
python ELMO/inference.py <path_to_saved_classifier.pt> "Your text description here"
```

Example using the basic classifier:
```bash
python ELMO/inference.py ag_news_classifier.pt "Rocket launch successful, placing new satellite into orbit for weather monitoring."
```

Example using a model from hyperparameter tuning:
```bash
python ELMO/inference.py results/model_trainable_lambdas.pt "Stock market dips slightly after federal reserve announcement on interest rates."
```

Output: Prints the predicted probabilities for each AG News category (World, Sports, Business, Sci/Tech).

## File Structure
```
.
├── ELMO/
│   ├── calc.py             # Hyperparameter tuning script
│   ├── classification.py   # Basic classification script
│   ├── elmo.py             # ELMo model definition and pre-training
│   ├── inference.py        # Inference script
│   ├── report.md           # Results report
│   ├── vocab.json          # Vocabulary (generated by elmo.py)
│   └── news/
│       └── train.csv       # AG News dataset (needs to be added)
├── brown/
│   └── brown/
│       └── brown/          # Brown Corpus files (needs to be added/linked)
│           ├── ca01
│           ├── ca02
│           └── ...
├── results/                # Directory for calc.py outputs (generated by calc.py)
│   ├── confusion_matrix_*.png
│   ├── training_history_*.png
│   ├── model_*.pt
│   └── methods_comparison.png
├── bilstm.pt               # Pre-trained ELMo model weights (generated by elmo.py)
├── ag_news_classifier.pt   # Basic classifier model (generated by classification.py)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Results Summary

The hyperparameter tuning script (`calc.py`) compares different methods for combining ELMo embeddings. According to the results generated and summarized in `ELMO/report.md`, the Trainable Lambdas approach yielded the best classification accuracy on the AG News test set. Detailed plots and confusion matrices for each method can be found in the `results/` directory after running `calc.py`. The report also includes a comparison with non-contextual embedding methods.
