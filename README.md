Multi-Class [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EasonJia9598/BERT_Radiology_Analysis/blob/main/multi_class_BERT%20(2).ipynb)

Multi-Label [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EasonJia9598/BERT_Radiology_Analysis/blob/main/multi_label%20(2).ipynb)


# Abstract

The medical resources are becoming increasingly
scarce due to the pandemic and
the staff shortage of hospitals workers. Doctors
are facing more cases than they can
handle, which is causing missed diagnosis
and leading to more overtime working.
The raising of deep learning model of
NLP brings us the potential to solve this
task even if we don’t have the corresponding
medical domain linguistic knowledge.
In this paper, we will train and modify
the original Transformers pre-trained model
BERT on our radiology domain dataset for
doing two main downstream tasks - Named
Entity recognition task and examinations
prediction. We compared the performance
of BERT-base model with whether trained
or not trained tokenizer on a publicly NER
dataset. And we also compare the result
with other people’s medical BERT model
on the same dataset. On the examinations
prediction task, we tried two different methods.
Those are multi-class and multi-label
prediction BERT model. Overall, we got an
74% accuracy by the multi-label model and
59% by the multi-class model. In the meantime,
the normal BERT model only has 42%
accuracy.

# RadiBERT: Pre-trained BERT Model on Radiology Reports

## Overview
RadiBERT is a pre-trained BERT-based language model fine-tuned on radiology reports for two main downstream tasks:

1. **Named Entity Recognition (NER)** - Identifying key medical terms in radiology reports.
2. **Examinations Prediction** - Predicting potential examinations required for patients based on radiology reports.

This project evaluates BERT's performance in the medical domain by fine-tuning it on radiology data and comparing different approaches for prediction tasks.

---

## Features
- **Domain-Specific BERT Fine-Tuning:** Adapts a BERT model for medical text processing.
- **Named Entity Recognition (NER):** Extracts key observations from radiology reports.
- **Multi-Class & Multi-Label Classification:** Predicts required examinations for patients.
- **Comparison with Medical BERT Models:** Evaluates performance against other medical NLP models.
- **Custom Tokenizer Training:** Trains a tokenizer specifically for medical terms.

---

## Dataset
### 1. **RadGraph Dataset**
- Radiology reports dataset created by Stanford.
- Contains 500 annotated reports with 14,579 entities and 10,889 relations.
- Four entity types: `ANAT-DP`, `OBS-DP`, `OBS-U`, `OBS-DA`.
- Three relation types: `suggestive of`, `located at`, `modify`.

### 2. **NCBI Disease Dataset**
- Public dataset used to demonstrate model performance.
- Contains disease name and concept annotations from PubMed abstracts.

---

## Methodology
### Named Entity Recognition (NER)
1. Train a custom tokenizer using medical text corpus.
2. Fine-tune a BERT-base model with and without the custom tokenizer.
3. Evaluate model performance on the NER task.

### Examinations Prediction
#### **Multi-Class Classification**
- Each report is labeled with a single examination type (from 121 possible labels).
- Uses standard BERT classification with a softmax output layer.

#### **Multi-Label Classification**
- Each report may contain multiple examination labels.
- Uses one-hot encoding to represent labels.
- Trains BERT with a sigmoid output layer for multi-label predictions.

---

## Model Performance
| Model | Accuracy | F1 Score | ROC-AUC |
|--------|----------|---------|---------|
| Multi-Class BERT | 59% | - | - |
| Multi-Label BERT | **74%** | **81.24%** | **89.72%** |
| Baseline BERT | 42% | - | - |

---

## Installation
### Requirements
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Tokenizers
- Scikit-learn

### Setup
```sh
pip install -r requirements.txt
```

---

## Usage
### Train the Model
```sh
python train.py --task ner --dataset radgraph
```

```sh
python train.py --task classification --dataset examinations --model multi_label
```

### Inference
```sh
python inference.py --text "Sample radiology report text..."
```

---

## Results and Observations
- Custom tokenizer slightly improved entity recognition but did not significantly outperform the base tokenizer.
- Multi-label classification outperformed multi-class classification in examination prediction.
- BERT models fine-tuned on medical data perform significantly better than general BERT models.

---

## Future Improvements
- Train a larger tokenizer with more domain-specific text.
- Fine-tune models with additional medical datasets.
- Experiment with different BERT variants (e.g., BioBERT, ClinicalBERT).
- Implement attention-based interpretability for NER predictions.

---

## Citation
If you use this project, please cite:
```bibtex
@article{jia2025radiBERT,
  author = {Zesheng Jia},
  title = {RadiBERT: Pre-trained BERT Model on Radiology Reports with Named Entity Recognition and Examination Prediction},
  year = {2025}
}
```

---


