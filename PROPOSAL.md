# Project Proposal: Medical Specialty Classification from Clinical Transcriptions

## Project Option

**Modeling Experiment**– Building a classification model using labeled clinical transcription text.


## 1. Dataset
**Source** : Medical Transcriptions Dataset on Kaggle
[https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions]

**Description**:  
The collection includes anonymised medical transcription records from a range of specialities, including gastrointestinal, dermatology, and cardiology.  Every record has a label identifying the relevant medical speciality and a free-text transcription.


## 2. Project Goal
Building a **multi-class classification model** that can forecast the **medical speciality** from the content of a clinical transcription is the aim of this research.


## 3. Modeling Plan

### Preprocessing:
- Text cleaning and normalization
- Tokenization and stopword removal
- Label encoding

### Feature Engineering:
- TF-IDF vectorization
- Word embeddings (e.g., GloVe)
- Contextual embeddings using transformer models (e.g., DistilBERT or BioBERT)

### Models:
- Logistic Regression
- Random Forest
- Fine-tuned transformer model(DistilBERT)


## 4. Evaluation
Performance will be assessed using:
- Accuracy
- F1-score
- Confusion matrix

These metrics will help evaluate how well each model handles the classification across multiple specialties.


## 5. Project Development
The project will be developed in a **Jupyter Notebook** and tracked in this GitHub repository:  
[https://github.com/Bharath-ch40/NLP-Modeling-Experiment.git]


## 6. Originality
Although there are generic tutorials on medical text categorisation, the goal of this research is to compare transformer-based models (DistilBERT/BioBERT) with classic machine learning approaches (such as TF-IDF + Logistic Regression) on the particular objective of medical speciality prediction.  This project's distinctive significance is found in its practical assessment of various modelling techniques together with its emphasis on domain-specific categorisation using actual clinical narratives.
