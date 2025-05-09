# NLP-Modeling-Experiment
Identifying Medical Specialties from Clinical Narratives using NLP

**Introduction:**

Hospitals and clinics generate a lot of written notes every day — from patient checkups to specialist consultations. These notes, known as medical transcriptions, are often unstructured and come from different areas like cardiology, dermatology, or psychiatry. Sorting these notes manually by specialty takes time and can lead to mistakes.

In this project, we wanted to see if we could build a model that reads a transcription and guesses which medical specialty it belongs to. To do this, we tried out different types of models — from simpler ones like Logistic Regression and Random Forest to more advanced ones like DistilBERT, a type of deep learning model that understands language better.

Our goal was to figure out which method works best for this task and what kinds of specialties are easier or harder for the models to predict. This kind of tool could one day help hospitals save time and keep their records better organized.

**Dataset**(https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions):

For this project, we have utilized the Medical Transcriptions Dataset available on Kaggle. This dataset comprises over 5,000 anonymized clinical notes, each labeled with a specific medical specialty such as cardiology, dermatology, or psychiatry.

Each entry in the dataset includes:

Medical Specialty: The category of the medical field.

Transcription: The full text of the clinical note.

Keywords: Relevant terms associated with the transcription.

Sample Name: The title of the sample.

Description: A brief overview of the note.

This diverse collection of transcriptions presents a realistic challenge for classification tasks, as some specialties have overlapping terminology, while others are more distinct.

## Environment Setup

To install the required packages, run:

```bash
pip install -r requirements.txt

```
**Project Goal**

The main goal of this project is to build a model that can predict the medical specialty just by reading the text of a clinical transcription.

We wanted to explore how well different types of models — from basic machine learning to modern deep learning — could handle this kind of multi-class classification task. In total, we looked at over 40 different medical specialties.

The idea is that if a model can learn to recognize patterns in how different specialists write their notes, it could help automatically organize medical records or assist with hospital workflow in the future.

**Preprocessing & Feature Engineering** :

Preprocessing Steps:
1. Dropped rows with missing values in transcription or medical_specialty
2. Converted all transcription text to lowercase
3. Removed punctuation, numbers, and extra spaces using regex
4. Removed stopwords using NLTK's English stopword list
5. Created a new column clean_transcription with the cleaned text
6. Encoded the medical_specialty column into numeric labels using LabelEncoder

Feature Extraction:
1. Applied TF-IDF Vectorization (max_features=5000) for classic ML models
2. Used DistilBERT Tokenizer and Model to generate sentence embeddings ([CLS] token) for deep learning
3. For fine-tuning: tokenized the text using DistilBertTokenizerFast and used Hugging Face’s Dataset format


Modeling 

We trained and compared several models to predict the medical specialty based on the cleaned transcription text.

Classic Machine Learning Models (using TF-IDF features)
* Logistic Regression: Trained on TF-IDF features with max_iter=1000.
* Random Forest Classifier:
    * First trained with default settings.
    * Then tuned using:
        * GridSearchCV (tested multiple combinations of depth, splits, and features)
        * RandomizedSearchCV (explored a larger range with random combinations)

Deep Learning Model
* DistilBERT Embedding Extraction:
    * Loaded pre-trained distilbert-base-uncased
    * Used it to extract [CLS] embeddings for 2,000 sampled transcriptions
* Fine-tuned DistilBERT:
    * Converted data into Hugging Face Dataset
    * Used DistilBertForSequenceClassification
    * Trained for 1 epoch on 2,000 samples with a batch size of 8
    * Used AdamW optimizer and CrossEntropyLoss

Evaluation & Results

After training the models, we evaluated how well each one performed using common classification metrics:
Metrics Used
* Accuracy – how often the model got the correct label
* F1-Score – balances precision and recall, useful for imbalanced classes
* Confusion Matrix – to visualize where models were making mistakes
* Per-class F1-scores – to see which specialties were easier or harder to predict


Model Performance

We tested four different models to see how well they could predict the medical specialty from a transcription. 

The Logistic Regression model gave us a baseline accuracy of 26%, while the Random Forest model with basic tuning using GridSearch improved that to 34%. 

The best results came from the Random Forest model tuned with RandomizedSearchCV, which reached an accuracy of 71.75% , making it the top-performing model in this project. 

Surprisingly, the fine-tuned DistilBERT model only reached 37% accuracy, which was lower than expected for a deep learning model.

This lower result is likely due to using a small sample size (2,000 notes), limited training (1 epoch), and no domain-specific pretraining. Random Forest performed better thanks to data balancing, TF-IDF features, and effective hyper parameter tuning.

When looking at specialty-wise performance, the model did best on transcriptions from Gastroenterology, Dermatology, and Psychiatry. 

These notes likely have more distinct terms that make them easier to classify. On the other hand, specialties like Internal Medicine and Radiology were the most challenging for the model. 

These transcriptions tend to be broader and may share vocabulary with multiple fields, making them harder to predict correctly.

To better understand model behavior, we created several visualizations. These included confusion matrices to show where models made mistakes, bar charts of F1 scores for each specialty, and plots highlighting the top 10 best and worst performing specialties. 

These visual insights helped us see not just how accurate the models were, but also which areas they struggled with the most.

Conclusion:
In this project, we explored different models to predict medical specialties from clinical transcriptions. Random Forest with hyperparameter tuning gave the best results with 71.75% accuracy, outperforming both Logistic Regression and DistilBERT. While we expected the deep learning model to do better, its performance was limited by data size and training time. Some specialties were easier to classify than others based on how distinct their language was. This project showed that with proper tuning and preprocessing, classic models can still be highly effective in real-world NLP tasks.


In the extension to it I have thought about the future possible work that can be done for this project

1. Use a larger dataset: Fine-tuning models like DistilBERT on more than 2,000 samples will likely improve performance.

2. Train for more epochs: Running the transformer model for multiple epochs can help it better learn the patterns in the data.
   
3. Try domain-specific models like BioBERT: These are pre-trained on medical texts and may perform better than general-purpose models.
