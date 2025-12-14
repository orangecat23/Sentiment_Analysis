# Sentiment_analysis

Restaurant Review Sentiment Classifier
This project trains a sentiment analysis model using DistilBERT to classify restaurant reviews as positive or negative. The model is fine-tuned on combined datasets from Yelp, Zomato, and a third review dataset.

📂 Project Overview
Key Steps:

1.Load and clean review datasets.

2.Combine and label reviews (positive if rating ≥ 4, negative otherwise).

3.Tokenize reviews with DistilBERT tokenizer.

4.Fine-tune DistilBertForSequenceClassification.

5.Evaluate and save the trained model.

6.Create a pipeline for inference on new reviews.

🛠️ Setup
Install Dependencies

pip install transformers datasets scikit-learn pandas
📈 Training Details
Model: distilbert-base-uncased

Dataset Split: 80% training, 20% validation

Epochs: 3

Batch Size: 16 (train), 8 (eval)

Learning Rate: 2e-5

Metrics Tracked: Accuracy, F1, Precision, Recall

Hardware: Supports GPU (CUDA) acceleration if available

🧩 Project Structure
Data Loading: Reads CSVs from Google Drive:

yelp.csv

zomato_reviews.csv

restrev.csv

Cleaning Function: Standardizes columns and removes missing data.

Labeling:

Ratings ≥ 4 → Positive (1)

Ratings < 4 → Negative (0)

Tokenizer: DistilBERT tokenizer with truncation and padding.

Dataset Class: PyTorch Dataset for input to Trainer.

Trainer: Fine-tunes the model and evaluates per epoch.

Metrics: Outputs accuracy, precision, recall, F1.

Model Saving: Stores the model and tokenizer for reuse.

🧪 Example Usage
Predict Sentiment
After training and saving the model:



test_review = "The food was bad!"

Example Output:


Sentiment: LABEL_0 (Negative) Score: 0.9426

📊 Final Evaluation Metrics
Metric	Value (%)
Accuracy	87.01
F1	90.56
Precision	90.43
Recall	90.69

💾 Model Export
The trained model and tokenizer are saved to:

/content/drive/My Drive/my_trained_model
You can reload for inference later.

🚀 Notes
If you change dataset paths, update the pd.read_csv() calls.

The model requires training before predictions.

You can increase num_train_epochs or adjust learning_rate for further tuning.

📘 References
Hugging Face Transformers

PyTorch

scikit-learn Metrics
