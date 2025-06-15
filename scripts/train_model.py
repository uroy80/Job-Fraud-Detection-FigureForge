#!/usr/bin/env python3
"""
Job Fraud Detection Model Training Script

This script trains a machine learning model to detect fraudulent job postings.
"""

import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE

# Download NLTK resources
print("Downloading required NLTK resources...")
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')  # Open Multilingual WordNet
except Exception as e:
    print(f"Warning: Error downloading NLTK resources: {e}")
    print("You may need to run scripts/download_nltk_data.py first")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def load_data(file_path):
    """Load and prepare the dataset."""
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with {len(df)} rows")
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')
        else:
            df[col] = df[col].fillna(0)
    
    # Convert boolean columns to integers if needed
    boolean_columns = ['telecommuting', 'has_company_logo', 'has_questions']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Ensure fraudulent column is properly formatted
    if 'fraudulent' in df.columns:
        df['fraudulent'] = df['fraudulent'].astype(int)
    
    return df

def preprocess_text(text):
    """Preprocess text data for model input."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    try:
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
        return ' '.join(tokens)
    except LookupError:
        # Fallback if tokenization fails
        print("Warning: NLTK tokenization failed. Using simple split instead.")
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)

def extract_features(df):
    """Extract features from job listings."""
    print("\nExtracting features...")
    
    # Text features
    df['processed_title'] = df['title'].apply(preprocess_text)
    df['processed_description'] = df['description'].apply(preprocess_text) if 'description' in df.columns else ''
    df['processed_requirements'] = df['requirements'].apply(preprocess_text) if 'requirements' in df.columns else ''
    df['processed_company_profile'] = df['company_profile'].apply(preprocess_text) if 'company_profile' in df.columns else ''
    df['processed_benefits'] = df['benefits'].apply(preprocess_text) if 'benefits' in df.columns else ''
    
    # Combine text features
    df['combined_text'] = df['processed_title'] + ' ' + df['processed_description'] + ' ' + \
                         df['processed_requirements'] + ' ' + df['processed_company_profile'] + ' ' + \
                         df['processed_benefits']
    
    # Extract numerical features
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))
    df['description_length'] = df['description'].apply(lambda x: len(str(x))) if 'description' in df.columns else 0
    df['has_salary'] = df.apply(lambda x: 1 if 'salary_range' in df.columns and str(x['salary_range']).strip() != '' else 0, axis=1)
    df['has_requirements'] = df.apply(lambda x: 1 if 'requirements' in df.columns and str(x['requirements']).strip() != '' else 0, axis=1)
    df['has_benefits'] = df.apply(lambda x: 1 if 'benefits' in df.columns and str(x['benefits']).strip() != '' else 0, axis=1)
    
    # Use existing boolean features
    if 'telecommuting' in df.columns:
        df['telecommuting'] = df['telecommuting'].astype(int)
    if 'has_company_logo' in df.columns:
        df['has_company_logo'] = df['has_company_logo'].astype(int)
    if 'has_questions' in df.columns:
        df['has_questions'] = df['has_questions'].astype(int)
    
    return df

def train_model(df, target_column='fraudulent'):
    """Train the fraud detection model."""
    print("\nTraining model...")
    
    # Split features and target
    X_text = df['combined_text']
    y = df[target_column]
    
    # Get numerical features - update with your dataset's columns
    numerical_features = ['title_length', 'description_length', 'has_salary', 
                         'has_requirements', 'has_benefits']
    
    # Add boolean columns if they exist
    if 'telecommuting' in df.columns:
        numerical_features.append('telecommuting')
    if 'has_company_logo' in df.columns:
        numerical_features.append('has_company_logo')
    if 'has_questions' in df.columns:
        numerical_features.append('has_questions')
    
    X_num = df[numerical_features]
    
    # Split data
    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF Vectorization for text
    vectorizer = TfidfVectorizer(max_features=1000)
    X_text_train_vec = vectorizer.fit_transform(X_text_train)
    X_text_test_vec = vectorizer.transform(X_text_test)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)
    
    # Combine features
    X_train = np.hstack((X_text_train_vec.toarray(), X_num_train_scaled))
    X_test = np.hstack((X_text_test_vec.toarray(), X_num_test_scaled))
    
    # Check class distribution
    print("\nClass distribution:")
    print(y_train.value_counts())
    
    # Apply SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("After SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nF1 Score:", f1_score(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Save model and vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nModel, vectorizer, and scaler saved.")
    
    return model, vectorizer, scaler

def main():
    """Main function to run the training pipeline."""
    # Load data
    df = load_data('training_data.csv')
    
    # Extract features
    df = extract_features(df)
    
    # Train model
    model, vectorizer, scaler = train_model(df)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
