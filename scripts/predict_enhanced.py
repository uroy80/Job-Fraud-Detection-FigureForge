#!/usr/bin/env python3
"""
Enhanced Job Fraud Detection Prediction Script

This script uses the enhanced model with advanced features for better accuracy.
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.preprocessing import LabelEncoder

# Simple stopwords list
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'i', 'you', 'we', 'they', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall',
    'but', 'or', 'if', 'because', 'as', 'until', 'while', 'when', 'where', 'why',
    'how', 'what', 'which', 'who', 'whom', 'whose', 'whether', 'not', 'no', 'nor',
    'so', 'than', 'too', 'very', 'just', 'now', 'then', 'here', 'there', 'up',
    'down', 'out', 'off', 'over', 'under', 'again', 'further', 'once'
}

# Fraud indicator keywords
FRAUD_KEYWORDS = {
    'urgent', 'immediate', 'asap', 'quick', 'fast', 'easy', 'guaranteed', 'no experience',
    'work from home', 'make money', 'earn money', 'cash', 'payment upfront', 'wire transfer',
    'western union', 'moneygram', 'bitcoin', 'cryptocurrency', 'investment', 'pyramid',
    'mlm', 'multi level', 'network marketing', 'get rich', 'financial freedom',
    'limited time', 'act now', 'hurry', 'exclusive', 'secret', 'confidential'
}

def advanced_text_preprocessing(text):
    """Advanced text preprocessing with fraud-specific features."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep some punctuation for context
    text = re.sub(r'[^a-zA-Z\s!?$]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Remove stopwords but keep important words
    words = [word for word in words if word not in STOPWORDS and len(word) > 2]
    
    return ' '.join(words)

def extract_company_name(company_profile):
    """Extract company name from company_profile field."""
    if not isinstance(company_profile, str) or company_profile.strip() == '':
        return "Not Available"
    
    # Clean the company profile text
    company_profile = company_profile.strip()
    
    # Try to extract company name from the beginning of the profile
    patterns = [
        r'^([A-Za-z0-9\s&.,\-]+?)(?:\s+is\s+|\s+was\s+|\s+has\s+|\s+provides\s+|\s+offers\s+)',
        r'^(?:At\s+|About\s+)?([A-Za-z0-9\s&.,\-]+?)(?:\s*[,:]|\s+we\s+|\s+our\s+)',
        r'^([A-Za-z0-9\s&.,\-]{2,50}?)(?:\s*\n|\s*\r)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, company_profile, re.IGNORECASE)
        if match:
            company_name = match.group(1).strip()
            company_name = re.sub(r'\s+', ' ', company_name)
            if len(company_name) > 3 and len(company_name) < 100:
                return company_name
    
    # If no pattern matches, take the first 50 characters
    first_part = company_profile[:50].strip()
    if len(first_part) > 3:
        words = first_part.split()
        if len(words) > 1:
            return ' '.join(words[:-1]) if len(' '.join(words[:-1])) > 3 else first_part
        return first_part
    
    return "Not Available"

def extract_advanced_features(df):
    """Extract the same advanced features used in training."""
    print("Extracting advanced features...")
    
    # Basic text preprocessing
    df['processed_title'] = df['title'].apply(advanced_text_preprocessing)
    df['processed_description'] = df['description'].apply(advanced_text_preprocessing) if 'description' in df.columns else ''
    df['processed_requirements'] = df['requirements'].apply(advanced_text_preprocessing) if 'requirements' in df.columns else ''
    df['processed_company_profile'] = df['company_profile'].apply(advanced_text_preprocessing) if 'company_profile' in df.columns else ''
    df['processed_benefits'] = df['benefits'].apply(advanced_text_preprocessing) if 'benefits' in df.columns else ''
    
    # Combine all text
    df['combined_text'] = (
        df['processed_title'] + ' ' + 
        df['processed_description'] + ' ' + 
        df['processed_requirements'] + ' ' + 
        df['processed_company_profile'] + ' ' + 
        df['processed_benefits']
    )
    
    # Advanced text features
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))
    df['description_length'] = df['description'].apply(lambda x: len(str(x))) if 'description' in df.columns else 0
    df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))
    df['description_word_count'] = df['description'].apply(lambda x: len(str(x).split())) if 'description' in df.columns else 0
    
    # Fraud keyword features
    df['fraud_keywords_count'] = df['combined_text'].apply(
        lambda x: sum(1 for keyword in FRAUD_KEYWORDS if keyword in x.lower())
    )
    df['has_fraud_keywords'] = (df['fraud_keywords_count'] > 0).astype(int)
    
    # Urgency indicators
    urgency_words = ['urgent', 'immediate', 'asap', 'hurry', 'quick', 'fast']
    df['urgency_score'] = df['combined_text'].apply(
        lambda x: sum(1 for word in urgency_words if word in x.lower())
    )
    
    # Money-related features
    money_patterns = [r'\$\d+', r'salary', r'pay', r'wage', r'income', r'earn']
    df['money_mentions'] = df['combined_text'].apply(
        lambda x: sum(1 for pattern in money_patterns if re.search(pattern, x.lower()))
    )
    
    # Contact information features
    df['has_email'] = df['combined_text'].apply(lambda x: 1 if '@' in x else 0)
    df['has_phone'] = df['combined_text'].apply(
        lambda x: 1 if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', x) else 0
    )
    df['has_website'] = df['combined_text'].apply(
        lambda x: 1 if re.search(r'www\.|http|\.com|\.org', x.lower()) else 0
    )
    
    # Experience and education features
    df['requires_experience'] = df['combined_text'].apply(
        lambda x: 1 if any(word in x.lower() for word in ['experience', 'years', 'background']) else 0
    )
    df['requires_education'] = df['combined_text'].apply(
        lambda x: 1 if any(word in x.lower() for word in ['degree', 'education', 'bachelor', 'master', 'phd']) else 0
    )
    
    # Location features
    if 'location' in df.columns:
        df['location_length'] = df['location'].apply(lambda x: len(str(x)))
        df['is_remote'] = df['location'].apply(
            lambda x: 1 if any(word in str(x).lower() for word in ['remote', 'anywhere', 'home']) else 0
        )
    else:
        df['location_length'] = 0
        df['is_remote'] = 0
    
    # Company features
    if 'company_profile' in df.columns:
        df['company_profile_length'] = df['company_profile'].apply(lambda x: len(str(x)))
        df['has_company_description'] = (df['company_profile_length'] > 50).astype(int)
    else:
        df['company_profile_length'] = 0
        df['has_company_description'] = 0
    
    # Salary features
    if 'salary_range' in df.columns:
        df['has_salary'] = df['salary_range'].apply(lambda x: 1 if str(x).strip() != '' and str(x) != 'nan' else 0)
        df['salary_length'] = df['salary_range'].apply(lambda x: len(str(x)) if str(x) != 'nan' else 0)
    else:
        df['has_salary'] = 0
        df['salary_length'] = 0
    
    # Department and function encoding (simplified for prediction)
    if 'department' in df.columns:
        df['department_encoded'] = df['department'].apply(lambda x: hash(str(x)) % 1000)
    else:
        df['department_encoded'] = 0
    
    if 'function' in df.columns:
        df['function_encoded'] = df['function'].apply(lambda x: hash(str(x)) % 1000)
    else:
        df['function_encoded'] = 0
    
    # Employment type features
    if 'employment_type' in df.columns:
        df['is_full_time'] = df['employment_type'].apply(
            lambda x: 1 if 'full' in str(x).lower() else 0
        )
        df['is_part_time'] = df['employment_type'].apply(
            lambda x: 1 if 'part' in str(x).lower() else 0
        )
        df['is_contract'] = df['employment_type'].apply(
            lambda x: 1 if 'contract' in str(x).lower() else 0
        )
    else:
        df['is_full_time'] = 0
        df['is_part_time'] = 0
        df['is_contract'] = 0
    
    # Existing boolean features
    for col in ['telecommuting', 'has_company_logo', 'has_questions']:
        if col in df.columns:
            df[col] = df[col].astype(int)
        else:
            df[col] = 0
    
    return df

def load_enhanced_model():
    """Load the enhanced model and preprocessors."""
    try:
        with open('enhanced_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('enhanced_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            
        with open('enhanced_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            
        print("Loaded enhanced model and preprocessors")
        return model, vectorizer, scaler, feature_names, True
    except Exception as e:
        print(f"Could not load enhanced model: {e}")
        print("Falling back to basic model...")
        
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
                
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
                
            # Basic feature names
            feature_names = [
                'title_length', 'description_length', 'has_salary', 
                'has_requirements', 'has_benefits', 'telecommuting', 
                'has_company_logo', 'has_questions'
            ]
            
            print("Loaded basic model")
            return model, vectorizer, scaler, feature_names, True
        except Exception as e2:
            print(f"Could not load any model: {e2}")
            return None, None, None, None, False

def predict_enhanced(input_file, output_file):
    """Make predictions using the enhanced model."""
    print(f"Processing input file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    try:
        # Load the data
        df = pd.read_csv(input_file)
        print(f"Loaded data with {len(df)} rows")
        
        # Ensure required columns exist
        required_columns = ['title']
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Required column '{col}' not found in input file")
                return False
        
        # Add missing columns if needed
        optional_columns = [
            'description', 'company_profile', 'requirements', 'benefits', 
            'company', 'location', 'telecommuting', 'has_company_logo', 'has_questions',
            'department', 'function', 'employment_type', 'salary_range'
        ]
        for col in optional_columns:
            if col not in df.columns:
                df[col] = ""
        
        # Extract advanced features
        df = extract_advanced_features(df)
        
        # Load enhanced model
        model, vectorizer, scaler, feature_names, model_loaded = load_enhanced_model()
        
        if not model_loaded:
            print("No model could be loaded. Please train a model first.")
            return False
        
        # Prepare features
        print("Preparing features for prediction...")
        X_text = vectorizer.transform(df['combined_text'])
        
        # Get numerical features that exist in the dataframe
        available_features = [f for f in feature_names if f in df.columns]
        missing_features = [f for f in feature_names if f not in df.columns]
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                df[feature] = 0
        
        X_num = df[feature_names]
        
        # Scale numerical features
        print("Scaling numerical features...")
        X_num_scaled = scaler.transform(X_num)
        
        # Combine features
        X = np.hstack((X_text.toarray(), X_num_scaled))
        
        # Make predictions
        print("Making predictions...")
        fraud_probs = model.predict_proba(X)[:, 1]
        
        # Add predictions to the dataframe
        df['fraud_probability'] = fraud_probs
        df['prediction'] = df['fraud_probability'].apply(lambda x: 'fraudulent' if x > 0.5 else 'genuine')
        
        # Handle job_id and company extraction
        if 'job_id' in df.columns:
            df['id'] = df['job_id']
        elif 'id' not in df.columns:
            df['id'] = [f"job_{i}" for i in range(len(df))]
        
        # Extract company name from company_profile
        if 'company_profile' in df.columns:
            df['company'] = df['company_profile'].apply(extract_company_name)
        elif 'company' not in df.columns:
            df['company'] = "Not Available"
        
        # Ensure location column exists
        if 'location' not in df.columns:
            df['location'] = "Not Available"
        
        # Select columns for output
        output_columns = ['id', 'title', 'company', 'location', 'fraud_probability', 'prediction']
        
        # Add job_id if it exists
        if 'job_id' in df.columns:
            output_columns.insert(1, 'job_id')
        
        output_df = df[output_columns]
        
        # Save to output file
        output_df.to_csv(output_file, index=False)
        
        print(f"Predictions saved to {output_file}")
        print(f"Processed {len(df)} job listings")
        print(f"Found {len(df[df['prediction'] == 'fraudulent'])} potentially fraudulent jobs")
        print(f"Average fraud probability: {fraud_probs.mean():.3f}")
        
        return True
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_enhanced.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = predict_enhanced(input_file, output_file)
    if not success:
        sys.exit(1)
