#!/usr/bin/env python3
"""
Ultra Enhanced Job Fraud Detection Prediction Script

This script uses the ultra-enhanced model with all advanced features.
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.preprocessing import LabelEncoder

# Import the same preprocessing functions from training
from train_ultra_enhanced_model import (
    advanced_text_preprocessing, 
    extract_ultra_advanced_features,
    FRAUD_KEYWORDS,
    LEGITIMATE_KEYWORDS
)

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

def load_ultra_enhanced_model():
    """Load the ultra-enhanced model and all preprocessors."""
    try:
        # Load main model
        with open('enhanced_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessors
        with open('enhanced_vectorizer.pkl', 'rb') as f:
            preprocessors = pickle.load(f)
            
        with open('enhanced_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            
        print("✓ Loaded ultra-enhanced model and all preprocessors")
        return model, preprocessors, scaler, feature_names, True
        
    except Exception as e:
        print(f"Could not load ultra-enhanced model: {e}")
        print("Falling back to enhanced model...")
        
        try:
            with open('enhanced_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            with open('enhanced_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
                
            with open('enhanced_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
                
            with open('feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
                
            # Create simple preprocessors dict for compatibility
            preprocessors = {'tfidf_word': vectorizer}
            
            print("✓ Loaded enhanced model")
            return model, preprocessors, scaler, feature_names, True
            
        except Exception as e2:
            print(f"Could not load any enhanced model: {e2}")
            return None, None, None, None, False

def predict_ultra_enhanced(input_file, output_file):
    """Make predictions using the ultra-enhanced model."""
    print(f"🚀 Processing with ULTRA-ENHANCED model...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    try:
        # Load the data
        df = pd.read_csv(input_file)
        print(f"✓ Loaded data with {len(df)} rows")
        
        # Ensure required columns exist
        required_columns = ['title']
        for col in required_columns:
            if col not in df.columns:
                print(f"❌ Error: Required column '{col}' not found in input file")
                return False
        
        # Add missing columns if needed
        optional_columns = [
            'description', 'company_profile', 'requirements', 'benefits', 
            'company', 'location', 'telecommuting', 'has_company_logo', 'has_questions',
            'department', 'function', 'employment_type', 'salary_range',
            'required_experience', 'required_education'
        ]
        for col in optional_columns:
            if col not in df.columns:
                df[col] = ""
        
        # Extract ultra-advanced features
        print("🔧 Extracting ultra-advanced features...")
        df = extract_ultra_advanced_features(df)
        
        # Load ultra-enhanced model
        model, preprocessors, scaler, feature_names, model_loaded = load_ultra_enhanced_model()
        
        if not model_loaded:
            print("❌ No model could be loaded. Please train a model first.")
            return False
        
        # Prepare features based on model type
        print("🔧 Preparing features for prediction...")
        
        if 'tfidf_word' in preprocessors and 'tfidf_char' in preprocessors:
            # Ultra-enhanced model with multiple vectorizers
            print("   Using ultra-enhanced feature pipeline...")
            
            # Text features
            X_tfidf_word = preprocessors['tfidf_word'].transform(df['combined_text'])
            X_tfidf_char = preprocessors['tfidf_char'].transform(df['combined_text'])
            X_count = preprocessors['count_vec'].transform(df['combined_text'])
            
            # Numerical features
            available_features = [f for f in feature_names if f in df.columns]
            missing_features = [f for f in feature_names if f not in df.columns]
            
            if missing_features:
                print(f"   Warning: Missing {len(missing_features)} features, using defaults")
                for feature in missing_features:
                    df[feature] = 0
            
            X_num = df[feature_names]
            X_num_scaled = scaler.transform(X_num)
            
            # Feature selection and dimensionality reduction
            X_num_selected = preprocessors['selector'].transform(X_num_scaled)
            X_tfidf_word_svd = preprocessors['svd'].transform(X_tfidf_word)
            
            # Combine all features
            X = np.hstack([
                X_tfidf_word_svd,
                X_tfidf_char.toarray(),
                X_count.toarray(),
                X_num_selected
            ])
            
        else:
            # Standard enhanced model
            print("   Using standard enhanced feature pipeline...")
            X_text = preprocessors['tfidf_word'].transform(df['combined_text'])
            
            # Get numerical features that exist in the dataframe
            available_features = [f for f in feature_names if f in df.columns]
            missing_features = [f for f in feature_names if f not in df.columns]
            
            if missing_features:
                print(f"   Warning: Missing {len(missing_features)} features, using defaults")
                for feature in missing_features:
                    df[feature] = 0
            
            X_num = df[feature_names]
            X_num_scaled = scaler.transform(X_num)
            
            # Combine features
            X = np.hstack((X_text.toarray(), X_num_scaled))
        
        print(f"✓ Feature matrix shape: {X.shape}")
        
        # Make predictions
        print("🎯 Making predictions...")
        fraud_probs = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)
        
        # Add predictions to the dataframe
        df['fraud_probability'] = fraud_probs
        df['prediction'] = np.where(predictions == 1, 'fraudulent', 'genuine')
        
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
        
        # Enhanced reporting
        fraud_count = len(df[df['prediction'] == 'fraudulent'])
        high_risk_count = len(df[df['fraud_probability'] > 0.8])
        medium_risk_count = len(df[(df['fraud_probability'] > 0.5) & (df['fraud_probability'] <= 0.8)])
        
        print(f"\n📊 ULTRA-ENHANCED PREDICTION RESULTS:")
        print(f"   Total job listings processed: {len(df)}")
        print(f"   Fraudulent predictions: {fraud_count} ({fraud_count/len(df)*100:.1f}%)")
        print(f"   High risk (>80%): {high_risk_count}")
        print(f"   Medium risk (50-80%): {medium_risk_count}")
        print(f"   Average fraud probability: {fraud_probs.mean():.3f}")
        print(f"   Max fraud probability: {fraud_probs.max():.3f}")
        print(f"   Min fraud probability: {fraud_probs.min():.3f}")
        print(f"✓ Predictions saved to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during ultra-enhanced prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_ultra_enhanced.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = predict_ultra_enhanced(input_file, output_file)
    if not success:
        sys.exit(1)
