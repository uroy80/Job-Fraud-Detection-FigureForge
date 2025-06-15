#!/usr/bin/env python3
"""
Extract Real Model Performance Metrics

This script loads your trained model and calculates actual performance metrics
by evaluating it on real test data.
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

def load_model_and_preprocessors():
    """Load the trained model and preprocessors."""
    try:
        # Load enhanced model
        with open('enhanced_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load vectorizer
        with open('enhanced_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load scaler
        with open('enhanced_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load feature names
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        print("✅ Successfully loaded trained model and preprocessors")
        return model, vectorizer, scaler, feature_names
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None, None

def advanced_text_preprocessing(text):
    """Advanced text preprocessing (same as training)."""
    if not isinstance(text, str):
        return ""
    
    import re
    
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
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep some punctuation for context
    text = re.sub(r'[^a-zA-Z\s!?$]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Remove stopwords but keep important words
    words = [word for word in words if word not in STOPWORDS and len(word) > 2]
    
    return ' '.join(words)

def extract_features_for_evaluation(df):
    """Extract the same features used during training."""
    print("Extracting features for evaluation...")
    
    # Fraud indicator keywords
    FRAUD_KEYWORDS = {
        'urgent', 'immediate', 'asap', 'quick', 'fast', 'easy', 'guaranteed', 'no experience',
        'work from home', 'make money', 'earn money', 'cash', 'payment upfront', 'wire transfer',
        'western union', 'moneygram', 'bitcoin', 'cryptocurrency', 'investment', 'pyramid',
        'mlm', 'multi level', 'network marketing', 'get rich', 'financial freedom',
        'limited time', 'act now', 'hurry', 'exclusive', 'secret', 'confidential'
    }
    
    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')
        else:
            df[col] = df[col].fillna(0)
    
    # Text preprocessing
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
    
    # Extract all the same features as training
    import re
    from sklearn.preprocessing import LabelEncoder
    
    # Text features
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
    
    # Department and function encoding
    if 'department' in df.columns:
        le_dept = LabelEncoder()
        df['department_encoded'] = le_dept.fit_transform(df['department'].astype(str))
    else:
        df['department_encoded'] = 0
    
    if 'function' in df.columns:
        le_func = LabelEncoder()
        df['function_encoded'] = le_func.fit_transform(df['function'].astype(str))
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
    
    # Boolean features
    boolean_columns = ['telecommuting', 'has_company_logo', 'has_questions']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
        else:
            df[col] = 0
    
    return df

def evaluate_model_performance():
    """Evaluate the model and extract real performance metrics."""
    print("🔍 Extracting REAL performance metrics from your trained model...")
    print("=" * 70)
    
    # Load model and preprocessors
    model, vectorizer, scaler, feature_names = load_model_and_preprocessors()
    if model is None:
        return False
    
    # Load the original training data
    try:
        print("📂 Loading training data...")
        df = pd.read_csv('training_data.csv')
        print(f"✅ Loaded {len(df)} samples")
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return False
    
    # Extract features (same as training)
    df = extract_features_for_evaluation(df)
    
    # Prepare features for evaluation
    X_text = df['combined_text']
    
    # Get numerical features that exist
    all_numerical_features = [
        'title_length', 'description_length', 'title_word_count', 'description_word_count',
        'fraud_keywords_count', 'has_fraud_keywords', 'urgency_score', 'money_mentions',
        'has_email', 'has_phone', 'has_website', 'requires_experience', 'requires_education',
        'location_length', 'is_remote', 'company_profile_length', 'has_company_description',
        'has_salary', 'salary_length', 'department_encoded', 'function_encoded',
        'is_full_time', 'is_part_time', 'is_contract', 'telecommuting', 
        'has_company_logo', 'has_questions'
    ]
    
    # Filter features that exist in the dataframe
    numerical_features = [f for f in all_numerical_features if f in df.columns]
    X_num = df[numerical_features]
    y = df['fraudulent'].astype(int)
    
    print(f"📊 Dataset info:")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Fraudulent: {y.sum()} ({y.mean()*100:.2f}%)")
    print(f"   - Legitimate: {len(y) - y.sum()} ({(1-y.mean())*100:.2f}%)")
    print(f"   - Numerical features: {len(numerical_features)}")
    
    # Split data (same as training)
    from sklearn.model_selection import train_test_split
    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Split info:")
    print(f"   - Training samples: {len(X_text_train)}")
    print(f"   - Test samples: {len(X_text_test)}")
    
    # Transform features using saved preprocessors
    print("🔄 Transforming features...")
    X_text_test_vec = vectorizer.transform(X_text_test)
    X_num_test_scaled = scaler.transform(X_num_test)
    
    # Combine features
    X_test_combined = np.hstack((X_text_test_vec.toarray(), X_num_test_scaled))
    
    print(f"✅ Feature matrix shape: {X_test_combined.shape}")
    
    # Make predictions
    print("🎯 Making predictions...")
    y_pred = model.predict(X_test_combined)
    y_pred_proba = model.predict_proba(X_test_combined)[:, 1]
    
    # Calculate all metrics
    print("📈 Calculating performance metrics...")
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation on training data
    print("🔄 Performing cross-validation...")
    X_text_train_vec = vectorizer.transform(X_text_train)
    X_num_train_scaled = scaler.transform(X_num_train)
    X_train_combined = np.hstack((X_text_train_vec.toarray(), X_num_train_scaled))
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_combined, y_train, cv=cv, scoring='f1')
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Create comprehensive metrics dictionary
    real_metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_score': float(auc),
        'cv_f1_mean': float(cv_scores.mean()),
        'cv_f1_std': float(cv_scores.std()),
        'training_samples': int(len(X_text_train)),
        'test_samples': int(len(X_text_test)),
        'feature_count': int(X_test_combined.shape[1]),
        'model_type': 'Enhanced Model (Real Evaluation)',
        'evaluation_date': datetime.now().isoformat(),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'class_distribution': {
            'total_samples': int(len(y)),
            'fraudulent_samples': int(y.sum()),
            'legitimate_samples': int(len(y) - y.sum()),
            'fraud_rate': float(y.mean())
        },
        'cv_scores_individual': cv_scores.tolist(),
        'test_class_distribution': {
            'fraudulent_test': int(y_test.sum()),
            'legitimate_test': int(len(y_test) - y_test.sum())
        }
    }
    
    # Save the real metrics
    with open('model_performance.pkl', 'wb') as f:
        pickle.dump(real_metrics, f)
    
    # Display results
    print("\n" + "=" * 70)
    print("🎉 REAL MODEL PERFORMANCE METRICS EXTRACTED!")
    print("=" * 70)
    print(f"🎯 Accuracy: {accuracy:.1%}")
    print(f"🎯 Precision: {precision:.1%}")
    print(f"🎯 Recall: {recall:.1%}")
    print(f"🎯 F1 Score: {f1:.1%}")
    print(f"🎯 AUC Score: {auc:.1%}")
    print(f"🎯 CV F1: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    print(f"📊 Training Samples: {len(X_text_train):,}")
    print(f"📊 Test Samples: {len(X_text_test):,}")
    print(f"📊 Total Features: {X_test_combined.shape[1]:,}")
    print(f"📊 Text Features: {X_text_test_vec.shape[1]:,}")
    print(f"📊 Numerical Features: {len(numerical_features):,}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"   True Negatives (Correct Legitimate): {tn:,}")
    print(f"   False Positives (Wrong Fraud Alert): {fp:,}")
    print(f"   False Negatives (Missed Fraud): {fn:,}")
    print(f"   True Positives (Caught Fraud): {tp:,}")
    
    print(f"\n📊 Test Set Breakdown:")
    print(f"   Fraudulent jobs in test: {y_test.sum():,}")
    print(f"   Legitimate jobs in test: {len(y_test) - y_test.sum():,}")
    
    print(f"\n🔄 Cross-Validation Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"   Fold {i}: {score:.1%}")
    
    print("\n✅ Real metrics saved to model_performance.pkl")
    print("🔄 Refresh your dashboard to see the actual performance!")
    
    return True

if __name__ == "__main__":
    success = evaluate_model_performance()
    if success:
        print("\n🎉 SUCCESS! Your dashboard will now show REAL performance metrics!")
    else:
        print("\n❌ Failed to extract real metrics. Please check the errors above.")
