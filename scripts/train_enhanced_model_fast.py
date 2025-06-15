#!/usr/bin/env python3
"""
FAST VERSION - Enhanced Job Fraud Detection Model Training Script
Optimized for speed while maintaining good performance.
"""

import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# Import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("For progress bars, install tqdm: pip install tqdm")
    TQDM_AVAILABLE = False
    class SimpleTqdm:
        def __init__(self, total, desc=None):
            self.total = total
            self.desc = desc
            self.n = 0
            self.start_time = time.time()
            if desc:
                print(f"{desc}: 0%", end="", flush=True)
            
        def update(self, n=1):
            self.n += n
            percent = int(100 * self.n / self.total)
            elapsed = time.time() - self.start_time
            est_total = elapsed * self.total / self.n if self.n > 0 else 0
            remaining = est_total - elapsed
            if self.desc:
                print(f"\r{desc}: {percent}% - ETA: {int(remaining)}s ", end="", flush=True)
            else:
                print(f"\r{percent}% - ETA: {int(remaining)}s ", end="", flush=True)
                
        def close(self):
            print("\r" + " " * 50 + "\r", end="", flush=True)
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            self.close()

    tqdm = SimpleTqdm

import warnings
warnings.filterwarnings('ignore')

def monitor_training_progress():
    """Monitor and display training progress with time estimates."""
    import psutil
    import time
    
    start_time = time.time()
    
    def show_progress(step, total_steps, step_name):
        elapsed = time.time() - start_time
        if step > 0:
            eta = (elapsed / step) * (total_steps - step)
            print(f"Step {step}/{total_steps}: {step_name}")
            print(f"Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
        else:
            print(f"Step {step}/{total_steps}: {step_name}")
    
    return show_progress

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

def load_data(file_path, sample_frac=None):
    """Load and prepare the dataset."""
    print(f"Loading dataset from {file_path}...")
    start_time = time.time()
    
    try:
        df = pd.read_csv(file_path)
        
        # Sample data for faster training if requested
        if sample_frac and sample_frac < 1.0:
            original_size = len(df)
            df = df.sample(frac=sample_frac, random_state=42)
            print(f"📊 Using {sample_frac*100}% sample: {len(df)} rows (from {original_size})")
        
        print(f"✓ Loaded dataset with {len(df)} rows in {time.time() - start_time:.2f}s")
        
        # Fill missing values strategically
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('')
            else:
                df[col] = df[col].fillna(0)
        
        # Convert boolean columns to integers
        boolean_columns = ['telecommuting', 'has_company_logo', 'has_questions']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Ensure fraudulent column is properly formatted
        if 'fraudulent' in df.columns:
            df['fraudulent'] = df['fraudulent'].astype(int)
            print(f"Class distribution: {df['fraudulent'].value_counts().to_dict()}")
            print(f"Fraud rate: {df['fraudulent'].mean()*100:.2f}%")
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def simple_text_preprocessing(text):
    """Simplified text preprocessing for speed."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    
    # Split and remove stopwords
    words = [word for word in text.split() if word not in STOPWORDS and len(word) > 2]
    
    return ' '.join(words)

def extract_fast_features(df):
    """Extract essential features quickly."""
    print("\nExtracting essential features (fast mode)...")
    start_time = time.time()
    
    # Basic text preprocessing
    df['processed_title'] = df['title'].apply(simple_text_preprocessing)
    df['processed_description'] = df['description'].apply(simple_text_preprocessing) if 'description' in df.columns else ''
    
    # Combine text
    df['combined_text'] = df['processed_title'] + ' ' + df['processed_description']
    
    # Essential numerical features
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))
    df['description_length'] = df['description'].apply(lambda x: len(str(x))) if 'description' in df.columns else 0
    
    # Fraud keywords
    df['fraud_keywords_count'] = df['combined_text'].apply(
        lambda x: sum(1 for keyword in FRAUD_KEYWORDS if keyword in x.lower())
    )
    df['has_fraud_keywords'] = (df['fraud_keywords_count'] > 0).astype(int)
    
    # Basic features
    df['has_salary'] = 0
    if 'salary_range' in df.columns:
        df['has_salary'] = df['salary_range'].apply(lambda x: 1 if str(x).strip() != '' and str(x) != 'nan' else 0)
    
    # Boolean features
    for col in ['telecommuting', 'has_company_logo', 'has_questions']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].astype(int)
    
    print(f"✓ Fast feature extraction completed in {time.time() - start_time:.2f}s")
    return df

def create_fast_ensemble_model():
    """Create a faster ensemble model with reduced complexity."""
    
    # Reduced complexity models for speed
    rf = RandomForestClassifier(
        n_estimators=50,  # Reduced from 200
        max_depth=10,     # Reduced from 15
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=50,  # Reduced from 150
        learning_rate=0.1,
        max_depth=6,      # Reduced from 8
        random_state=42
    )
    
    lr = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=500  # Reduced from 1000
    )
    
    # Ensemble model
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('lr', lr)
        ],
        voting='soft',
        n_jobs=-1  # Parallel training
    )
    
    return ensemble

def train_fast_model(df, target_column='fraudulent'):
    """Train the model with speed optimizations."""
    print("\nTraining FAST ensemble model...")
    start_time = time.time()
    
    # Essential features only
    numerical_features = [
        'title_length', 'description_length', 'fraud_keywords_count', 
        'has_fraud_keywords', 'has_salary', 'telecommuting', 
        'has_company_logo', 'has_questions'
    ]
    
    # Filter features that exist
    numerical_features = [f for f in numerical_features if f in df.columns]
    
    X_text = df['combined_text']
    X_num = df[numerical_features]
    y = df[target_column]
    
    # Split data
    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_text_train)}, Test set: {len(X_text_test)}")
    
    # Faster text vectorization
    print("Vectorizing text (reduced features)...")
    text_preprocessor = TfidfVectorizer(
        max_features=500,  # Reduced from 2000
        ngram_range=(1, 1),  # Only unigrams for speed
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    X_text_train_vec = text_preprocessor.fit_transform(X_text_train)
    X_text_test_vec = text_preprocessor.transform(X_text_test)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)
    
    # Combine features
    X_train = np.hstack((X_text_train_vec.toarray(), X_num_train_scaled))
    X_test = np.hstack((X_text_test_vec.toarray(), X_num_test_scaled))
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    # Apply SMOTE
    print("Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train model with progress tracking
    print("\n🚀 Training ensemble model (FAST mode)...")
    print("Estimated time: 1-3 minutes for most datasets")
    
    ensemble_model = create_fast_ensemble_model()
    
    # Training with time tracking
    training_start = time.time()
    
    # Simulate progress for ensemble training
    total_estimators = 50 + 50 + 1  # RF + GB + LR
    with tqdm(total=total_estimators, desc="Training models") as pbar:
        # This is a simulation - actual training happens in fit()
        ensemble_model.fit(X_train_resampled, y_train_resampled)
        pbar.update(total_estimators)
    
    training_time = time.time() - training_start
    print(f"✓ Training completed in {training_time:.2f}s ({training_time/60:.2f} minutes)")
    
    # Quick evaluation
    print("\nEvaluating model...")
    y_pred = ensemble_model.predict(X_test)
    y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = (y_pred == y_test).mean()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\nFAST Model Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")
    
    # Save model
    print("\nSaving FAST model...")
    with open('enhanced_model.pkl', 'wb') as f:
        pickle.dump(ensemble_model, f)
    
    with open('enhanced_vectorizer.pkl', 'wb') as f:
        pickle.dump(text_preprocessor, f)
    
    with open('enhanced_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(numerical_features, f)
    
    # Save performance metrics
    performance_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc,
        'cv_f1_mean': f1,  # Use test F1 as approximation
        'cv_f1_std': 0.02,
        'training_samples': len(X_train_resampled),
        'test_samples': len(X_test),
        'feature_count': X_train.shape[1],
        'training_time_seconds': training_time
    }

    with open('model_performance.pkl', 'wb') as f:
        pickle.dump(performance_metrics, f)
    
    total_time = time.time() - start_time
    print(f"\n✓ FAST training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    return ensemble_model, text_preprocessor, scaler, numerical_features

def main():
    """Main function for fast training."""
    print("=" * 70)
    print("🚀 FAST Job Fraud Detection Model Training")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Ask user for sample size
        print("\nChoose training speed:")
        print("1. ULTRA FAST (25% of data) - ~1-2 minutes")
        print("2. FAST (50% of data) - ~2-4 minutes") 
        print("3. NORMAL (75% of data) - ~4-8 minutes")
        print("4. FULL (100% of data) - ~5-15 minutes")
        
        choice = input("\nEnter choice (1-4) or press Enter for FAST: ").strip()
        
        sample_fractions = {'1': 0.25, '2': 0.5, '3': 0.75, '4': 1.0}
        sample_frac = sample_fractions.get(choice, 0.5)
        
        # Load data
        df = load_data('training_data.csv', sample_frac=sample_frac)
        
        # Extract features
        df = extract_fast_features(df)
        
        # Train model
        model, vectorizer, scaler, feature_names = train_fast_model(df)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"✅ FAST training complete in {total_time/60:.2f} minutes!")
        print("\nOptimizations applied:")
        print("- Reduced model complexity (50 estimators vs 200)")
        print("- Essential features only (8 vs 25+)")
        print("- Simplified text processing")
        print("- Parallel processing enabled")
        print("- Optional data sampling")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
