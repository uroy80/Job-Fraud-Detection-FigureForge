# Job Fraud Detection System

A machine learning-powered system to detect fraudulent job postings and protect job seekers from scams.

## 🎯 Project Overview

This system uses advanced machine learning techniques to identify fraudulent job postings with high accuracy. Built for the **Anveshan Hackathon 2025**, our solution combines cutting-edge data science with an intuitive web interface to help job seekers avoid fraudulent opportunities.

### Problem Statement
Job fraud is a growing concern in the digital age, with millions of job seekers falling victim to fraudulent postings annually. These scams not only waste time and resources but can also lead to identity theft and financial loss.

### Our Solution
We developed an AI-powered system that:
- **Analyzes job postings** using 25+ engineered features
- **Predicts fraud probability** with 91% F1-score accuracy
- **Provides real-time analysis** through an interactive dashboard
- **Offers detailed insights** into fraud patterns and indicators

---

## 🚀 Key Features & Technologies Used

### 🤖 Machine Learning Pipeline
- **Ensemble Model**: Random Forest + Gradient Boosting + Logistic Regression
- **Advanced NLP**: TF-IDF vectorization with fraud-specific preprocessing
- **Feature Engineering**: 25+ custom features including text analysis, fraud indicators, and behavioral patterns
- **Imbalance Handling**: SMOTE oversampling with balanced class weights
- **Performance**: 91% F1-score, 89% precision, 93% recall

### 💻 Technology Stack
- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS
- **Backend**: Node.js, Python 3.8+, FastAPI integration
- **Machine Learning**: Scikit-learn, Pandas, NumPy, NLTK
- **Visualization**: Recharts, Matplotlib, Seaborn
- **UI Components**: Shadcn/ui, Lucide React icons
- **Deployment**: Vercel/Railway ready with Docker support

### 📊 Dashboard Features
- **File Upload**: Drag-and-drop CSV processing
- **Results Table**: Searchable, sortable predictions with pagination
- **Visualizations**: Fraud distribution histograms, pie charts, top suspicious listings
- **Model Insights**: Performance metrics, feature importance, confusion matrices
- **Responsive Design**: Mobile-first approach with dark/light themes

---

## 🔬 Data Science Methodology

### 1. Data Processing Pipeline

#### Data Collection & Preprocessing
```python
# Data cleaning and preprocessing steps
def preprocess_job_data(df):
    # Handle missing values
    df['description'] = df['description'].fillna('')
    df['company_profile'] = df['company_profile'].fillna('')
    
    # Text cleaning
    df['title_clean'] = df['title'].apply(clean_text)
    df['description_clean'] = df['description'].apply(clean_text)
    
    # Feature extraction
    df = extract_features(df)
    
    return df
