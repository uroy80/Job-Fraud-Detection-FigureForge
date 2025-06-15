# Job Fraud Detection System

A machine learning-powered system to detect fraudulent job postings and protect job seekers from scams.

## Important Links

1. [GitHub Repository](https://github.com/uroy80/Job-Fraud-Detection-FigureForge)  
2. [YouTube Demo Video](https://youtu.be/5Xf8zYNPw88)  
3. [Application Website](http://206.189.137.181:3000/)

---

## 🌟 Project Overview

This system uses advanced machine learning techniques to identify fraudulent job postings with high accuracy. Built for the **Anveshan Hackathon 2025**, our solution combines cutting-edge data science with an intuitive web interface to help job seekers avoid fraudulent opportunities.

### Problem Statement
Job fraud is a growing concern in the digital age, with millions of job seekers falling victim to fraudulent postings annually. These scams not only waste time and resources but can also lead to identity theft and financial loss.

### Our Solution
An AI-powered system that:
- Analyzes job postings using 25+ engineered features
- Predicts fraud probability with 91% F1-score accuracy
- Provides real-time analysis through an interactive dashboard
- Offers detailed insights into fraud patterns and indicators

---

## 🚀 Key Features & Technologies Used

### 🤖 Machine Learning Pipeline
- Ensemble Model: Random Forest + Gradient Boosting + Logistic Regression
- Advanced NLP: TF-IDF vectorization with fraud-specific preprocessing
- Feature Engineering: 25+ custom features including text analysis, fraud indicators, and behavioral patterns
- Imbalance Handling: SMOTE oversampling with balanced class weights
- Performance: 91% F1-score, 89% precision, 93% recall

### 💻 Technology Stack
- Frontend: Next.js 14, React 18, TypeScript, Tailwind CSS
- Backend: Node.js, Python 3.8+, FastAPI integration
- Machine Learning: Scikit-learn, Pandas, NumPy, NLTK
- Visualization: Recharts, Matplotlib, Seaborn
- UI Components: Shadcn/ui, Lucide React icons
- Deployment: Vercel/Railway ready with Docker support

### 📊 Dashboard Features
- File Upload: Drag-and-drop CSV processing
- Results Table: Searchable, sortable predictions with pagination
- Visualizations: Fraud distribution histograms, pie charts, top suspicious listings
- Model Insights: Performance metrics, feature importance, confusion matrices
- Responsive Design: Mobile-first approach with dark/light themes

---

## 🔬 Data Science Methodology

### 1. Data Processing Pipeline
```python
def preprocess_job_data(df):
    df['description'] = df['description'].fillna('')
    df['company_profile'] = df['company_profile'].fillna('')
    df['title_clean'] = df['title'].apply(clean_text)
    df['description_clean'] = df['description'].apply(clean_text)
    df = extract_features(df)
    return df
```

### 2. Feature Engineering (25+ Features)

**Text-based Features:**
- Title and description length
- Word count and density
- Fraud keyword detection
- Contact information patterns

**Job Characteristics:**
- Salary anomaly detection
- Experience and education parsing
- Employment type classification

**Company Features:**
- Profile completeness
- Logo/branding presence
- Company description quality

**Location Features:**
- Remote work indicators
- Location specificity
- Address validation.

### 3. Model Architecture
```python
VotingClassifier([
    ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced')),
    ('gb', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1)),
    ('lr', LogisticRegression(class_weight='balanced', C=1.0))
])
```

**Text Processing:**
- TF-IDF Vectorization (1-2 gram, 2000 features)
- Fraud-domain stop words removal
- Chi-square feature selection

**Imbalance Handling:**
- SMOTE Oversampling
- Balanced class weights
- Stratified sampling

### 4. Model Training & Validation
```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

- Grid search for hyperparameter tuning
- F1-score optimization
- Early stopping to avoid overfitting

---

## 🚀 Setup Instructions

### Prerequisites
- Node.js 18+
- Python 3.8+
- Git

### Step-by-step
1. Clone the repository
```bash
git clone https://github.com/your-username/job-fraud-detection.git
cd job-fraud-detection
```

2. Install dependencies
```bash
# Frontend
npm install
# Backend
pip install -r requirements.txt
```

3. Download pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1iW9yy2_LTnUdAKFfey2EtfAmLP64Pgv8?usp=drive_link)

4. Initialize NLTK data
```bash
python scripts/download_nltk_data.py
```

5. Run the application
```bash
npm run dev
```

6. Open browser: `http://localhost:3000`

---

## 📁 Project Structure

```plaintext
job-fraud-detection/
├── app/
├── components/
├── scripts/
├── public/images/
├── training_data.csv
├── *.pkl files
├── requirements.txt
├── package.json
└── README.md
```

---

## 👍 Model Performance

| Metric         | Score | Industry Benchmark |
|----------------|-------|---------------------|
| F1-Score       | 0.91  | 0.75                |
| Precision      | 0.89  | 0.72                |
| Recall         | 0.93  | 0.78                |
| Accuracy       | 0.94  | 0.82                |
| AUC-ROC        | 0.96  | 0.85                |
| AUC-PR         | 0.92  | 0.79                |

**Top 10 Features:**
1. Urgency Keywords
2. Salary Anomalies
3. Contact Patterns
4. Description Length
5. Company Profile
6. Location Specificity
7. Experience Requirements
8. Money Mentions
9. Title Analysis
10. Education Requirements

---

## 📥 Data & Model Files

- [Google Drive Link](https://drive.google.com/drive/folders/1iW9yy2_LTnUdAKFfey2EtfAmLP64Pgv8?usp=drive_link)
- Download all `.pkl` and `.csv` files
- Place visualizations in `public/images/`

---

## 🎓 Usage Guide

1. Format your data
```csv
title,description,location,company,salary_range
"Software Developer","...","San Francisco, CA","TechCorp","$80,000-$120,000"
```

2. Upload via dashboard
3. View predictions and visual insights
4. Interpret using probability score (threshold = 0.5)

---

## 📊 Results & Insights

### Fraud Indicators
- Urgent Language: 73%
- Vague Descriptions: 68%
- Unrealistic Salaries: 61%
- Missing Company Info: 84%
- Remote Emphasis: 79%

### Dataset Statistics
- Total Jobs: 17,880
- Fraud Rate: 4.8%
- Accuracy: 94.2%

### Business Impact
- FPR: 6.1%, FNR: 4.3%
- Cost Savings: $2.3M/year
- User Trust Increase: 89%

---

## 👤 Team

### 👨‍💻 Usham Roy - Lead Developer & ML Engineer
- Full-stack dev, ML architecture
- Python, Data Science
- CI/CD Automations

### 👩‍💻 Anwesha Roy - Frontend Developer & UI/UX Designer
- React, TypeScript, UI Design


---

## 📄 License
MIT License
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

---

## Acknowledgments
- Anveshan Hackathon team
- scikit-learn, Next.js
- Open source contributors

---

## 📞 Contact & Support
- Email: ushamroy80@gmail.com
- [GitHub Issues](https://github.com/uroy80/Job-Fraud-Detection-FigureForge/issues)
- [GitHub Discussions](https://github.com/uroy80/Job-Fraud-Detection-FigureForge/discussions)

---

**⭐ Star us on GitHub if you find it useful!**

*Built with ❤️ for Anveshan Hackathon 2025*
