# 🔍 Job Fraud Detection System

> **An AI-powered machine learning system to detect fraudulent job postings and protect job seekers from employment scams.**

[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat-square&logo=next.js)](https://nextjs.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue?style=flat-square&logo=typescript)](https://typescriptlang.org/)

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🛠️ Technologies Used](#️-technologies-used)
- [📊 Data Science Approach](#-data-science-approach)
- [🚀 Setup Instructions](#-setup-instructions)
- [📁 Project Structure](#-project-structure)
- [🔬 Model Performance](#-model-performance)
- [💾 Large Files & Models](#-large-files--models)
- [🎮 Usage Guide](#-usage-guide)
- [📈 Results & Insights](#-results--insights)
- [🚀 Deployment](#-deployment)
- [👥 Team](#-team)
- [📄 License](#-license)

---

## 🎯 Project Overview

The **Job Fraud Detection System** is an advanced machine learning application designed to identify fraudulent job postings with high accuracy. Built for the **Anveshan Hackathon**, this system combines cutting-edge NLP techniques with ensemble machine learning models to protect job seekers from employment scams.

### 🎯 Problem Statement
- **17.6%** of online job postings are fraudulent
- Job seekers lose **$2.7 billion annually** to employment scams
- Traditional keyword-based detection has **low accuracy** (60-70%)
- Need for **real-time, scalable** fraud detection system

### 🎯 Solution
Our system uses an **ensemble machine learning approach** with advanced feature engineering to achieve **90%+ accuracy** in detecting fraudulent job postings through:
- Advanced text analysis and NLP processing
- 25+ engineered features including fraud indicators
- Ensemble model combining Random Forest, Gradient Boosting, and Logistic Regression
- Interactive web dashboard for real-time analysis

---

## ✨ Key Features

### 🔍 **Core Detection Capabilities**
- **High Accuracy**: 90%+ F1-score with optimized ensemble model
- **Real-time Processing**: Analyze 1000+ job postings per second
- **Comprehensive Analysis**: 25+ engineered features for fraud detection
- **Probability Scoring**: Confidence scores for each prediction

### 📊 **Interactive Dashboard**
- **File Upload**: Drag-and-drop CSV file processing
- **Results Visualization**: Interactive charts and tables
- **Fraud Distribution**: Histogram of fraud probability scores
- **Top Suspicious Listings**: Ranked list of highest-risk jobs
- **Model Insights**: Performance metrics and feature analysis

### 🎨 **User Experience**
- **Responsive Design**: Mobile-first approach with dark/light themes
- **Real-time Updates**: Live processing status and results
- **Export Capabilities**: Download results and visualizations
- **Accessibility**: WCAG compliant interface

### 🔧 **Technical Features**
- **Scalable Architecture**: Next.js with Python ML backend
- **Model Training**: Automated retraining with new data
- **API Integration**: RESTful endpoints for external integration
- **Performance Monitoring**: Real-time model performance tracking

---

## 🛠️ Technologies Used

### **Frontend Stack**
- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Shadcn/ui** - Modern component library
- **Recharts** - Data visualization library
- **Lucide React** - Icon system

### **Backend & ML Stack**
- **Python 3.8+** - Core ML development
- **Scikit-learn** - Machine learning algorithms
- **Pandas & NumPy** - Data manipulation and analysis
- **NLTK** - Natural language processing
- **Matplotlib & Seaborn** - Data visualization
- **Imbalanced-learn** - Handling class imbalance

### **Development & Deployment**
- **Node.js** - JavaScript runtime
- **Git** - Version control
- **Railway/Vercel** - Cloud deployment
- **Docker** - Containerization (optional)

---

## 📊 Data Science Approach

### 🔬 **Data Processing Pipeline**

#### **1. Data Collection & Preprocessing**
```python
# Data cleaning and preprocessing steps
- Remove duplicates and handle missing values
- Text normalization (lowercase, remove special characters)
- Feature extraction from job descriptions and titles
- Label encoding for categorical variables
