// Static Model Insights Data - Based on your actual training results
export const STATIC_MODEL_INSIGHTS = {
  // ROC Curve data (extracted from your comprehensive image)
  roc_curve: {
    auc: 0.991, // From your training log and ROC curve
    // Simplified curve points for display
    fpr: [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    tpr: [0.0, 0.85, 0.92, 0.96, 0.98, 0.99, 0.995, 1.0],
  },

  // Precision-Recall curve data
  precision_recall_curve: {
    avg_precision: 0.887, // From your comprehensive image
    recall: [0.0, 0.1, 0.2, 0.4, 0.6, 0.79, 0.9, 1.0],
    precision: [1.0, 0.98, 0.95, 0.9, 0.85, 0.79, 0.65, 0.05],
  },

  // Learning curve data (from your comprehensive image)
  learning_curve: {
    train_sizes: [500, 1000, 1500, 2000, 2500, 3000],
    train_scores_mean: [0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    train_scores_std: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    val_scores_mean: [0.05, 0.21, 0.36, 0.43, 0.47, 0.52],
    val_scores_std: [0.02, 0.03, 0.04, 0.04, 0.04, 0.05],
  },

  // Model comparison (from your model_comparison.png)
  model_comparison: {
    models: ["Random Forest", "Gradient Boosting", "Logistic Regression", "Ensemble"],
    scores: [0.764, 0.795, 0.674, 0.789],
  },

  // Threshold analysis (optimized values)
  threshold_analysis: {
    thresholds: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    f1_scores: [0.65, 0.72, 0.76, 0.785, 0.789, 0.78, 0.75, 0.7, 0.6],
    precision_scores: [0.55, 0.65, 0.72, 0.76, 0.79, 0.82, 0.85, 0.88, 0.92],
    recall_scores: [0.95, 0.88, 0.82, 0.8, 0.79, 0.75, 0.7, 0.65, 0.45],
  },

  // Feature importance (from your feature_importance.png - top 15)
  feature_importance: {
    features: [
      "Has Company Description",
      "Has Company Logo",
      "Company Profile Length",
      "Has Questions",
      "Function Encoded",
      "Description Word Count",
      "Requires Experience",
      "Money Mentions",
      "Department Encoded",
      "Description Length",
      "Fraud Keywords Count",
      "Title Word Count",
      "Location Length",
      "Salary Length",
      "Requires Education",
    ],
    importance: [0.053, 0.04, 0.04, 0.017, 0.009, 0.005, 0.004, 0.004, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002, 0.001],
  },

  // Class distribution (from your training log and comprehensive image)
  class_distribution: {
    original: {
      "0": 13611, // Genuine jobs
      "1": 693, // Fraudulent jobs
    },
    after_smote: {
      "0": 10889, // Balanced after SMOTE
      "1": 10889, // Balanced after SMOTE
    },
  },

  // Model information (from your training log)
  model_info: {
    name: "FigureForge-Anveshan Enhanced Ensemble",
    version: "v2.1.0",
    algorithm: "Ensemble (RF + GB + LR)",
    training_time: "18.57 minutes",
    dataset_size: 14304,
    feature_count: 2027,
    accuracy: 0.983,
    precision: 0.79,
    recall: 0.79,
    f1_score: 0.7885,
  },

  // Fraud keywords analysis (from your fraud_keywords_analysis.png)
  fraud_keywords: [
    { keyword: "fast", frequency: 4200 },
    { keyword: "quick", frequency: 2100 },
    { keyword: "easy", frequency: 800 },
    { keyword: "immediate", frequency: 750 },
    { keyword: "confidential", frequency: 650 },
    { keyword: "investment", frequency: 500 },
    { keyword: "cash", frequency: 400 },
    { keyword: "exclusive", frequency: 300 },
    { keyword: "urgent", frequency: 250 },
    { keyword: "secret", frequency: 200 },
    { keyword: "guaranteed", frequency: 150 },
    { keyword: "asap", frequency: 120 },
    { keyword: "make money", frequency: 80 },
    { keyword: "wire transfer", frequency: 60 },
    { keyword: "earn money", frequency: 50 },
  ],

  // Confusion matrix (from your confusion_matrix_enhanced.png)
  confusion_matrix: {
    true_negatives: 2692,
    false_positives: 30,
    false_negatives: 29,
    true_positives: 110,
  },
}
