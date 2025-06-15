// Static Model Performance Data
// Based on your actual training results from the enhanced model

export const STATIC_MODEL_PERFORMANCE = {
  // Your actual training results
  accuracy: 0.983, // 98.3% from your training log
  precision: 0.79, // 79% fraud detection precision
  recall: 0.79, // 79% fraud detection recall
  f1_score: 0.7885, // 78.85% F1 score from your log
  auc_score: 0.9908, // 99.08% AUC from your log
  cv_f1_mean: 0.9908, // 99.08% CV F1 mean from your log
  cv_f1_std: 0.0015, // ±0.15% CV F1 std from your log

  // Your actual dataset info
  training_samples: 11443, // From your training log
  test_samples: 2861, // From your training log
  feature_count: 2027, // Combined feature matrix from your log

  // Model metadata
  model_type: "Enhanced Model",
  last_updated: "2025-01-14T22:42:55.000Z",
}
