"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { RefreshCw, TrendingUp, BarChart3, PieChart, Target, AlertCircle, Brain, Database, Zap } from "lucide-react"
import { STATIC_MODEL_INSIGHTS } from "@/lib/static-insights-data"

interface ModelInsights {
  roc_curve?: {
    fpr: number[]
    tpr: number[]
    auc: number
  }
  precision_recall_curve?: {
    precision: number[]
    recall: number[]
    avg_precision: number
  }
  learning_curve?: {
    train_sizes: number[]
    train_scores_mean: number[]
    train_scores_std: number[]
    val_scores_mean: number[]
    val_scores_std: number[]
  }
  model_comparison?: {
    models: string[]
    scores: number[]
  }
  threshold_analysis?: {
    thresholds: number[]
    f1_scores: number[]
    precision_scores: number[]
    recall_scores: number[]
  }
  feature_importance?: {
    features: string[]
    importance: number[]
  }
  class_distribution?: {
    original: { [key: string]: number }
    after_smote?: { [key: string]: number }
  }
  model_info?: {
    name: string
    version: string
    algorithm: string
    training_time: string
    dataset_size: number
    feature_count: number
    accuracy: number
    precision: number
    recall: number
    f1_score: number
  }
}

export default function ModelInsights() {
  const [insights, setInsights] = useState<ModelInsights | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Load static insights data immediately
    setInsights(STATIC_MODEL_INSIGHTS)
    setLoading(false)
    setError(null)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-6 w-6 animate-spin mr-2" />
        <span>Loading model insights...</span>
      </div>
    )
  }

  if (error || !insights) {
    return (
      <div className="flex flex-col items-center justify-center h-64 space-y-4">
        <AlertCircle className="h-12 w-12 text-muted-foreground" />
        <div className="text-center space-y-2">
          <h3 className="text-lg font-semibold">No Model Insights Available</h3>
          <p className="text-sm text-muted-foreground max-w-md text-center">
            Train the FigureForge-Anveshan model first to generate comprehensive insights, performance curves, and
            detailed analysis.
          </p>
        </div>
        <Button onClick={() => setInsights(STATIC_MODEL_INSIGHTS)} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Load Static Data
        </Button>
      </div>
    )
  }

  // Prepare data for charts using static data
  const rocData =
    insights.roc_curve?.fpr?.map((fpr, i) => ({
      fpr: Number.parseFloat(fpr.toFixed(4)),
      tpr: Number.parseFloat(insights.roc_curve!.tpr[i].toFixed(4)),
    })) || []

  const prData =
    insights.precision_recall_curve?.recall?.map((recall, i) => ({
      recall: Number.parseFloat(recall.toFixed(4)),
      precision: Number.parseFloat(insights.precision_recall_curve!.precision[i].toFixed(4)),
    })) || []

  const learningData =
    insights.learning_curve?.train_sizes?.map((size, i) => ({
      size,
      train_score: Number.parseFloat(insights.learning_curve!.train_scores_mean[i].toFixed(4)),
      val_score: Number.parseFloat(insights.learning_curve!.val_scores_mean[i].toFixed(4)),
    })) || []

  const modelComparisonData =
    insights.model_comparison?.models?.map((model, i) => ({
      model: model.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()),
      score: Number.parseFloat(insights.model_comparison!.scores[i].toFixed(4)),
    })) || []

  const thresholdData =
    insights.threshold_analysis?.thresholds?.map((threshold, i) => ({
      threshold: Number.parseFloat(threshold.toFixed(3)),
      f1: Number.parseFloat(insights.threshold_analysis!.f1_scores[i].toFixed(4)),
      precision: Number.parseFloat(insights.threshold_analysis!.precision_scores[i].toFixed(4)),
      recall: Number.parseFloat(insights.threshold_analysis!.recall_scores[i].toFixed(4)),
    })) || []

  const featureImportanceData =
    insights.feature_importance?.features
      ?.map((feature, i) => ({
        feature:
          feature
            .replace(/_/g, " ")
            .replace(/\b\w/g, (l) => l.toUpperCase())
            .substring(0, 20) + (feature.length > 20 ? "..." : ""),
        importance: Number.parseFloat(insights.feature_importance!.importance[i].toFixed(4)),
      }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 15) || []

  const classDistributionData = insights.class_distribution?.original
    ? [
        {
          name: "Genuine Jobs",
          value: insights.class_distribution.original["0"] || 0,
          color: "#10b981",
        },
        {
          name: "Fraudulent Jobs",
          value: insights.class_distribution.original["1"] || 0,
          color: "#ef4444",
        },
      ]
    : []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="h-6 w-6 text-blue-600" />
            FigureForge-Anveshan Analysis
          </h2>
          <p className="text-muted-foreground text-left">
            Advanced job fraud detection model with comprehensive performance analysis
          </p>
        </div>
        <Button onClick={() => setInsights(STATIC_MODEL_INSIGHTS)} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh Data
        </Button>
      </div>

      {/* Model Architecture Overview */}
      {insights.model_info && (
        <Card className="border-blue-200 bg-blue-50/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-blue-900">
              <Zap className="h-5 w-5" />
              Model Architecture & Performance
            </CardTitle>
            <CardDescription className="text-blue-700 text-left">
              Core specifications and performance metrics of the trained model
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4 text-blue-600" />
                  <span className="text-sm font-medium text-muted-foreground">Model Details</span>
                </div>
                <div className="space-y-2 text-left">
                  <div>
                    <p className="text-xs text-muted-foreground">Name</p>
                    <p className="font-semibold">{insights.model_info.name}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Algorithm</p>
                    <p className="font-semibold">{insights.model_info.algorithm}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Version</p>
                    <p className="font-semibold">{insights.model_info.version}</p>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Target className="h-4 w-4 text-green-600" />
                  <span className="text-sm font-medium text-muted-foreground">Performance Metrics</span>
                </div>
                <div className="space-y-2 text-left">
                  <div>
                    <p className="text-xs text-muted-foreground">Accuracy</p>
                    <p className="font-semibold text-green-600">{(insights.model_info.accuracy * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">F1 Score</p>
                    <p className="font-semibold text-green-600">{(insights.model_info.f1_score * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Precision</p>
                    <p className="font-semibold text-blue-600">{(insights.model_info.precision * 100).toFixed(1)}%</p>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4 text-purple-600" />
                  <span className="text-sm font-medium text-muted-foreground">Training Data</span>
                </div>
                <div className="space-y-2 text-left">
                  <div>
                    <p className="text-xs text-muted-foreground">Dataset Size</p>
                    <p className="font-semibold">{insights.model_info.dataset_size.toLocaleString()} samples</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Features</p>
                    <p className="font-semibold">{insights.model_info.feature_count.toLocaleString()} features</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Training Time</p>
                    <p className="font-semibold">{insights.model_info.training_time}</p>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-orange-600" />
                  <span className="text-sm font-medium text-muted-foreground">Detection Capability</span>
                </div>
                <div className="space-y-2 text-left">
                  <div>
                    <p className="text-xs text-muted-foreground">Recall Rate</p>
                    <p className="font-semibold text-orange-600">{(insights.model_info.recall * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Fraud Detection</p>
                    <p className="font-semibold text-red-600">High Precision</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">False Positives</p>
                    <p className="font-semibold text-green-600">Minimized</p>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="performance" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="performance" className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            ROC & PR Curves
          </TabsTrigger>
          <TabsTrigger value="curves" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Learning Analysis
          </TabsTrigger>
          <TabsTrigger value="features" className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            Feature Insights
          </TabsTrigger>
          <TabsTrigger value="distribution" className="flex items-center gap-2">
            <PieChart className="h-4 w-4" />
            Data Overview
          </TabsTrigger>
        </TabsList>

        <TabsContent value="performance" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-left">ROC Curve Analysis</CardTitle>
                <CardDescription className="text-left">
                  Receiver Operating Characteristic - AUC Score: {insights.roc_curve?.auc.toFixed(3)}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[350px] flex items-center justify-center">
                  <img
                    src="/images/model_insights_comprehensive.png"
                    alt="ROC Curve and Performance Analysis"
                    className="max-w-full max-h-full object-contain"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-left">Model Performance Comparison</CardTitle>
                <CardDescription className="text-left">
                  F1 Score Performance Across Different Algorithms
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[350px] flex items-center justify-center">
                  <img
                    src="/images/model_comparison.png"
                    alt="Model Performance Comparison"
                    className="max-w-full max-h-full object-contain"
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="curves" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-left">Feature Importance Analysis</CardTitle>
                <CardDescription className="text-left">
                  Top 20 Most Critical Features for Fraud Detection
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[350px] flex items-center justify-center">
                  <img
                    src="/images/feature_importance.png"
                    alt="Feature Importance Analysis"
                    className="max-w-full max-h-full object-contain"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-left">Confusion Matrix</CardTitle>
                <CardDescription className="text-left">Enhanced Model Performance Matrix</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[350px] flex items-center justify-center">
                  <img
                    src="/images/confusion_matrix_enhanced.png"
                    alt="Confusion Matrix Enhanced"
                    className="max-w-full max-h-full object-contain"
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="features" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-left">Feature Correlation Matrix</CardTitle>
                <CardDescription className="text-left">Correlation Between Different Features</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[400px] flex items-center justify-center">
                  <img
                    src="/images/feature_correlation.png"
                    alt="Feature Correlation Matrix"
                    className="max-w-full max-h-full object-contain"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-left">Fraud Keywords Analysis</CardTitle>
                <CardDescription className="text-left">Top Fraud Keywords Found in Dataset</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[400px] flex items-center justify-center">
                  <img
                    src="/images/fraud_keywords_analysis.png"
                    alt="Fraud Keywords Analysis"
                    className="max-w-full max-h-full object-contain"
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="distribution" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-left">Training Dataset Statistics</CardTitle>
              <CardDescription className="text-left">Comprehensive Overview of Model Training Data</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {(classDistributionData[0]?.value + classDistributionData[1]?.value || 0).toLocaleString()}
                  </div>
                  <div className="text-sm text-blue-700 font-medium">Total Job Postings</div>
                </div>
                <div className="text-center p-3 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {classDistributionData[0]?.value
                      ? (
                          (classDistributionData[0].value /
                            (classDistributionData[0].value + classDistributionData[1].value)) *
                          100
                        ).toFixed(1)
                      : "0"}
                    %
                  </div>
                  <div className="text-sm text-green-700 font-medium">Genuine Jobs</div>
                </div>
                <div className="text-center p-3 bg-red-50 rounded-lg">
                  <div className="text-2xl font-bold text-red-600">
                    {classDistributionData[1]?.value
                      ? (
                          (classDistributionData[1].value /
                            (classDistributionData[0].value + classDistributionData[1].value)) *
                          100
                        ).toFixed(1)
                      : "0"}
                    %
                  </div>
                  <div className="text-sm text-red-700 font-medium">Fraudulent Jobs</div>
                </div>
                <div className="text-center p-3 bg-purple-50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">
                    {insights.feature_importance?.features.length || 0}
                  </div>
                  <div className="text-sm text-purple-700 font-medium">Analyzed Features</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
