"use client"

import { useEffect, useState } from "react"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { HelpCircle, RefreshCw, Eye } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

interface ModelMetrics {
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  auc_score: number
  cv_f1_mean: number
  cv_f1_std: number
  training_samples: number
  test_samples: number
  feature_count: number
  last_updated?: string
  model_type?: string
  is_static?: boolean
  training_time_minutes?: number
}

export default function ModelPerformanceReal() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchMetrics = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch("/api/model-performance")
      if (!response.ok) {
        throw new Error("Failed to fetch model performance")
      }
      const data = await response.json()
      setMetrics(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchMetrics()
  }, [])

  const handleInsightsClick = () => {
    // This will toggle the insights dialog
    window.dispatchEvent(new CustomEvent("toggleModelInsights"))
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-32">
        <RefreshCw className="h-6 w-6 animate-spin" />
        <span className="ml-2">Loading model performance...</span>
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="flex items-center justify-center h-32 text-muted-foreground">
        <span>Unable to load model performance</span>
      </div>
    )
  }

  const getPerformanceColor = (value: number, type: "percentage" | "score" = "percentage") => {
    if (value === 0) return "text-gray-500"
    if (type === "percentage") {
      if (value >= 0.9) return "text-green-600"
      if (value >= 0.8) return "text-green-500"
      if (value >= 0.7) return "text-yellow-600"
      return "text-red-500"
    }
    return "text-blue-600"
  }

  const formatPercentage = (value: number) => {
    return value === 0 ? "N/A" : `${(value * 100).toFixed(1)}%`
  }

  const formatNumber = (value: number) => {
    return value === 0 ? "N/A" : value.toFixed(3)
  }

  const formatCount = (value: number) => {
    return value === 0 ? "N/A" : value.toLocaleString()
  }

  return (
    <div className="space-y-4">
      {/* Header with model info and status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Badge variant="default" className="bg-green-600">
            {metrics.model_type || "Enhanced Model"}
          </Badge>
          {metrics.last_updated && (
            <span className="text-xs text-muted-foreground">
              Trained: {new Date(metrics.last_updated).toLocaleDateString()}
            </span>
          )}
        </div>
        <div className="flex items-center space-x-1">
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={handleInsightsClick}
            title="Toggle Model Insights"
          >
            <Eye className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="sm" onClick={fetchMetrics} className="h-8 w-8 p-0">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Main metrics grid */}
      <div className="grid grid-cols-2 gap-4">
        <div className="flex flex-col items-center justify-center space-y-1">
          <div className="flex items-center">
            <span className="text-sm font-medium text-muted-foreground">Accuracy</span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="ml-1 h-3 w-3 text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs text-xs">Overall prediction accuracy on test set</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <span className={`text-2xl font-bold ${getPerformanceColor(metrics.accuracy)}`}>
            {formatPercentage(metrics.accuracy)}
          </span>
        </div>

        <div className="flex flex-col items-center justify-center space-y-1">
          <div className="flex items-center">
            <span className="text-sm font-medium text-muted-foreground">Precision</span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="ml-1 h-3 w-3 text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs text-xs">When model predicts fraud, how often is it correct?</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <span className={`text-2xl font-bold ${getPerformanceColor(metrics.precision)}`}>
            {formatPercentage(metrics.precision)}
          </span>
        </div>

        <div className="flex flex-col items-center justify-center space-y-1">
          <div className="flex items-center">
            <span className="text-sm font-medium text-muted-foreground">Recall</span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="ml-1 h-3 w-3 text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs text-xs">How many actual frauds did the model catch?</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <span className={`text-2xl font-bold ${getPerformanceColor(metrics.recall)}`}>
            {formatPercentage(metrics.recall)}
          </span>
        </div>

        <div className="flex flex-col items-center justify-center space-y-1">
          <div className="flex items-center">
            <span className="text-sm font-medium text-muted-foreground">F1 Score</span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="ml-1 h-3 w-3 text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs text-xs">Balance between precision and recall - your actual training result</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <span className={`text-2xl font-bold ${getPerformanceColor(metrics.f1_score)}`}>
            {formatPercentage(metrics.f1_score)}
          </span>
        </div>
      </div>

      {/* AUC Score - highlighted as excellent */}
      <div className="flex flex-col items-center justify-center space-y-1 pt-2 border-t">
        <span className="text-sm font-medium text-muted-foreground">AUC Score</span>
        <span className={`text-lg font-semibold ${getPerformanceColor(metrics.auc_score)}`}>
          {formatPercentage(metrics.auc_score)}
        </span>
      </div>

      {/* Cross-validation */}
      <div className="flex flex-col items-center justify-center space-y-1 pt-2 border-t">
        <span className="text-sm font-medium text-muted-foreground">Cross-Validation F1</span>
        <span className={`text-lg font-semibold ${getPerformanceColor(metrics.cv_f1_mean)}`}>
          {formatPercentage(metrics.cv_f1_mean)} ±{formatPercentage(metrics.cv_f1_std)}
        </span>
      </div>

      {/* Training info */}
      <div className="text-xs text-muted-foreground space-y-1 pt-2 border-t">
        <div className="flex justify-between">
          <span>Training Samples:</span>
          <span className="font-medium">{formatCount(metrics.training_samples)}</span>
        </div>
        <div className="flex justify-between">
          <span>Test Samples:</span>
          <span className="font-medium">{formatCount(metrics.test_samples)}</span>
        </div>
        <div className="flex justify-between">
          <span>Total Features:</span>
          <span className="font-medium">{formatCount(metrics.feature_count)}</span>
        </div>
      </div>
    </div>
  )
}
