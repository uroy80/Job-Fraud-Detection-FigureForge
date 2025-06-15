"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { AlertCircle, CheckCircle, RefreshCw, Play } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

export default function TrainingStatus() {
  const [isTraining, setIsTraining] = useState(false)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState<string>("")
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const startTraining = async () => {
    setIsTraining(true)
    setProgress(0)
    setError(null)
    setSuccess(null)
    setStatus("Initializing training...")

    try {
      // Simulate training progress
      const progressSteps = [
        { progress: 10, status: "Loading training data..." },
        { progress: 25, status: "Extracting features..." },
        { progress: 40, status: "Preprocessing text..." },
        { progress: 60, status: "Training ensemble model..." },
        { progress: 80, status: "Evaluating performance..." },
        { progress: 95, status: "Saving model..." },
        { progress: 100, status: "Training complete!" },
      ]

      for (const step of progressSteps) {
        await new Promise((resolve) => setTimeout(resolve, 2000))
        setProgress(step.progress)
        setStatus(step.status)
      }

      // Trigger actual training via API
      const response = await fetch("/api/train-model", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      })

      if (!response.ok) {
        throw new Error("Training failed")
      }

      const result = await response.json()
      setSuccess(`Training completed successfully! F1 Score: ${result.f1_score?.toFixed(3) || "N/A"}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed")
    } finally {
      setIsTraining(false)
      setProgress(0)
      setStatus("")
    }
  }

  const evaluateModel = async () => {
    try {
      setStatus("Evaluating model performance...")
      const response = await fetch("/api/model-performance", {
        method: "POST",
      })

      if (!response.ok) {
        throw new Error("Evaluation failed")
      }

      setSuccess("Model evaluation completed!")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation failed")
    } finally {
      setStatus("")
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <RefreshCw className="h-5 w-5" />
          Model Training
        </CardTitle>
        <CardDescription>Train or retrain the fraud detection model with your data</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {success && (
          <Alert>
            <CheckCircle className="h-4 w-4" />
            <AlertDescription>{success}</AlertDescription>
          </Alert>
        )}

        {isTraining && (
          <div className="space-y-2">
            <Progress value={progress} className="w-full" />
            <p className="text-sm text-muted-foreground">{status}</p>
          </div>
        )}

        <div className="flex gap-2">
          <Button onClick={startTraining} disabled={isTraining} className="flex-1">
            {isTraining ? <RefreshCw className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
            {isTraining ? "Training..." : "Train Enhanced Model"}
          </Button>

          <Button variant="outline" onClick={evaluateModel} disabled={isTraining}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Evaluate
          </Button>
        </div>

        <div className="text-xs text-muted-foreground">
          <p>• Enhanced model uses 25+ features and ensemble methods</p>
          <p>• Training typically takes 2-5 minutes depending on data size</p>
          <p>• Model performance will update automatically after training</p>
        </div>
      </CardContent>
    </Card>
  )
}
