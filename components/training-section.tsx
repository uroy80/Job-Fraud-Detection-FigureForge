"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Brain, Clock, Zap, Target } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

export default function TrainingSection() {
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [selectedModel, setSelectedModel] = useState("enhanced")
  const [trainingStatus, setTrainingStatus] = useState("")
  const [estimatedTime, setEstimatedTime] = useState("")
  const { toast } = useToast()

  const modelTypes = {
    fast: {
      name: "Fast Model",
      time: "2-4 minutes",
      accuracy: "85-90%",
      description: "Quick training with essential features",
      icon: <Zap className="h-4 w-4" />,
    },
    enhanced: {
      name: "Enhanced Model",
      time: "5-8 minutes",
      accuracy: "90-94%",
      description: "Advanced feature engineering",
      icon: <Brain className="h-4 w-4" />,
    },
    ultra: {
      name: "Ultra-Enhanced Model",
      time: "10-15 minutes",
      accuracy: "94-98%",
      description: "Maximum accuracy with all techniques",
      icon: <Target className="h-4 w-4" />,
    },
  }

  const handleTraining = async () => {
    setIsTraining(true)
    setTrainingProgress(0)
    setTrainingStatus("Initializing training...")
    setEstimatedTime(modelTypes[selectedModel as keyof typeof modelTypes].time)

    try {
      const response = await fetch("/api/train-model", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ modelType: selectedModel }),
      })

      if (!response.ok) {
        throw new Error("Training failed")
      }

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setTrainingProgress((prev) => {
          if (prev >= 95) {
            clearInterval(progressInterval)
            return 95
          }
          return prev + Math.random() * 10
        })
      }, 2000)

      const result = await response.json()

      clearInterval(progressInterval)
      setTrainingProgress(100)
      setTrainingStatus("Training completed successfully!")

      toast({
        title: "Model Training Complete",
        description: `${modelTypes[selectedModel as keyof typeof modelTypes].name} trained successfully with ${result.accuracy}% accuracy`,
      })
    } catch (error) {
      console.error("Training error:", error)
      setTrainingStatus("Training failed. Please try again.")
      toast({
        title: "Training Failed",
        description: "An error occurred during model training.",
        variant: "destructive",
      })
    } finally {
      setIsTraining(false)
      setTimeout(() => {
        setTrainingProgress(0)
        setTrainingStatus("")
        setEstimatedTime("")
      }, 3000)
    }
  }

  const currentModel = modelTypes[selectedModel as keyof typeof modelTypes]

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          Model Training
        </CardTitle>
        <CardDescription>Train enhanced models with advanced techniques for better accuracy</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Select Model Type</label>
          <Select value={selectedModel} onValueChange={setSelectedModel} disabled={isTraining}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(modelTypes).map(([key, model]) => (
                <SelectItem key={key} value={key}>
                  <div className="flex items-center gap-2">
                    {model.icon}
                    <span>{model.name}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Expected Accuracy:</span>
            <Badge variant="secondary">{currentModel.accuracy}</Badge>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Training Time:</span>
            <div className="flex items-center gap-1">
              <Clock className="h-3 w-3" />
              <span className="text-sm text-muted-foreground">{currentModel.time}</span>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">{currentModel.description}</p>
        </div>

        {isTraining && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Progress:</span>
              <span className="text-sm text-muted-foreground">{Math.round(trainingProgress)}%</span>
            </div>
            <Progress value={trainingProgress} className="w-full" />
            <p className="text-xs text-muted-foreground">{trainingStatus}</p>
            {estimatedTime && <p className="text-xs text-muted-foreground">Estimated time: {estimatedTime}</p>}
          </div>
        )}

        <Button onClick={handleTraining} disabled={isTraining} className="w-full">
          {isTraining ? "Training..." : `Train ${currentModel.name}`}
        </Button>
      </CardContent>
    </Card>
  )
}
