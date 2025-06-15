"use client"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { HelpCircle } from "lucide-react"

export default function ModelPerformance() {
  // These would typically come from your model evaluation
  const metrics = {
    accuracy: 0.92,
    precision: 0.89,
    recall: 0.85,
    f1: 0.87,
  }

  return (
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
                <p className="max-w-xs text-xs">Proportion of correct predictions among the total predictions</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <span className="text-2xl font-bold">{(metrics.accuracy * 100).toFixed(1)}%</span>
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
                <p className="max-w-xs text-xs">Proportion of true fraud predictions among all fraud predictions</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <span className="text-2xl font-bold">{(metrics.precision * 100).toFixed(1)}%</span>
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
                <p className="max-w-xs text-xs">Proportion of actual frauds that were correctly identified</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <span className="text-2xl font-bold">{(metrics.recall * 100).toFixed(1)}%</span>
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
                <p className="max-w-xs text-xs">Harmonic mean of precision and recall, best for imbalanced datasets</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <span className="text-2xl font-bold">{(metrics.f1 * 100).toFixed(1)}%</span>
      </div>
    </div>
  )
}
