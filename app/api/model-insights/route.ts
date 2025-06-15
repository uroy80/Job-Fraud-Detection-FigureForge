import { NextResponse } from "next/server"
import fs from "fs"
import path from "path"
import { exec } from "child_process"
import { promisify } from "util"

const execAsync = promisify(exec)

export async function GET() {
  try {
    // Try to load saved insights
    const insightsPath = path.join(process.cwd(), "model_insights.pkl")

    if (fs.existsSync(insightsPath)) {
      try {
        const { stdout } = await execAsync(`python -c "
import pickle
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

try:
    with open('model_insights.pkl', 'rb') as f:
        insights = pickle.load(f)
    
    print(json.dumps(insights, cls=NumpyEncoder))
except Exception as e:
    print(json.dumps({'error': str(e)}))
"`)

        const insights = JSON.parse(stdout.trim())

        if (insights.error) {
          throw new Error(insights.error)
        }

        return NextResponse.json(insights)
      } catch (pythonError) {
        console.error("Error loading model insights:", pythonError)
      }
    }

    // Return empty response if no insights available
    return NextResponse.json(
      {
        error: "No model insights available. Train a model first to generate insights.",
        available_files: {
          model_insights: fs.existsSync(path.join(process.cwd(), "model_insights.png")),
          feature_correlation: fs.existsSync(path.join(process.cwd(), "feature_correlation.png")),
          data_analysis: fs.existsSync(path.join(process.cwd(), "data_analysis.png")),
          prediction_analysis: fs.existsSync(path.join(process.cwd(), "prediction_analysis.png")),
        },
      },
      { status: 404 },
    )
  } catch (error) {
    console.error("Error fetching model insights:", error)
    return NextResponse.json(
      {
        error: "Failed to fetch model insights",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
