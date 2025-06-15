import { NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import fs from "fs"

const execAsync = promisify(exec)

export async function POST(request: Request) {
  try {
    const { modelType = "enhanced" } = await request.json()

    console.log(`Starting ${modelType} model training...`)

    // Check if training data exists
    if (!fs.existsSync("training_data.csv")) {
      return NextResponse.json(
        { error: "No training data found. Please upload training_data.csv to the project root." },
        { status: 400 },
      )
    }

    // Choose training script based on model type
    let scriptPath = "scripts/train_enhanced_model.py"
    let timeout = 300000 // 5 minutes default

    switch (modelType) {
      case "ultra":
        scriptPath = "scripts/train_ultra_enhanced_model.py"
        timeout = 900000 // 15 minutes for ultra model
        break
      case "fast":
        scriptPath = "scripts/train_enhanced_model_fast.py"
        timeout = 180000 // 3 minutes for fast model
        break
      case "enhanced":
      default:
        scriptPath = "scripts/train_enhanced_model.py"
        timeout = 300000 // 5 minutes for enhanced model
        break
    }

    console.log(`Using script: ${scriptPath}`)

    // Run the training script
    const { stdout, stderr } = await execAsync(`python ${scriptPath}`, {
      timeout: timeout,
      maxBuffer: 1024 * 1024 * 20, // 20MB buffer for ultra model
    })

    console.log("Training completed successfully")

    if (stderr) {
      console.warn("Training script stderr:", stderr)
    }

    // Try to load the performance metrics
    let metrics = null
    try {
      const { stdout: metricsOutput } = await execAsync(`python -c "
import pickle
import json
try:
    with open('model_performance.pkl', 'rb') as f:
        metrics = pickle.load(f)
    # Convert numpy types to native Python types for JSON serialization
    for key, value in metrics.items():
        if hasattr(value, 'item'):
            metrics[key] = value.item()
    print(json.dumps(metrics))
except Exception as e:
    print(json.dumps({'error': str(e)}))
"`)
      metrics = JSON.parse(metricsOutput.trim())
    } catch (e) {
      console.warn("Could not load performance metrics:", e)
    }

    // Determine model type name
    const modelTypeNames = {
      ultra: "Ultra-Enhanced Stacking Ensemble",
      enhanced: "Enhanced Ensemble Model",
      fast: "Fast Enhanced Model",
    }

    return NextResponse.json({
      success: true,
      message: `${modelTypeNames[modelType] || "Enhanced"} training completed successfully`,
      metrics: metrics,
      f1_score: metrics?.f1_score || null,
      model_type: modelTypeNames[modelType] || "Enhanced Model",
      training_time: metrics?.training_time_seconds || null,
      feature_count: metrics?.feature_count || null,
    })
  } catch (error: any) {
    console.error("Error during model training:", error)

    // Handle timeout specifically
    if (error.code === "ETIMEDOUT") {
      return NextResponse.json(
        {
          error: "Model training timed out",
          details: "Training took longer than expected. Try using the 'fast' model type for quicker training.",
          code: "TIMEOUT",
        },
        { status: 408 },
      )
    }

    return NextResponse.json(
      {
        error: "Model training failed",
        details: error.message,
        code: error.code,
      },
      { status: 500 },
    )
  }
}
