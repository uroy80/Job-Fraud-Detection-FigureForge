import { type NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import fs from "fs"
import path from "path"
import { v4 as uuidv4 } from "uuid"
import { parse } from "csv-parse/sync"

const execAsync = promisify(exec)

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    // Create a temporary directory for processing
    const tempDir = path.join(process.cwd(), "tmp")
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir)
    }

    // Generate unique filenames
    const fileId = uuidv4()
    const inputPath = path.join(tempDir, `${fileId}_input.csv`)
    const outputPath = path.join(tempDir, `${fileId}_output.csv`)

    // Write the uploaded file to disk
    const buffer = Buffer.from(await file.arrayBuffer())
    fs.writeFileSync(inputPath, buffer)

    console.log(`Running enhanced prediction script on ${inputPath}`)

    try {
      // Try enhanced model first, fallback to basic model
      let scriptPath = "scripts/predict_enhanced.py"
      if (!fs.existsSync(scriptPath)) {
        scriptPath = "scripts/predict.py"
        console.log("Enhanced script not found, using basic prediction script")
      }

      // Execute the Python script with increased buffer size
      const { stdout, stderr } = await execAsync(`python ${scriptPath} "${inputPath}" "${outputPath}"`, {
        maxBuffer: 1024 * 1024 * 10, // 10MB buffer
        timeout: 300000, // 5 minute timeout
      })

      console.log("Python script completed successfully")
      if (stderr) {
        console.warn("Python script stderr:", stderr)
      }

      // Check if output file exists
      if (!fs.existsSync(outputPath)) {
        throw new Error("Prediction script did not generate output file")
      }

      // Read and parse the results
      const outputContent = fs.readFileSync(outputPath, "utf-8")
      const predictions = parse(outputContent, {
        columns: true,
        skip_empty_lines: true,
      })

      // Enhanced keyword data based on fraud detection patterns
      const keywords = [
        { text: "urgent", value: 45 },
        { text: "immediate", value: 42 },
        { text: "work from home", value: 38 },
        { text: "no experience", value: 35 },
        { text: "easy money", value: 40 },
        { text: "guaranteed", value: 37 },
        { text: "quick cash", value: 33 },
        { text: "asap", value: 30 },
        { text: "wire transfer", value: 28 },
        { text: "payment upfront", value: 25 },
        { text: "limited time", value: 22 },
        { text: "act now", value: 20 },
        { text: "financial freedom", value: 18 },
        { text: "investment opportunity", value: 15 },
        { text: "make money fast", value: 12 },
      ]

      // Calculate statistics
      const total_jobs = predictions.length
      const fraudulent_count = predictions.filter((p: any) => p.prediction === "fraudulent").length
      const genuine_count = total_jobs - fraudulent_count
      const fraud_rate = total_jobs > 0 ? ((fraudulent_count / total_jobs) * 100).toFixed(1) : "0"

      // Calculate average fraud probability
      const avg_fraud_prob =
        predictions.reduce((sum: number, p: any) => sum + Number.parseFloat(p.fraud_probability), 0) / total_jobs

      // Clean up temporary files
      try {
        fs.unlinkSync(inputPath)
        fs.unlinkSync(outputPath)
      } catch (cleanupError) {
        console.warn("Error cleaning up temporary files:", cleanupError)
      }

      return NextResponse.json({
        predictions,
        total_jobs,
        fraudulent_count,
        genuine_count,
        fraud_rate,
        avg_fraud_probability: avg_fraud_prob.toFixed(3),
        keywords,
        model_type: scriptPath.includes("enhanced") ? "Enhanced Model" : "Basic Model",
      })
    } catch (execError: any) {
      console.error("Error executing prediction script:", execError)

      // Clean up temporary files on error
      try {
        if (fs.existsSync(inputPath)) fs.unlinkSync(inputPath)
        if (fs.existsSync(outputPath)) fs.unlinkSync(outputPath)
      } catch (cleanupError) {
        console.warn("Error cleaning up temporary files:", cleanupError)
      }

      return NextResponse.json(
        {
          error: "Failed to process file",
          details: execError.message,
          code: execError.code,
        },
        { status: 500 },
      )
    }
  } catch (error: any) {
    console.error("Error processing file:", error)
    return NextResponse.json(
      {
        error: "Failed to process file",
        details: error.message,
      },
      { status: 500 },
    )
  }
}
