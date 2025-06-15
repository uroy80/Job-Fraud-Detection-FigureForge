import { type NextRequest, NextResponse } from "next/server"
import fs from "fs"
import path from "path"

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const filename = searchParams.get("file")

    if (!filename) {
      return NextResponse.json({ error: "No filename provided" }, { status: 400 })
    }

    // Validate filename to prevent directory traversal
    const allowedFiles = [
      "model_insights.png",
      "feature_correlation.png",
      "data_analysis.png",
      "prediction_analysis.png",
      "feature_importance.png",
      "confusion_matrix_enhanced.png",
    ]

    if (!allowedFiles.includes(filename)) {
      return NextResponse.json({ error: "File not allowed" }, { status: 403 })
    }

    const filePath = path.join(process.cwd(), filename)

    if (!fs.existsSync(filePath)) {
      return NextResponse.json({ error: "File not found" }, { status: 404 })
    }

    const fileBuffer = fs.readFileSync(filePath)

    return new NextResponse(fileBuffer, {
      headers: {
        "Content-Type": "image/png",
        "Content-Disposition": `attachment; filename="${filename}"`,
      },
    })
  } catch (error) {
    console.error("Error downloading visualization:", error)
    return NextResponse.json(
      {
        error: "Failed to download visualization",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
