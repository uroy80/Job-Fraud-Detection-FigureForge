import { NextResponse } from "next/server"
import { STATIC_MODEL_PERFORMANCE } from "@/lib/static-performance-data"

export async function GET() {
  try {
    // Return your actual static performance data
    console.log("Returning static model performance data")

    return NextResponse.json({
      ...STATIC_MODEL_PERFORMANCE,
      message: "Performance data from trained model",
    })
  } catch (error) {
    console.error("Error fetching static model performance:", error)
    return NextResponse.json(
      {
        error: "Failed to fetch model performance",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}

// POST endpoint for refresh (just returns the same static data)
export async function POST() {
  console.log("Refresh requested - returning static data")
  return GET()
}
