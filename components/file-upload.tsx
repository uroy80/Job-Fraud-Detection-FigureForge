"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Upload, FileText, AlertCircle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface FileUploadProps {
  onFileUpload: (file: File) => void
  isLoading: boolean
}

export default function FileUpload({ onFileUpload, isLoading }: FileUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isLoadingTestData, setIsLoadingTestData] = useState(false)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null)
    const file = e.target.files?.[0]

    if (!file) {
      return
    }

    if (!file.name.endsWith(".csv")) {
      setError("Please upload a CSV file")
      return
    }

    setSelectedFile(file)
  }

  const handleUpload = () => {
    if (selectedFile) {
      onFileUpload(selectedFile)
    }
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setError(null)

    const file = e.dataTransfer.files?.[0]

    if (!file) {
      return
    }

    if (!file.name.endsWith(".csv")) {
      setError("Please upload a CSV file")
      return
    }

    setSelectedFile(file)
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
  }

  const handleTestDataset = async () => {
    setIsLoadingTestData(true)
    setError(null)

    try {
      const response = await fetch(
        "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/test_data-edRByqq8jiF8qdQ7ItEsnmWYJrpQOj.csv",
      )
      const csvText = await response.text()

      // Create a File object from the CSV text
      const blob = new Blob([csvText], { type: "text/csv" })
      const file = new File([blob], "test_data.csv", { type: "text/csv" })

      onFileUpload(file)
    } catch (error) {
      console.error("Error loading test dataset:", error)
      setError("Failed to load test dataset. Please try again.")
    } finally {
      setIsLoadingTestData(false)
    }
  }

  return (
    <div className="space-y-4">
      <div
        className="flex flex-col items-center justify-center rounded-lg border border-dashed border-gray-300 p-6 cursor-pointer"
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <div className="flex flex-col items-center justify-center space-y-2">
          <div className="rounded-full bg-primary/10 p-2">
            <Upload className="h-6 w-6 text-primary" />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium">Click to upload or drag and drop</p>
            <p className="text-xs text-muted-foreground">CSV files only (max 10MB)</p>
          </div>
        </div>
      </div>

      <input type="file" ref={fileInputRef} onChange={handleFileChange} accept=".csv" className="hidden" />

      {/* Test Dataset Option */}
      <div className="space-y-3">
        <div className="flex items-center">
          <div className="flex-1 border-t border-gray-300"></div>
          <span className="px-3 text-xs text-muted-foreground bg-background">OR</span>
          <div className="flex-1 border-t border-gray-300"></div>
        </div>

        <div className="p-4 bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h4 className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-1">Data Starter Pack</h4>
              <p className="text-xs text-blue-700 dark:text-blue-300 mb-3">
                Try dataset from Evolve by Masai to explore fraud detection capabilities.
              </p>
            </div>
          </div>
          <Button
            onClick={handleTestDataset}
            disabled={isLoading || isLoadingTestData}
            variant="outline"
            size="sm"
            className="w-full border-blue-300 text-blue-700 hover:bg-blue-100 dark:border-blue-700 dark:text-blue-300 dark:hover:bg-blue-900/20"
          >
            {isLoadingTestData ? "Loading Test Data..." : "Use Test Dataset"}
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {selectedFile && (
        <div className="flex items-center justify-between rounded-lg border p-3">
          <div className="flex items-center space-x-3">
            <FileText className="h-5 w-5 text-muted-foreground" />
            <span className="text-sm font-medium">{selectedFile.name}</span>
          </div>
          <Button onClick={handleUpload} disabled={isLoading}>
            {isLoading ? "Processing..." : "Analyze"}
          </Button>
        </div>
      )}
    </div>
  )
}
