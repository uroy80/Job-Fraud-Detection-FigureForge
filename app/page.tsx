import type { Metadata } from "next"
import Dashboard from "@/components/dashboard"
import SpaceBackground from "@/components/space-background"
import { ThemeToggle } from "@/components/theme-toggle"

export const metadata: Metadata = {
  title: "Job Fraud Detection System",
  description: "Detect fraudulent job postings using machine learning",
}

export default function Home() {
  return (
    <div className="min-h-screen relative">
      <SpaceBackground />
      <ThemeToggle />
      <div className="relative z-10">
        <main className="container mx-auto py-12 px-4 sm:px-6 lg:px-8">
          <div className="mb-12 space-y-6 text-center fade-in">
            <h1 className="heading-primary text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight gradient-text leading-tight px-2">
              Job Fraud Detection System
            </h1>
            <p className="text-body text-lg text-foreground max-w-2xl mx-auto leading-relaxed px-4">
              Upload a CSV file with job listings to detect potential fraudulent job postings using advanced machine
              learning algorithms and natural language processing.
            </p>
          </div>
          <div className="fade-in">
            <Dashboard />
          </div>
        </main>
      </div>
    </div>
  )
}
