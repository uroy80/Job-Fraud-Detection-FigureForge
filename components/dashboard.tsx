"use client"

import { useState, useEffect } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import FileUpload from "@/components/file-upload"
import ResultsTable from "@/components/results-table"
import FraudDistribution from "@/components/fraud-distribution"
import FraudPieChart from "@/components/fraud-pie-chart"
import TopSuspiciousListings from "@/components/top-suspicious-listings"
import ModelPerformanceReal from "@/components/model-performance-real"
import ModelInsights from "@/components/model-insights"
import AboutDialog from "@/components/about-dialog"
import { TrendingUp, AlertTriangle, Shield, X, Info } from "lucide-react"

interface JobListing {
  id: string
  job_id?: string
  title: string
  company: string
  location: string
  fraud_probability: number
  prediction: "genuine" | "fraudulent"
}

interface PredictionResults {
  predictions: JobListing[]
  total_jobs: number
  fraudulent_count: number
  genuine_count: number
  fraud_rate: string
  avg_fraud_probability: string
  keywords: Array<{ text: string; value: number }>
  model_type: string
}

export default function Dashboard() {
  const [results, setResults] = useState<PredictionResults | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [activeTab, setActiveTab] = useState("overview")
  const [showResultInsights, setShowResultInsights] = useState(false)
  const [showModelInsights, setShowModelInsights] = useState(false)
  const [showAbout, setShowAbout] = useState(false)

  // Listen for model insights toggle event
  useEffect(() => {
    const handleToggleInsights = () => {
      setShowModelInsights((prev) => !prev)
    }

    window.addEventListener("toggleModelInsights", handleToggleInsights)

    return () => {
      window.removeEventListener("toggleModelInsights", handleToggleInsights)
    }
  }, [])

  const handleFileUpload = async (file: File) => {
    setIsLoading(true)

    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Failed to process file")
      }

      const data = await response.json()
      setResults(data)
      setActiveTab("overview")
    } catch (error) {
      console.error("Error processing file:", error)
      alert("Error processing file. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  // Filter jobs with >51% fraud probability
  const highRiskJobs = results?.predictions?.filter((job: JobListing) => job.fraud_probability > 0.51) || []

  // Calculate insights for results
  const getResultInsights = () => {
    if (!results) return null

    const predictions = results.predictions
    const avgProb = Number.parseFloat(results.avg_fraud_probability)
    const highRisk = predictions.filter((job) => job.fraud_probability > 0.8).length
    const mediumRisk = predictions.filter((job) => job.fraud_probability > 0.5 && job.fraud_probability <= 0.8).length
    const lowRisk = predictions.filter((job) => job.fraud_probability <= 0.5).length

    // Most common fraud indicators
    const topKeywords = results.keywords.slice(0, 5)

    // Risk distribution
    const riskDistribution = {
      high: ((highRisk / predictions.length) * 100).toFixed(1),
      medium: ((mediumRisk / predictions.length) * 100).toFixed(1),
      low: ((lowRisk / predictions.length) * 100).toFixed(1),
    }

    return {
      avgProb,
      highRisk,
      mediumRisk,
      lowRisk,
      riskDistribution,
      topKeywords,
      totalJobs: predictions.length,
      fraudRate: Number.parseFloat(results.fraud_rate),
    }
  }

  const resultInsights = getResultInsights()

  return (
    <div className="space-y-10">
      {/* Header with About Us Button - Better spacing */}
      <div className="flex justify-between items-center mb-8">
        <div></div>
        <Button onClick={() => setShowAbout(true)} className="btn-space" size="sm">
          <Info className="h-4 w-4 mr-2" />
          About Us
        </Button>
      </div>

      {/* Top Section - Upload and Stats - Improved spacing */}
      <div className="grid gap-8 md:grid-cols-3">
        <Card className="glass-card card-copper hover:scale-105 transition-transform duration-300">
          <CardHeader className="pb-6">
            <CardTitle className="heading-secondary font-semibold text-foreground">Upload Job Listings</CardTitle>
            <CardDescription className="text-body text-muted-foreground">
              Upload a CSV file containing job listings to analyze.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <FileUpload onFileUpload={handleFileUpload} isLoading={isLoading} />
          </CardContent>
        </Card>

        <Card className="glass-card card-copper hover:scale-105 transition-transform duration-300">
          <CardHeader className="pb-6">
            <CardTitle className="heading-secondary font-semibold text-foreground">Model Performance</CardTitle>
            <CardDescription className="text-body text-muted-foreground">
              Current model metrics on validation data.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ModelPerformanceReal />
          </CardContent>
        </Card>

        <Card className="glass-card card-copper hover:scale-105 transition-transform duration-300">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-6">
            <div>
              <CardTitle className="heading-secondary font-semibold text-foreground">Quick Stats</CardTitle>
              <CardDescription className="text-body text-muted-foreground">
                Summary of detection results
              </CardDescription>
            </div>
            {results && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowResultInsights(true)}
                className="h-8 w-8 p-0 hover:bg-white/20"
                title="View Fraud Insights"
              >
                <TrendingUp className="h-4 w-4 text-gold" />
              </Button>
            )}
          </CardHeader>
          <CardContent>
            {results ? (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex flex-col">
                    <span className="text-sm font-medium text-muted-foreground">Total Jobs</span>
                    <span className="text-2xl font-bold text-gold">{results.total_jobs.toLocaleString()}</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-sm font-medium text-muted-foreground">High Risk ({">"}51%)</span>
                    <span className="text-2xl font-bold text-red-400">{highRiskJobs.length.toLocaleString()}</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-sm font-medium text-muted-foreground">Genuine</span>
                    <span className="text-2xl font-bold text-green-400">{results.genuine_count.toLocaleString()}</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-sm font-medium text-muted-foreground">Fraud Rate</span>
                    <span className="text-2xl font-bold text-red-400">{results.fraud_rate}%</span>
                  </div>
                </div>
                <div className="pt-4 border-t border-copper">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Avg Fraud Probability:</span>
                    <span className="font-medium text-accent">
                      {(Number.parseFloat(results.avg_fraud_probability) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Model Used:</span>
                    <span className="font-medium text-accent">{results.model_type}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex h-[120px] items-center justify-center">
                <p className="text-sm text-muted-foreground text-body">Upload a file to see stats</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Results Section - Better spacing */}
      {results && (
        <div className="glass-card card-copper rounded-lg p-8">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-6 bg-black/20 border border-copper mb-8">
              <TabsTrigger value="overview" className="data-[state=active]:bg-copper/20 data-[state=active]:text-gold">
                Overview
              </TabsTrigger>
              <TabsTrigger value="results" className="data-[state=active]:bg-copper/20 data-[state=active]:text-gold">
                All Results ({results.total_jobs})
              </TabsTrigger>
              <TabsTrigger value="highrisk" className="data-[state=active]:bg-copper/20 data-[state=active]:text-gold">
                High Risk ({highRiskJobs.length})
              </TabsTrigger>
              <TabsTrigger
                value="distribution"
                className="data-[state=active]:bg-copper/20 data-[state=active]:text-gold"
              >
                Distribution
              </TabsTrigger>
              <TabsTrigger
                value="suspicious"
                className="data-[state=active]:bg-copper/20 data-[state=active]:text-gold"
              >
                Top 10
              </TabsTrigger>
              <TabsTrigger value="insights" className="data-[state=active]:bg-copper/20 data-[state=active]:text-gold">
                Analysis
              </TabsTrigger>
            </TabsList>

            <div className="mt-8">
              <TabsContent value="overview">
                <div className="grid gap-8 md:grid-cols-2">
                  <Card className="glass-card-dark card-copper">
                    <CardHeader className="pb-6">
                      <CardTitle className="heading-secondary font-semibold text-gold">
                        Fraud Probability Distribution
                      </CardTitle>
                      <CardDescription className="text-body">
                        Histogram showing distribution of fraud probabilities across {results.total_jobs} job listings
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <FraudDistribution data={results.predictions} />
                    </CardContent>
                  </Card>
                  <Card className="glass-card-dark card-copper">
                    <CardHeader className="pb-6">
                      <CardTitle className="heading-secondary font-semibold text-gold">
                        Genuine vs Fraudulent Jobs
                      </CardTitle>
                      <CardDescription className="text-body">
                        Classification results: {results.genuine_count} genuine, {results.fraudulent_count} fraudulent
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <FraudPieChart genuine={results.genuine_count} fraudulent={results.fraudulent_count} />
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="results">
                <Card className="glass-card-dark card-copper">
                  <CardHeader className="pb-6">
                    <CardTitle className="heading-secondary font-semibold text-gold">
                      All Results ({results.total_jobs} jobs)
                    </CardTitle>
                    <CardDescription className="text-body">
                      Complete list of job listings with fraud predictions and probabilities
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResultsTable data={results.predictions} />
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="highrisk">
                <Card className="glass-card-dark card-copper">
                  <CardHeader className="pb-6">
                    <CardTitle className="heading-secondary font-semibold text-gold">
                      High Risk Jobs ({highRiskJobs.length} found)
                    </CardTitle>
                    <CardDescription className="text-body">
                      Job listings with fraud probability greater than 51% - requires immediate attention
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {highRiskJobs.length > 0 ? (
                      <div className="space-y-6">
                        <div className="p-4 bg-red-900/20 border border-red-500/30 rounded-lg">
                          <div className="flex items-center gap-2 text-red-300 font-medium">
                            <AlertTriangle className="h-4 w-4" />
                            Warning: {highRiskJobs.length} high-risk job(s) detected
                          </div>
                          <div className="text-sm text-red-200 mt-1">
                            These listings show strong indicators of potential fraud. Review carefully before
                            proceeding.
                          </div>
                        </div>
                        <ResultsTable data={highRiskJobs} />
                      </div>
                    ) : (
                      <div className="flex h-32 items-center justify-center text-muted-foreground">
                        <div className="text-center">
                          <Shield className="h-8 w-8 mx-auto mb-2 text-green-400" />
                          <p className="text-body">No high-risk jobs found</p>
                          <p className="text-sm text-body">All jobs have ≤51% fraud probability</p>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="distribution">
                <Card className="glass-card-dark card-copper">
                  <CardHeader className="pb-6">
                    <CardTitle className="heading-secondary font-semibold text-gold">
                      Fraud Probability Distribution
                    </CardTitle>
                    <CardDescription className="text-body">
                      Detailed histogram analysis of fraud probability distribution across all {results.total_jobs}{" "}
                      listings
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <FraudDistribution data={results.predictions} />
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="suspicious">
                <Card className="glass-card-dark card-copper">
                  <CardHeader className="pb-6">
                    <CardTitle className="heading-secondary font-semibold text-gold">
                      Top 10 Most Suspicious Listings
                    </CardTitle>
                    <CardDescription className="text-body">
                      Job listings ranked by highest fraud probability scores
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <TopSuspiciousListings data={results.predictions} />
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="insights">
                <Card className="glass-card-dark card-copper">
                  <CardHeader className="pb-6">
                    <CardTitle className="heading-secondary font-semibold text-gold">Fraud Analysis Summary</CardTitle>
                    <CardDescription className="text-body">
                      Key insights and patterns detected in your job listings dataset
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {resultInsights ? (
                      <div className="space-y-8">
                        {/* Risk Overview */}
                        <div className="grid gap-6 md:grid-cols-3">
                          <div className="p-6 bg-red-900/20 border border-red-500/30 rounded-lg">
                            <div className="flex items-center gap-2 text-red-300 font-medium mb-2">
                              <AlertTriangle className="h-4 w-4" />
                              High Risk
                            </div>
                            <div className="text-2xl font-bold text-red-400">{resultInsights.highRisk}</div>
                            <div className="text-sm text-red-300">{resultInsights.riskDistribution.high}% of total</div>
                            <div className="text-xs text-muted-foreground mt-1">Fraud probability {">"} 80%</div>
                          </div>

                          <div className="p-6 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
                            <div className="flex items-center gap-2 text-yellow-300 font-medium mb-2">
                              <TrendingUp className="h-4 w-4" />
                              Medium Risk
                            </div>
                            <div className="text-2xl font-bold text-yellow-400">{resultInsights.mediumRisk}</div>
                            <div className="text-sm text-yellow-300">
                              {resultInsights.riskDistribution.medium}% of total
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">Fraud probability 50-80%</div>
                          </div>

                          <div className="p-6 bg-green-900/20 border border-green-500/30 rounded-lg">
                            <div className="flex items-center gap-2 text-green-300 font-medium mb-2">
                              <Shield className="h-4 w-4" />
                              Low Risk
                            </div>
                            <div className="text-2xl font-bold text-green-400">{resultInsights.lowRisk}</div>
                            <div className="text-sm text-green-300">
                              {resultInsights.riskDistribution.low}% of total
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">Fraud probability {"<"} 50%</div>
                          </div>
                        </div>

                        {/* Top Fraud Indicators */}
                        <div className="space-y-6">
                          <h3 className="heading-secondary text-lg font-semibold gradient-text">
                            Top Fraud Indicators Detected
                          </h3>
                          <div className="grid gap-4 md:grid-cols-2">
                            {resultInsights.topKeywords.map((keyword, index) => (
                              <div
                                key={index}
                                className="flex items-center justify-between p-4 bg-red-900/20 rounded-lg border border-red-500/30"
                              >
                                <span className="font-medium text-red-300">"{keyword.text}"</span>
                                <span className="text-sm text-red-400">{keyword.value} occurrences</span>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Summary Statistics */}
                        <div className="p-6 bg-black/20 rounded-lg border border-copper">
                          <h3 className="heading-secondary text-lg font-semibold mb-6 gradient-text">
                            Analysis Summary
                          </h3>
                          <div className="grid gap-4 md:grid-cols-2">
                            <div className="flex justify-between">
                              <span className="text-body">Average Fraud Probability:</span>
                              <span className="font-medium text-accent">
                                {(resultInsights.avgProb * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-body">Overall Fraud Rate:</span>
                              <span className="font-medium text-accent">{resultInsights.fraudRate.toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-body">Total Jobs Analyzed:</span>
                              <span className="font-medium text-accent">
                                {resultInsights.totalJobs.toLocaleString()}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-body">Model Accuracy:</span>
                              <span className="font-medium text-accent">FigureForge-Anveshan</span>
                            </div>
                          </div>
                        </div>

                        {/* Recommendations */}
                        <div className="space-y-4">
                          <h3 className="heading-secondary text-lg font-semibold gradient-text">Recommendations</h3>
                          {resultInsights.highRisk > 0 && (
                            <div className="p-4 bg-red-900/20 border border-red-500/30 rounded-lg">
                              <div className="font-medium text-red-300">⚠️ Immediate Action Required</div>
                              <div className="text-sm text-red-200 mt-1 text-body">
                                {resultInsights.highRisk} job(s) flagged as high risk. Review these listings immediately
                                for potential fraud indicators.
                              </div>
                            </div>
                          )}
                          {resultInsights.mediumRisk > 0 && (
                            <div className="p-4 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
                              <div className="font-medium text-yellow-300">⚡ Additional Verification Needed</div>
                              <div className="text-sm text-yellow-200 mt-1 text-body">
                                {resultInsights.mediumRisk} job(s) require additional verification. Check for missing
                                information or suspicious patterns.
                              </div>
                            </div>
                          )}
                          <div className="p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg">
                            <div className="font-medium text-blue-300">💡 General Safety Tips</div>
                            <div className="text-sm text-blue-200 mt-1 text-body">
                              Always verify company information, be cautious of jobs requiring upfront payments, and
                              trust your instincts about suspicious offers.
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-8">
                        <p className="text-muted-foreground text-body">No analysis data available</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </div>
          </Tabs>
        </div>
      )}

      {/* Fraud Insights Modal */}
      {showResultInsights && resultInsights && (
        <Dialog open={showResultInsights} onOpenChange={setShowResultInsights}>
          <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto glass-card p-8">
            <DialogHeader className="mb-8">
              <div className="flex items-center justify-between">
                <div>
                  <DialogTitle className="heading-primary flex items-center gap-2 gradient-text">
                    <AlertTriangle className="h-5 w-5" />
                    Comprehensive Fraud Detection Insights
                  </DialogTitle>
                  <p className="text-sm text-muted-foreground mt-1 text-body">
                    Detailed analysis of {resultInsights.totalJobs} job listings processed
                  </p>
                </div>
                <Button variant="ghost" size="sm" onClick={() => setShowResultInsights(false)}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </DialogHeader>

            <div className="space-y-8">
              {/* Risk Overview Cards */}
              <div className="grid gap-6 md:grid-cols-3">
                <Card className="glass-card-dark border-red-500/30">
                  <CardHeader className="pb-4">
                    <CardTitle className="heading-secondary text-lg text-red-400 flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4" />
                      Critical Risk
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-red-400">{resultInsights.highRisk}</div>
                    <div className="text-sm text-red-300">{resultInsights.riskDistribution.high}% of total jobs</div>
                    <div className="text-xs text-muted-foreground mt-1">Fraud probability {">"} 80%</div>
                    <div className="text-xs text-red-400 mt-2 font-medium">Requires immediate review</div>
                  </CardContent>
                </Card>

                <Card className="glass-card-dark border-yellow-500/30">
                  <CardHeader className="pb-4">
                    <CardTitle className="heading-secondary text-lg text-yellow-400 flex items-center gap-2">
                      <TrendingUp className="h-4 w-4" />
                      Moderate Risk
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-yellow-400">{resultInsights.mediumRisk}</div>
                    <div className="text-sm text-yellow-300">
                      {resultInsights.riskDistribution.medium}% of total jobs
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">Fraud probability 50-80%</div>
                    <div className="text-xs text-yellow-400 mt-2 font-medium">Additional verification needed</div>
                  </CardContent>
                </Card>

                <Card className="glass-card-dark border-green-500/30">
                  <CardHeader className="pb-4">
                    <CardTitle className="heading-secondary text-lg text-green-400 flex items-center gap-2">
                      <Shield className="h-4 w-4" />
                      Low Risk
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-green-400">{resultInsights.lowRisk}</div>
                    <div className="text-sm text-green-300">{resultInsights.riskDistribution.low}% of total jobs</div>
                    <div className="text-xs text-muted-foreground mt-1">Fraud probability {"<"} 50%</div>
                    <div className="text-xs text-green-400 mt-2 font-medium">Generally safe to proceed</div>
                  </CardContent>
                </Card>
              </div>

              {/* Detailed Analysis */}
              <div className="grid gap-8 md:grid-cols-2">
                <Card className="glass-card-dark">
                  <CardHeader className="pb-6">
                    <CardTitle className="heading-secondary gradient-text">Fraud Indicators Found</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {resultInsights.topKeywords.map((keyword, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-4 bg-red-900/20 rounded-lg border border-red-500/30"
                      >
                        <span className="font-medium text-red-300">"{keyword.text}"</span>
                        <span className="text-sm text-red-400 font-medium">{keyword.value} times</span>
                      </div>
                    ))}
                  </CardContent>
                </Card>

                <Card className="glass-card-dark">
                  <CardHeader className="pb-6">
                    <CardTitle className="heading-secondary gradient-text">Statistical Summary</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex justify-between p-4 bg-black/20 rounded-lg">
                      <span className="text-body">Average Fraud Probability:</span>
                      <span className="font-bold text-accent">{(resultInsights.avgProb * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between p-4 bg-black/20 rounded-lg">
                      <span className="text-body">Overall Fraud Rate:</span>
                      <span className="font-bold text-accent">{resultInsights.fraudRate.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between p-4 bg-black/20 rounded-lg">
                      <span className="text-body">Jobs Analyzed:</span>
                      <span className="font-bold text-accent">{resultInsights.totalJobs.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between p-4 bg-black/20 rounded-lg">
                      <span className="text-body">Detection Model:</span>
                      <span className="font-bold text-accent">FigureForge-Anveshan</span>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Action Items */}
              <Card className="glass-card-dark">
                <CardHeader className="pb-6">
                  <CardTitle className="heading-secondary gradient-text">Recommended Actions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {resultInsights.highRisk > 0 && (
                    <div className="p-6 bg-red-900/20 border border-red-500/30 rounded-lg">
                      <div className="font-semibold text-red-300 flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4" />
                        Critical Priority
                      </div>
                      <div className="text-sm text-red-200 mt-2 text-body">
                        <strong>{resultInsights.highRisk} high-risk job(s)</strong> detected. These require immediate
                        manual review:
                      </div>
                      <ul className="text-sm text-red-200 mt-2 ml-4 list-disc text-body">
                        <li>Verify company legitimacy and contact information</li>
                        <li>Check for unrealistic salary promises or requirements</li>
                        <li>Look for requests for personal information or upfront payments</li>
                      </ul>
                    </div>
                  )}

                  {resultInsights.mediumRisk > 0 && (
                    <div className="p-6 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
                      <div className="font-semibold text-yellow-300 flex items-center gap-2">
                        <TrendingUp className="h-4 w-4" />
                        Medium Priority
                      </div>
                      <div className="text-sm text-yellow-200 mt-2 text-body">
                        <strong>{resultInsights.mediumRisk} medium-risk job(s)</strong> need additional verification
                        before proceeding.
                      </div>
                    </div>
                  )}

                  <div className="p-6 bg-blue-900/20 border border-blue-500/30 rounded-lg">
                    <div className="font-semibold text-blue-300 flex items-center gap-2">
                      <Shield className="h-4 w-4" />
                      General Best Practices
                    </div>
                    <div className="text-sm text-blue-200 mt-2 text-body">
                      Always research companies independently, never pay upfront fees, and trust your instincts about
                      suspicious offers.
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </DialogContent>
        </Dialog>
      )}

      {/* Model Insights Dialog */}
      {showModelInsights && <ModelInsights isOpen={showModelInsights} onClose={() => setShowModelInsights(false)} />}

      {/* About Dialog */}
      <AboutDialog isOpen={showAbout} onClose={() => setShowAbout(false)} />
    </div>
  )
}
