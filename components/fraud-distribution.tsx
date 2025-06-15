"use client"

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

interface FraudDistributionProps {
  data: Array<{
    fraud_probability: number
  }>
}

export default function FraudDistribution({ data }: FraudDistributionProps) {
  // Create histogram data
  const createHistogramData = () => {
    const bins = 10
    const histogramData = Array(bins)
      .fill(0)
      .map((_, i) => ({
        range: `${i * 10}%-${(i + 1) * 10}%`,
        count: 0,
        min: i / 10,
        max: (i + 1) / 10,
      }))

    data.forEach((item) => {
      const prob = item.fraud_probability
      const binIndex = Math.min(Math.floor(prob * 10), 9)
      histogramData[binIndex].count++
    })

    return histogramData
  }

  const histogramData = createHistogramData()

  return (
    <ChartContainer
      config={{
        count: {
          label: "Count",
          color: "hsl(var(--chart-1))",
        },
      }}
      className="h-[300px]"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={histogramData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="range" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
          <YAxis tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Bar dataKey="count" fill="var(--color-count)" radius={[4, 4, 0, 0]} barSize={30} />
        </BarChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}
