"use client"

import { PieChart, Pie, Cell, ResponsiveContainer, Legend } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

interface FraudPieChartProps {
  genuine: number
  fraudulent: number
}

export default function FraudPieChart({ genuine, fraudulent }: FraudPieChartProps) {
  const data = [
    { name: "Genuine", value: genuine },
    { name: "Fraudulent", value: fraudulent },
  ]

  const COLORS = ["#10b981", "#ef4444"]

  return (
    <ChartContainer
      config={{
        genuine: {
          label: "Genuine",
          color: "#10b981",
        },
        fraudulent: {
          label: "Fraudulent",
          color: "#ef4444",
        },
      }}
      className="h-[300px]"
    >
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
            nameKey="name"
            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Legend verticalAlign="bottom" height={36} />
          <ChartTooltip content={<ChartTooltipContent />} />
        </PieChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}
