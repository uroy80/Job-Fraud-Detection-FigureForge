"use client"

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

interface JobListing {
  id: string
  job_id?: string
  title: string
  company: string
  location: string
  fraud_probability: number
  prediction: "genuine" | "fraudulent"
}

interface TopSuspiciousListingsProps {
  data: JobListing[]
}

export default function TopSuspiciousListings({ data }: TopSuspiciousListingsProps) {
  // Sort by fraud probability and take top 10
  const topSuspicious = [...data].sort((a, b) => b.fraud_probability - a.fraud_probability).slice(0, 10)

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Rank</TableHead>
            <TableHead>Job ID</TableHead>
            <TableHead>Job Title</TableHead>
            <TableHead>Company</TableHead>
            <TableHead>Location</TableHead>
            <TableHead className="text-right">Fraud Probability</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {topSuspicious.map((job, index) => (
            <TableRow key={job.id}>
              <TableCell className="font-medium">{index + 1}</TableCell>
              <TableCell className="font-mono text-sm">{job.job_id || job.id}</TableCell>
              <TableCell className="max-w-xs truncate">{job.title}</TableCell>
              <TableCell className="max-w-xs truncate">{job.company}</TableCell>
              <TableCell className="max-w-xs truncate">{job.location}</TableCell>
              <TableCell className="text-right">
                <Badge variant="outline" className="bg-red-500 text-white">
                  {(job.fraud_probability * 100).toFixed(2)}%
                </Badge>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
