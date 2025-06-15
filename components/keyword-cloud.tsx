"use client"

import { useEffect, useRef } from "react"
import * as d3 from "d3"
import cloud from "d3-cloud"

interface KeywordCloudProps {
  data: Array<{
    text: string
    value: number
  }>
}

export default function KeywordCloud({ data }: KeywordCloudProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return

    const width = 500
    const height = 300

    // Clear previous content
    d3.select(svgRef.current).selectAll("*").remove()

    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${width / 2},${height / 2})`)

    // Color scale
    const color = d3.scaleOrdinal(d3.schemeCategory10)

    // Font size scale
    const size = d3
      .scaleLinear()
      .domain([0, d3.max(data, (d) => d.value) || 1])
      .range([10, 50])

    // Generate word cloud
    cloud()
      .size([width, height])
      .words(data.map((d) => ({ text: d.text, size: size(d.value) })))
      .padding(5)
      .rotate(() => 0)
      .font("Arial")
      .fontSize((d) => d.size as number)
      .on("end", draw)
      .start()

    function draw(words: any[]) {
      svg
        .selectAll("text")
        .data(words)
        .enter()
        .append("text")
        .style("font-size", (d) => `${d.size}px`)
        .style("font-family", "Arial")
        .style("fill", (_, i) => color(i.toString()))
        .attr("text-anchor", "middle")
        .attr("transform", (d) => `translate(${d.x},${d.y})`)
        .text((d) => d.text)
    }
  }, [data])

  return (
    <div className="flex justify-center">
      <svg ref={svgRef} width="500" height="300" />
    </div>
  )
}
