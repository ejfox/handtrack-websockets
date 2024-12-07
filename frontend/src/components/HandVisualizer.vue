<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import * as d3 from 'd3'
import type { HandLandmark } from '../composables/useHands'

const props = defineProps<{
  landmarks: HandLandmark[]
  confidence?: number
}>()

// Canvas refs
const svgRef = ref<SVGSVGElement>()
const width = 300
const height = 300

// Hand connections for skeleton visualization
const handConnections = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20]
]

// Setup D3 scales
const xScale = d3.scaleLinear().domain([-0.3, 0.3]).range([0, width])
const yScale = d3.scaleLinear().domain([-0.3, 0.3]).range([height, 0])

// Draw hand skeleton
const drawHand = () => {
  if (!svgRef.value || !props.landmarks) return

  const svg = d3.select(svgRef.value)
  svg.selectAll('*').remove()

  // Draw connections
  handConnections.forEach(([start, end]) => {
    const startPoint = props.landmarks[start]
    const endPoint = props.landmarks[end]

    svg.append('line')
      .attr('x1', xScale(startPoint.x))
      .attr('y1', yScale(startPoint.y))
      .attr('x2', xScale(endPoint.x))
      .attr('y2', yScale(endPoint.y))
      .attr('stroke', 'rgba(74, 222, 128, 0.6)')
      .attr('stroke-width', 2)
  })

  // Draw landmarks
  svg.selectAll('circle')
    .data(props.landmarks)
    .enter()
    .append('circle')
    .attr('cx', d => xScale(d.x))
    .attr('cy', d => yScale(d.y))
    .attr('r', 4)
    .attr('fill', 'rgb(74, 222, 128)')

  // Draw confidence indicator if available
  if (props.confidence) {
    svg.append('text')
      .attr('x', 10)
      .attr('y', 20)
      .attr('fill', 'rgb(74, 222, 128)')
      .text(`Confidence: ${(props.confidence * 100).toFixed(1)}%`)
  }
}

watch(() => props.landmarks, drawHand, { deep: true })
onMounted(drawHand)
</script>

<template>
  <svg ref="svgRef" :width="width" :height="height" class="bg-black/40 rounded-lg" />
</template>