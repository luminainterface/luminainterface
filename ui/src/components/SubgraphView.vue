<template>
  <div class="relative w-full h-full">
    <canvas ref="canvasRef" class="w-full h-full"></canvas>
    <div v-if="loading" 
         class="absolute inset-0 flex items-center justify-center bg-gray-900/50">
      <div class="text-white">Loading subgraph...</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as d3 from 'd3'
import axios from 'axios'
import eventBus from '../store/eventBus'

interface Node {
  id: string
  label: string
  type: string
  x?: number
  y?: number
  vx?: number
  vy?: number
}

interface Link {
  source: string
  target: string
  type: string
}

interface Subgraph {
  nodes: Node[]
  links: Link[]
}

const props = defineProps<{
  subgraphId: string
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const ctx = ref<CanvasRenderingContext2D | null>(null)
const simulation = ref<d3.Simulation<Node, undefined> | null>(null)
const subgraph = ref<Subgraph | null>(null)
const loading = ref(false)

const fetchSubgraph = async () => {
  if (!props.subgraphId) return
  
  loading.value = true
  try {
    const response = await axios.get(`/api/subgraph/${props.subgraphId}`)
    subgraph.value = response.data
    initializeSimulation()
  } catch (error) {
    console.error('Failed to fetch subgraph:', error)
  } finally {
    loading.value = false
  }
}

const initializeSimulation = () => {
  if (!subgraph.value || !canvasRef.value) return

  const width = canvasRef.value.clientWidth
  const height = canvasRef.value.clientHeight

  // Initialize canvas context
  ctx.value = canvasRef.value.getContext('2d')
  if (!ctx.value) return

  // Create force simulation
  simulation.value = d3.forceSimulation<Node>(subgraph.value.nodes)
    .force('link', d3.forceLink<Node, Link>(subgraph.value.links)
      .id(d => d.id)
      .distance(50))
    .force('charge', d3.forceManyBody().strength(-100))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(30))

  // Start animation
  requestAnimationFrame(animate)
}

const animate = () => {
  if (!ctx.value || !simulation.value || !subgraph.value) return

  const width = canvasRef.value!.clientWidth
  const height = canvasRef.value!.clientHeight

  // Clear canvas
  ctx.value.clearRect(0, 0, width, height)

  // Draw links
  ctx.value.strokeStyle = '#4B5563'
  ctx.value.lineWidth = 1
  subgraph.value.links.forEach(link => {
    const source = link.source as Node
    const target = link.target as Node
    if (source.x && source.y && target.x && target.y) {
      ctx.value.beginPath()
      ctx.value.moveTo(source.x, source.y)
      ctx.value.lineTo(target.x, target.y)
      ctx.value.stroke()
    }
  })

  // Draw nodes
  subgraph.value.nodes.forEach(node => {
    if (node.x && node.y) {
      // Draw node circle
      ctx.value.beginPath()
      ctx.value.arc(node.x, node.y, 15, 0, 2 * Math.PI)
      ctx.value.fillStyle = '#6B7280'
      ctx.value.fill()
      ctx.value.strokeStyle = '#1F2937'
      ctx.value.lineWidth = 2
      ctx.value.stroke()

      // Draw node label
      ctx.value.fillStyle = 'white'
      ctx.value.font = '12px sans-serif'
      ctx.value.textAlign = 'center'
      ctx.value.textBaseline = 'middle'
      ctx.value.fillText(node.label, node.x, node.y)
    }
  })

  // Continue animation
  requestAnimationFrame(animate)
}

// Handle window resize
const handleResize = () => {
  if (simulation.value) {
    const width = canvasRef.value!.clientWidth
    const height = canvasRef.value!.clientHeight
    simulation.value.force('center', d3.forceCenter(width / 2, height / 2))
    simulation.value.alpha(0.3).restart()
  }
}

watch(() => props.subgraphId, () => {
  fetchSubgraph()
})

onMounted(() => {
  fetchSubgraph()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  if (simulation.value) {
    simulation.value.stop()
  }
  window.removeEventListener('resize', handleResize)
})
</script> 