<template>
  <div class="relative w-full h-full">
    <svg ref="svgRef" class="w-full h-full"></svg>
    <div v-if="breadcrumbs.length > 1" 
         class="absolute top-4 left-4 flex space-x-2">
      <button v-for="(crumb, index) in breadcrumbs.slice(0, -1)" 
              :key="crumb.id"
              @click="navigateTo(index)"
              class="px-2 py-1 text-sm bg-white dark:bg-gray-800 rounded shadow hover:bg-gray-50 dark:hover:bg-gray-700">
        {{ crumb.name }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as d3 from 'd3'
import axios from 'axios'
import eventBus from '../store/eventBus'
import { useRouter } from 'vue-router'

interface Node {
  id: string
  name: string
  value?: number
  children?: Node[]
}

const svgRef = ref<SVGElement | null>(null)
const hierarchy = ref<Node | null>(null)
const breadcrumbs = ref<Node[]>([])
const zoom = ref(d3.zoom<SVGElement, unknown>())
const currentTransform = ref<d3.ZoomTransform | null>(null)
const router = useRouter()

const fetchHierarchy = async () => {
  try {
    const base = import.meta.env.VITE_API_URL || 'http://localhost:8201'
    const response = await axios.get(`${base}/hierarchy`)
    hierarchy.value = response.data
    breadcrumbs.value = [hierarchy.value]
    renderHierarchy()
  } catch (error) {
    console.error('Failed to fetch hierarchy:', error)
  }
}

const renderHierarchy = () => {
  if (!svgRef.value || !hierarchy.value) return

  const width = svgRef.value.clientWidth
  const height = svgRef.value.clientHeight

  // Clear previous content
  d3.select(svgRef.value).selectAll('*').remove()

  // Create pack layout
  const pack = d3.pack<Node>()
    .size([width, height])
    .padding(3)

  // Create hierarchy
  const root = d3.hierarchy(hierarchy.value)
    .sum(d => d.value || 1)
    .sort((a, b) => (b.value || 0) - (a.value || 0))

  const packed = pack(root)

  // Create zoom behavior
  zoom.value = d3.zoom<SVGElement, unknown>()
    .scaleExtent([1, 8])
    .on('zoom', (event) => {
      currentTransform.value = event.transform
      g.attr('transform', event.transform)
    })

  d3.select(svgRef.value).call(zoom.value)

  // Create group for all elements
  const g = d3.select(svgRef.value)
    .append('g')

  // Draw circles
  const node = g.selectAll('g')
    .data(packed.descendants())
    .join('g')
    .attr('transform', d => `translate(${d.x},${d.y})`)

  node.append('circle')
    .attr('r', d => d.r)
    .attr('fill', d => d.data.children ? '#4B5563' : '#6B7280')
    .attr('stroke', '#1F2937')
    .attr('stroke-width', 1)
    .style('cursor', d => d.data.children ? 'pointer' : 'default')
    .on('click', (event, d) => {
      if (d.data.children) {
        zoomToNode(d)
        breadcrumbs.value.push(d.data)
      } else if (d.data.id) {
        router.push(`/graph/${d.data.id}`)
      }
    })

  node.append('text')
    .attr('dy', '0.3em')
    .attr('text-anchor', 'middle')
    .text(d => d.data.name)
    .style('font-size', d => `${Math.min(2 * d.r, 14)}px`)
    .style('fill', 'white')
    .style('pointer-events', 'none')
}

const zoomToNode = (node: d3.HierarchyCircularNode<Node>) => {
  if (!svgRef.value || !zoom.value) return

  const width = svgRef.value.clientWidth
  const height = svgRef.value.clientHeight

  const transform = d3.zoomIdentity
    .translate(width / 2, height / 2)
    .scale(0.5)
    .translate(-node.x, -node.y)

  d3.select(svgRef.value)
    .transition()
    .duration(750)
    .call(zoom.value.transform, transform)
}

const navigateTo = (index: number) => {
  breadcrumbs.value = breadcrumbs.value.slice(0, index + 1)
  if (index === 0) {
    renderHierarchy()
  } else {
    const node = d3.select(svgRef.value)
      .selectAll('g')
      .data()
      .find((d: any) => d.data.id === breadcrumbs.value[index].id)
    if (node) {
      zoomToNode(node)
    }
  }
}

// Handle window resize
const handleResize = () => {
  renderHierarchy()
}

onMounted(() => {
  fetchHierarchy()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
})

// Handle ESC key
const handleKeyDown = (event: KeyboardEvent) => {
  if (event.key === 'Escape' && breadcrumbs.value.length > 1) {
    navigateTo(0)
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeyDown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyDown)
})
</script> 