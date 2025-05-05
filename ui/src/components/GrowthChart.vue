<template>
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
    <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
      Graph Growth Metrics
    </h3>
    <div class="h-64">
      <Line
        v-if="chartData"
        :data="chartData"
        :options="chartOptions"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'
import eventBus, { EventType, EventPayload } from '../store/eventBus'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

interface MetricPoint {
  timestamp: number
  nodes: number
  edges: number
  fractalDim: number
}

const metrics = ref<MetricPoint[]>([])
const maxPoints = 100

const chartData = ref({
  labels: [] as string[],
  datasets: [
    {
      label: 'Nodes',
      data: [] as number[],
      borderColor: '#60A5FA',
      tension: 0.4
    },
    {
      label: 'Edges',
      data: [] as number[],
      borderColor: '#34D399',
      tension: 0.4
    },
    {
      label: 'Fractal Dimension',
      data: [] as number[],
      borderColor: '#F87171',
      tension: 0.4
    }
  ]
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: {
    duration: 0
  },
  scales: {
    x: {
      type: 'time',
      time: {
        unit: 'second',
        displayFormats: {
          second: 'HH:mm:ss'
        }
      },
      grid: {
        color: 'rgba(156, 163, 175, 0.1)'
      }
    },
    y: {
      beginAtZero: true,
      grid: {
        color: 'rgba(156, 163, 175, 0.1)'
      }
    }
  },
  plugins: {
    legend: {
      position: 'top' as const
    }
  }
}

const updateChart = () => {
  if (metrics.value.length === 0) return

  const labels = metrics.value.map(m => new Date(m.timestamp))
  const nodeData = metrics.value.map(m => m.nodes)
  const edgeData = metrics.value.map(m => m.edges)
  const fractalData = metrics.value.map(m => m.fractalDim)

  chartData.value = {
    labels,
    datasets: [
      {
        ...chartData.value.datasets[0],
        data: nodeData
      },
      {
        ...chartData.value.datasets[1],
        data: edgeData
      },
      {
        ...chartData.value.datasets[2],
        data: fractalData
      }
    ]
  }
}

const handleMetricUpdate = (payload: EventPayload) => {
  if (!payload.metric) return

  const now = Date.now()
  const newMetric: MetricPoint = {
    timestamp: now,
    nodes: payload.metric.type === 'nodes' ? payload.metric.value : metrics.value[metrics.value.length - 1]?.nodes || 0,
    edges: payload.metric.type === 'edges' ? payload.metric.value : metrics.value[metrics.value.length - 1]?.edges || 0,
    fractalDim: payload.metric.type === 'fractal_dim' ? payload.metric.value : metrics.value[metrics.value.length - 1]?.fractalDim || 0
  }

  metrics.value.push(newMetric)
  if (metrics.value.length > maxPoints) {
    metrics.value.shift()
  }

  updateChart()
}

onMounted(() => {
  eventBus.on('metric.update', handleMetricUpdate)
})

onUnmounted(() => {
  eventBus.off('metric.update', handleMetricUpdate)
})
</script> 