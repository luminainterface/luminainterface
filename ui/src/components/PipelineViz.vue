<template>
  <aside
    class="fixed bottom-4 left-1/2 -translate-x-1/2 bg-gray-800/80 rounded-xl p-3 flex gap-2 items-center shadow-lg"
  >
    <template v-for="step in steps" :key="step.name">
      <div
        :class="badgeClass(step.status)"
        class="px-3 py-1 rounded-full text-xs font-semibold text-white transition-colors duration-300"
      >
        {{ step.name }}
      </div>
    </template>

    <!-- Service badges -->
    <div class="ml-6 flex gap-1">
      <ServiceBadge
        v-for="svc in services"
        :key="svc.name"
        :svc="svc"
      />
    </div>

    <!-- Retry button -->
    <button
      v-if="hasError"
      class="ml-4 px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-xs font-semibold rounded-full transition-colors duration-200"
      @click="handleRetry"
    >
      Retry Failed Step
    </button>
  </aside>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import ServiceBadge from './ServiceBadge.vue'
import { useHealthStore } from '@/stores/health'
import { useDebugStore } from '@/stores/debug'

type Status = 'pending' | 'running' | 'ok' | 'error'

interface Step {
  name: string
  status: Status
}

const steps = ref<Step[]>([
  { name: 'Crawl', status: 'pending' },
  { name: 'Summarise', status: 'pending' },
  { name: 'QA', status: 'pending' }
])

const hasError = ref(false)
let eventSource: EventSource | null = null

const healthStore = useHealthStore()
const { services } = healthStore

const debugStore = useDebugStore()

function badgeClass(st: Status) {
  return {
    pending: 'bg-gray-600',
    running: 'bg-yellow-500 animate-pulse',
    ok: 'bg-green-500',
    error: 'bg-red-600'
  }[st]
}

const handleRetry = () => {
  // Re-emit the last error event with retry flag
  if (eventSource) {
    const retryEvent = new CustomEvent('retry', {
      detail: { retryFailed: true }
    })
    eventSource.dispatchEvent(retryEvent)
  }
}

onMounted(() => {
  // Start health polling
  healthStore.startPolling()

  // Connect to SSE
  eventSource = new EventSource('/api/masterchat/stream')
  
  eventSource.addEventListener('log', (e) => {
    const { agent, status } = JSON.parse(e.data)
    const map: Record<string, string> = {
      CrawlAgent: 'Crawl',
      SummariseAgent: 'Summarise',
      QAAgent: 'QA'
    }
    
    const step = steps.value.find(s => s.name === map[agent])
    if (step) {
      step.status = status === 'start' ? 'running'
        : status === 'end' ? 'ok'
        : 'error'
      
      hasError.value = status === 'error'

      // Send to debug console
      debugStore.addLog({
        channel: 'Pipeline',
        level: status === 'error' ? 'error' : 'info',
        message: `${agent} ${status}`,
        metadata: {
          step: map[agent],
          status
        }
      })
    }
  })

  // Connect to system logs
  const systemLogSource = new EventSource('/api/logs/system')
  systemLogSource.addEventListener('message', (e) => {
    const { level, message } = JSON.parse(e.data)
    debugStore.addLog({
      channel: 'Pipeline',
      level: level.toLowerCase(),
      message,
      raw: e.data
    })
  })

  // Connect to metrics updates
  const metricsSource = new EventSource('/api/metrics/updates')
  metricsSource.addEventListener('message', (e) => {
    const metrics = JSON.parse(e.data)
    debugStore.addLog({
      channel: 'Service',
      level: 'info',
      message: 'Metrics updated',
      metadata: metrics
    })
  })
})

onUnmounted(() => {
  healthStore.stopPolling()
  if (eventSource) {
    eventSource.close()
  }
})
</script> 