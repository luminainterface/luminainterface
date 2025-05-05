import { ref, onMounted, onUnmounted } from 'vue'
import mitt from 'mitt'
import eventBus, { EventType, EventPayload } from '../store/eventBus'

interface GraphSocketOptions {
  wsUrl?: string
  sseUrl?: string
  maxRetries?: number
  retryDelay?: number
}

export function useGraphSocket(options: GraphSocketOptions = {}) {
  const {
    wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8101/ws',
    sseUrl = import.meta.env.VITE_SSE_URL || `${(import.meta.env.VITE_API_URL || 'http://localhost:8201').replace(/\/$/, '')}/sse`,
    maxRetries = 5,
    retryDelay = 1000
  } = options

  const wsRef = ref<WebSocket | null>(null)
  const eventSource = ref<EventSource | null>(null)
  const isConnected = ref(false)
  const retryCount = ref(0)
  const useSSE = ref(false)

  const connectWebSocket = () => {
    if (wsRef.value && wsRef.value.readyState === WebSocket.OPEN) return

    wsRef.value = new WebSocket(wsUrl)

    wsRef.value.onopen = () => {
      isConnected.value = true
      retryCount.value = 0
      useSSE.value = false
      console.log('WebSocket connected')
    }

    wsRef.value.onclose = () => {
      isConnected.value = false
      console.log('WebSocket closed')
      if (retryCount.value < maxRetries) {
        retryCount.value++
        setTimeout(connectWebSocket, retryDelay)
      } else if (!useSSE.value) {
        console.log('Switching to SSE fallback')
        useSSE.value = true
        connectSSE()
      }
    }

    wsRef.value.onerror = () => {
      console.warn('WebSocket error')
    }

    wsRef.value.onmessage = (ev) => {
      try {
        const evt = JSON.parse(ev.data)
        eventBus.emit(evt.type as EventType, evt.payload as EventPayload)
      } catch (e) {
        console.error('Bad WS payload', e)
      }
    }
  }

  const connectSSE = () => {
    if (eventSource.value) return

    eventSource.value = new EventSource(sseUrl)

    eventSource.value.onopen = () => {
      isConnected.value = true
      console.log('SSE connected')
    }

    eventSource.value.onerror = () => {
      isConnected.value = false
      eventSource.value?.close()
      eventSource.value = null
    }

    // Forward SSE events to the event bus
    eventSource.value.onmessage = (event) => {
      try {
        const { type, payload } = JSON.parse(event.data)
        eventBus.emit(type as EventType, payload as EventPayload)
      } catch (error) {
        console.error('Failed to parse SSE message:', error)
      }
    }
  }

  const disconnect = () => {
    if (wsRef.value) {
      wsRef.value.close()
      wsRef.value = null
    }
    if (eventSource.value) {
      eventSource.value.close()
      eventSource.value = null
    }
    isConnected.value = false
  }

  onMounted(() => {
    connectWebSocket()
  })

  onUnmounted(() => {
    disconnect()
  })

  return {
    isConnected,
    useSSE,
    disconnect
  }
} 