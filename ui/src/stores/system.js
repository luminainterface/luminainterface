import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useSystemStore = defineStore('system', () => {
  // Placeholder for future native WebSocket integration
  // const ws = new WebSocket(import.meta.env.VITE_WS_URL || 'ws://localhost:8101/ws')
  const metrics = ref({})
  const events = ref([])
  const isConnected = ref(false)

  // Add native WebSocket logic here if needed

  const getMetrics = computed(() => metrics.value)
  const getEvents = computed(() => events.value)
  const getConnectionStatus = computed(() => isConnected.value)

  return {
    // initializeWebSocket,
    getMetrics,
    getEvents,
    getConnectionStatus
  }
}) 