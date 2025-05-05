import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { io } from 'socket.io-client'

export const useSystemStore = defineStore('system', () => {
  const socket = ref(null)
  const metrics = ref({})
  const events = ref([])
  const isConnected = ref(false)

  const initializeWebSocket = () => {
    if (socket.value) return // Prevent multiple initializations
    
    socket.value = io(import.meta.env.VITE_WS_URL || 'ws://localhost:8101', {
      transports: ['websocket'],
      autoConnect: true
    })

    socket.value.on('connect', () => {
      isConnected.value = true
      console.log('WebSocket connected')
    })

    socket.value.on('disconnect', () => {
      isConnected.value = false
      console.log('WebSocket disconnected')
    })

    socket.value.on('metrics', (data) => {
      metrics.value = data
    })

    socket.value.on('event', (event) => {
      events.value.unshift(event)
      if (events.value.length > 100) {
        events.value.pop()
      }
    })
  }

  const getMetrics = computed(() => metrics.value)
  const getEvents = computed(() => events.value)
  const getConnectionStatus = computed(() => isConnected.value)

  return {
    initializeWebSocket,
    getMetrics,
    getEvents,
    getConnectionStatus
  }
}) 