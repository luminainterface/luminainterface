import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'

export interface Service {
  name: string
  status: 'ok' | 'down' | 'warn' | 'unknown'
  latency?: number
}

export const useHealthStore = defineStore('health', () => {
  const services = ref<Service[]>([
    { name: 'API', status: 'unknown' },
    { name: 'Event-Mux', status: 'unknown' },
    { name: 'Redis', status: 'unknown' },
    { name: 'MasterChat', status: 'unknown' }
  ])

  let pollInterval: number | null = null

  const updateServiceStatus = async () => {
    try {
      const response = await axios.get('/api/health')
      const healthData = response.data

      services.value = services.value.map(service => {
        const status = healthData[service.name.toLowerCase()]
        return {
          ...service,
          status: status?.status || 'unknown',
          latency: status?.latency
        }
      })
    } catch (error) {
      console.error('Failed to fetch health status:', error)
      services.value = services.value.map(service => ({
        ...service,
        status: 'unknown'
      }))
    }
  }

  const startPolling = () => {
    if (pollInterval) return
    updateServiceStatus() // Initial update
    pollInterval = window.setInterval(updateServiceStatus, 30000) // Poll every 30s
  }

  const stopPolling = () => {
    if (pollInterval) {
      clearInterval(pollInterval)
      pollInterval = null
    }
  }

  return {
    services,
    startPolling,
    stopPolling
  }
}) 