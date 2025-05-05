import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export type LogChannel = 'Pipeline' | 'Service' | 'Network' | 'Browser'
export type LogLevel = 'info' | 'warning' | 'error'

export interface LogEntry {
  id: string
  timestamp: number
  channel: LogChannel
  level: LogLevel
  message: string
  raw?: string
  metadata?: Record<string, any>
}

export const useDebugStore = defineStore('debug', () => {
  const logs = ref<LogEntry[]>([])
  const autoScroll = ref(true)
  const activeChannel = ref<LogChannel>('Pipeline')

  const filteredLogs = computed(() => {
    return logs.value.filter(log => log.channel === activeChannel.value)
  })

  const addLog = (entry: Omit<LogEntry, 'id' | 'timestamp'>) => {
    const newEntry: LogEntry = {
      ...entry,
      id: crypto.randomUUID(),
      timestamp: Date.now()
    }
    logs.value.push(newEntry)

    // Keep only last 1000 logs
    if (logs.value.length > 1000) {
      logs.value = logs.value.slice(-1000)
    }
  }

  const clearLogs = (channel?: LogChannel) => {
    if (channel) {
      logs.value = logs.value.filter(log => log.channel !== channel)
    } else {
      logs.value = []
    }
  }

  const setActiveChannel = (channel: LogChannel) => {
    activeChannel.value = channel
  }

  const toggleAutoScroll = () => {
    autoScroll.value = !autoScroll.value
  }

  // Initialize global error handlers
  const initErrorHandlers = () => {
    window.onerror = (message, source, lineno, colno, error) => {
      addLog({
        channel: 'Browser',
        level: 'error',
        message: `${message} at ${source}:${lineno}:${colno}`,
        raw: error?.stack
      })
    }

    window.onunhandledrejection = (event) => {
      addLog({
        channel: 'Browser',
        level: 'error',
        message: 'Unhandled Promise Rejection',
        raw: event.reason?.stack || event.reason?.message
      })
    }

    // Intercept fetch calls
    const originalFetch = window.fetch
    window.fetch = async (...args) => {
      const startTime = Date.now()
      try {
        const response = await originalFetch(...args)
        if (!response.ok) {
          addLog({
            channel: 'Network',
            level: 'error',
            message: `HTTP ${response.status} ${response.statusText}`,
            metadata: {
              url: args[0],
              method: args[1]?.method || 'GET',
              duration: Date.now() - startTime
            }
          })
        }
        return response
      } catch (error) {
        addLog({
          channel: 'Network',
          level: 'error',
          message: 'Network request failed',
          metadata: {
            url: args[0],
            method: args[1]?.method || 'GET',
            duration: Date.now() - startTime
          },
          raw: error instanceof Error ? error.stack : String(error)
        })
        throw error
      }
    }
  }

  return {
    logs,
    filteredLogs,
    autoScroll,
    activeChannel,
    addLog,
    clearLogs,
    setActiveChannel,
    toggleAutoScroll,
    initErrorHandlers
  }
}) 