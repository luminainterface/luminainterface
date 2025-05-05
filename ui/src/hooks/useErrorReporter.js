import { ref, onMounted, onUnmounted } from 'vue'

export function useErrorReporter() {
  const errors = ref([])
  const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8101'

  const connect = () => {
    const ws = new WebSocket(wsUrl)
    
    ws.onopen = () => {
      console.log('Error reporter connected')
    }
    
    ws.onclose = () => {
      // Attempt to reconnect after 5s
      setTimeout(connect, 5000)
    }
  }

  const reportError = (error) => {
    errors.value.push({
      message: error.message,
      timestamp: new Date().toISOString()
    })
    // Keep only last 100 errors
    if (errors.value.length > 100) {
      errors.value.shift()
    }
  }

  const handleError = (event) => {
    reportError(event.error || new Error(event.message))
  }

  const handleUnhandledRejection = (event) => {
    reportError(event.reason)
  }

  onMounted(() => {
    connect()
    window.addEventListener('error', handleError)
    window.addEventListener('unhandledrejection', handleUnhandledRejection)
  })

  onUnmounted(() => {
    window.removeEventListener('error', handleError)
    window.removeEventListener('unhandledrejection', handleUnhandledRejection)
  })

  return {
    errors,
    reportError
  }
} 