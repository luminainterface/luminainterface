<template>
  <div class="wiki-qa">
    <div class="input-group">
      <input
        v-model="question"
        type="text"
        placeholder="Ask a question about any topic..."
        class="input input-bordered w-full"
        @keyup.enter="submitQuestion"
      />
      <button
        class="btn btn-primary"
        :disabled="isLoading"
        @click="submitQuestion"
      >
        <span v-if="isLoading" class="loading loading-spinner"></span>
        Ask Wiki
      </button>
    </div>

    <div v-if="logs.length > 0" class="logs-panel">
      <div class="logs-header">
        <h3 class="text-lg font-semibold">Planner Logs</h3>
        <button class="btn btn-ghost btn-sm" @click="clearLogs">Clear</button>
      </div>
      <div class="logs-content">
        <div
          v-for="(log, index) in logs"
          :key="index"
          class="log-entry"
          :class="log.type"
        >
          <span class="timestamp">{{ formatTimestamp(log.timestamp) }}</span>
          <span class="message">{{ log.message }}</span>
        </div>
      </div>
    </div>

    <div v-if="answer" class="answer-bubble">
      <div class="answer-content">
        {{ answer }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
import eventBus from '@/store/eventBus'

interface LogEntry {
  timestamp: number
  message: string
  type: 'info' | 'warning' | 'error'
}

const question = ref('')
const isLoading = ref(false)
const logs = ref<LogEntry[]>([])
const answer = ref('')
let eventSource: EventSource | null = null

const formatTimestamp = (timestamp: number) => {
  return new Date(timestamp).toLocaleTimeString()
}

const clearLogs = () => {
  logs.value = []
}

const submitQuestion = async () => {
  if (!question.value.trim() || isLoading.value) return

  isLoading.value = true
  logs.value = []
  answer.value = ''

  try {
    // Start SSE connection
    eventSource = new EventSource('/api/masterchat/stream')
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data)
      logs.value.push({
        timestamp: Date.now(),
        message: data.message,
        type: data.type || 'info'
      })
    }

    // Submit question
    const response = await axios.post('/api/masterchat/plan', {
      mode: 'wiki_qa',
      question: question.value
    })

    answer.value = response.data.message
  } catch (error) {
    console.error('Error submitting question:', error)
    logs.value.push({
      timestamp: Date.now(),
      message: 'Error: Failed to get answer',
      type: 'error'
    })
  } finally {
    isLoading.value = false
    if (eventSource) {
      eventSource.close()
      eventSource = null
    }
  }
}

onUnmounted(() => {
  if (eventSource) {
    eventSource.close()
  }
})
</script>

<style scoped>
.wiki-qa {
  @apply p-4 space-y-4;
}

.input-group {
  @apply flex gap-2;
}

.logs-panel {
  @apply bg-base-200 rounded-lg p-4;
}

.logs-header {
  @apply flex justify-between items-center mb-2;
}

.logs-content {
  @apply max-h-48 overflow-y-auto space-y-1;
}

.log-entry {
  @apply text-sm font-mono;
}

.log-entry .timestamp {
  @apply text-gray-500 mr-2;
}

.log-entry.info {
  @apply text-info;
}

.log-entry.warning {
  @apply text-warning;
}

.log-entry.error {
  @apply text-error;
}

.answer-bubble {
  @apply bg-primary text-primary-content rounded-lg p-4 mt-4;
}

.answer-content {
  @apply whitespace-pre-wrap;
}
</style> 