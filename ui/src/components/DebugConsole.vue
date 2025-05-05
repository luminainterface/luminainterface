<template>
  <div
    ref="consoleRef"
    :class="[
      'debug-console bg-gray-900 text-gray-100 shadow-xl transition-all duration-300',
      isDocked ? 'fixed bottom-0 right-0 w-96 h-60' : 'absolute w-96 h-60'
    ]"
    :style="!isDocked ? { top: `${position.y}px`, left: `${position.x}px` } : {}"
  >
    <!-- Header -->
    <div
      class="flex items-center justify-between px-4 py-2 bg-gray-800 cursor-move"
      @mousedown="startDrag"
    >
      <div class="flex items-center space-x-2">
        <h3 class="text-sm font-semibold">Debug Console</h3>
        <button
          class="p-1 hover:bg-gray-700 rounded"
          @click="toggleDock"
          :title="isDocked ? 'Undock' : 'Dock'"
        >
          <component :is="isDocked ? 'ArrowsPointingOutIcon' : 'ArrowsPointingInIcon'" class="w-4 h-4" />
        </button>
      </div>
      <div class="flex items-center space-x-2">
        <button
          class="p-1 hover:bg-gray-700 rounded"
          @click="toggleAutoScroll"
          :title="autoScroll ? 'Pause Auto-scroll' : 'Enable Auto-scroll'"
        >
          <component :is="autoScroll ? 'PauseIcon' : 'PlayIcon'" class="w-4 h-4" />
        </button>
        <button
          class="p-1 hover:bg-gray-700 rounded"
          @click="clearLogs"
          title="Clear Logs"
        >
          <TrashIcon class="w-4 h-4" />
        </button>
      </div>
    </div>

    <!-- Channel Tabs -->
    <div class="flex border-b border-gray-700">
      <button
        v-for="channel in channels"
        :key="channel"
        class="px-3 py-1 text-xs font-medium"
        :class="[
          activeChannel === channel
            ? 'bg-gray-800 text-white'
            : 'text-gray-400 hover:text-white hover:bg-gray-800'
        ]"
        @click="setActiveChannel(channel)"
      >
        {{ channel }}
      </button>
    </div>

    <!-- Log Content -->
    <div
      ref="logContentRef"
      class="overflow-y-auto h-[calc(100%-4rem)] font-mono text-xs p-2"
      @scroll="handleScroll"
    >
      <div
        v-for="log in filteredLogs"
        :key="log.id"
        class="log-entry mb-1"
        :class="{
          'text-blue-400': log.level === 'info',
          'text-yellow-400': log.level === 'warning',
          'text-red-400': log.level === 'error'
        }"
      >
        <div class="flex items-start">
          <span class="text-gray-500 mr-2">{{ formatTimestamp(log.timestamp) }}</span>
          <span class="flex-1">{{ log.message }}</span>
          <button
            v-if="log.raw"
            class="ml-2 p-1 hover:bg-gray-800 rounded"
            @click="copyToClipboard(log.raw)"
            title="Copy to clipboard"
          >
            <ClipboardIcon class="w-3 h-3" />
          </button>
        </div>
        <div v-if="log.metadata" class="ml-6 text-gray-500">
          {{ formatMetadata(log.metadata) }}
        </div>
        <pre v-if="log.raw" class="ml-6 mt-1 text-gray-500 whitespace-pre-wrap">{{ log.raw }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import {
  ArrowsPointingOutIcon,
  ArrowsPointingInIcon,
  PauseIcon,
  PlayIcon,
  TrashIcon,
  ClipboardIcon
} from '@heroicons/vue/24/outline'
import { useDebugStore, type LogChannel } from '@/stores/debug'

const channels: LogChannel[] = ['Pipeline', 'Service', 'Network', 'Browser']

const consoleRef = ref<HTMLElement | null>(null)
const logContentRef = ref<HTMLElement | null>(null)
const isDocked = ref(true)
const position = ref({ x: 0, y: 0 })
const isDragging = ref(false)
const dragOffset = ref({ x: 0, y: 0 })

const debugStore = useDebugStore()
const {
  filteredLogs,
  autoScroll,
  activeChannel,
  addLog,
  clearLogs,
  setActiveChannel,
  toggleAutoScroll,
  initErrorHandlers
} = debugStore

const formatTimestamp = (timestamp: number) => {
  return new Date(timestamp).toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    fractionalSecondDigits: 3
  })
}

const formatMetadata = (metadata: Record<string, any>) => {
  return Object.entries(metadata)
    .map(([key, value]) => `${key}: ${value}`)
    .join(' | ')
}

const copyToClipboard = async (text: string) => {
  try {
    await navigator.clipboard.writeText(text)
    addLog({
      channel: 'Browser',
      level: 'info',
      message: 'Copied to clipboard'
    })
  } catch (error) {
    addLog({
      channel: 'Browser',
      level: 'error',
      message: 'Failed to copy to clipboard'
    })
  }
}

const toggleDock = () => {
  isDocked.value = !isDocked.value
  if (isDocked.value) {
    position.value = { x: 0, y: 0 }
  }
}

const startDrag = (e: MouseEvent) => {
  if (isDocked.value) return
  isDragging.value = true
  const rect = consoleRef.value!.getBoundingClientRect()
  dragOffset.value = {
    x: e.clientX - rect.left,
    y: e.clientY - rect.top
  }
  document.addEventListener('mousemove', handleDrag)
  document.addEventListener('mouseup', stopDrag)
}

const handleDrag = (e: MouseEvent) => {
  if (!isDragging.value) return
  position.value = {
    x: e.clientX - dragOffset.value.x,
    y: e.clientY - dragOffset.value.y
  }
}

const stopDrag = () => {
  isDragging.value = false
  document.removeEventListener('mousemove', handleDrag)
  document.removeEventListener('mouseup', stopDrag)
}

const handleScroll = () => {
  if (!logContentRef.value) return
  const { scrollTop, scrollHeight, clientHeight } = logContentRef.value
  const isAtBottom = scrollHeight - scrollTop - clientHeight < 10
  if (isAtBottom !== autoScroll.value) {
    toggleAutoScroll()
  }
}

// Auto-scroll to bottom when new logs arrive
watch(filteredLogs, async () => {
  if (autoScroll.value) {
    await nextTick()
    if (logContentRef.value) {
      logContentRef.value.scrollTop = logContentRef.value.scrollHeight
    }
  }
})

onMounted(() => {
  initErrorHandlers()
})

onUnmounted(() => {
  document.removeEventListener('mousemove', handleDrag)
  document.removeEventListener('mouseup', stopDrag)
})
</script>

<style scoped>
.debug-console {
  z-index: 1000;
  resize: both;
  overflow: auto;
}

.log-entry {
  transition: background-color 0.2s;
}

.log-entry:hover {
  background-color: rgba(255, 255, 255, 0.05);
}
</style> 