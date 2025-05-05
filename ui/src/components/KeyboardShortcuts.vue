<template>
  <div class="keyboard-shortcuts" tabindex="0" @keydown="handleKeyPress">
    <slot></slot>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

const handleKeyPress = (event) => {
  // Shift + M: Toggle metrics panel
  if (event.shiftKey && event.key.toLowerCase() === 'm') {
    event.preventDefault()
    router.push('/metrics')
  }
  
  // G: Navigate to graph view
  if (event.key.toLowerCase() === 'g') {
    event.preventDefault()
    router.push('/graph')
  }
  
  // Shift + G: Navigate to subgraph view
  if (event.shiftKey && event.key.toLowerCase() === 'g') {
    event.preventDefault()
    router.push('/subgraph')
  }
}

onMounted(() => {
  document.addEventListener('keydown', handleKeyPress)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeyPress)
})
</script>

<style scoped>
.keyboard-shortcuts {
  outline: none;
}
</style> 