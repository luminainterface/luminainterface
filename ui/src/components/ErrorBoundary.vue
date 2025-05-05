<template>
  <div v-if="error" class="error-boundary">
    <h2>Something went wrong</h2>
    <p class="error-message">{{ error.message }}</p>
    <button @click="resetError" class="retry-button">Retry</button>
  </div>
  <slot v-else></slot>
</template>

<script setup>
import { ref, onErrorCaptured } from 'vue'

const error = ref(null)

onErrorCaptured((err) => {
  error.value = err
  return false // Prevent error from propagating
})

const resetError = () => {
  error.value = null
}
</script>

<style scoped>
.error-boundary {
  padding: 2rem;
  margin: 1rem;
  border: 1px solid #ff4444;
  border-radius: 4px;
  background-color: rgba(255, 68, 68, 0.1);
  color: #ff4444;
}

.error-message {
  margin: 1rem 0;
  font-family: monospace;
  white-space: pre-wrap;
}

.retry-button {
  padding: 0.5rem 1rem;
  background-color: #ff4444;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.retry-button:hover {
  background-color: #ff6666;
}
</style> 