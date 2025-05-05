<template>
  <canvas ref="canvas" class="confetti-canvas"></canvas>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import confetti from 'canvas-confetti'
import bus from '@/hooks/useGraphSocket'

const canvas = ref(null)

onMounted(() => {
  bus.on('confetti', () => {
    confetti.create(canvas.value, { resize: true })({
      particleCount: 80,
      spread: 70,
      origin: { y: 0.6 }
    })
  })
})
</script>

<style>
.confetti-canvas {
  position: fixed;
  pointer-events: none;
  top: 0; left: 0; width: 100vw; height: 100vh;
  z-index: 9999;
}
</style> 