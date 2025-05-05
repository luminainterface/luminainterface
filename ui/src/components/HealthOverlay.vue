<template>
  <div v-if="Object.values(statuses).some(v=>!v)" class="overlay">
    <span v-for="(ok,svc) in statuses" :key="svc" :class="ok?'green':'red'">{{svc}}</span>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import bus from '@/hooks/useGraphSocket'

const statuses = ref({})
const services = ['graph-api','masterchat','event-mux','redis']

onMounted(poll)
async function poll () {
  for (const s of services) {
    try {
      const base = import.meta.env.VITE_API_URL || 'http://localhost:8201'
      const port = portOf(s)
      const resp = await fetch(`${base.replace('8201', port)}/health`)
      statuses.value[s] = resp.ok
      if (!resp.ok) {
        bus.emit('error', { message: `Service ${s} is not healthy` })
      }
    } catch (e) {
      statuses.value[s] = false
      bus.emit('error', { message: `Service ${s} is not reachable` })
    }
  }
  setTimeout(poll, 10000)
}
function portOf (s){ return { 'graph-api':'8201','masterchat':'8301','event-mux':'8101','redis':'6381' }[s] }
</script>

<style>
.overlay{position:fixed;top:48px;left:8px;background:#000a;padding:4px 8px;border-radius:6px;z-index:1000}
.green{color:#0f0;margin-right:6px;animation:pulse 1.2s infinite alternate}
.red{color:#f33;margin-right:6px;animation:pulse 0.7s infinite alternate}
@keyframes pulse{to{filter:brightness(1.5);}}
</style> 