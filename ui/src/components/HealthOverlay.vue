<template>
  <div v-if="Object.values(statuses).some(v=>!v)" class="overlay">
    <div v-for="(ok,svc) in statuses" :key="svc" class="status-item">
      <span :class="ok?'green':'red'">{{svc}}</span>
      <span v-if="!ok" class="error-message">(not reachable)</span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import bus from '@/hooks/useGraphSocket'

const statuses = ref({
  'graph-api': false,
  'masterchat': false,
  'event-mux': false,
  'redis': false
})

onMounted(poll)
async function poll () {
  // Check Redis health
  try {
    const redisHealthUrl = 'http://localhost:8201/redis-health'
    const redisResp = await fetch(redisHealthUrl)
    const redisData = await redisResp.json()
    statuses.value['redis'] = redisData.status === 'healthy'
  } catch (e) {
    statuses.value['redis'] = false
  }
  
  // Check other services
  for (const s of ['graph-api','masterchat','event-mux']) {
    try {
      const base = import.meta.env.VITE_API_URL || 'http://localhost:8201'
      const port = portOf(s)
      const url = `${base.replace('8201', port)}/health`
      const resp = await fetch(url)
      const data = await resp.json()
      statuses.value[s] = data.status === 'healthy'
    } catch (e) {
      statuses.value[s] = false
    }
  }
  
  setTimeout(poll, 10000)
}

function portOf (s){ return { 'graph-api':'8201','masterchat':'8301','event-mux':'8101' }[s] }
</script>

<style>
.overlay {
  position: fixed;
  top: 48px;
  left: 8px;
  background: #000a;
  padding: 8px 12px;
  border-radius: 6px;
  z-index: 1000;
}

.status-item {
  margin: 4px 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.green {
  color: #0f0;
  animation: pulse 1.2s infinite alternate;
}

.red {
  color: #f33;
  animation: pulse 0.7s infinite alternate;
}

.error-message {
  color: #f33;
  font-size: 0.9em;
  opacity: 0.8;
}

@keyframes pulse {
  to {
    filter: brightness(1.5);
  }
}
</style> 