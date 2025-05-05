<template>
  <aside class="debug" v-show="open">
    <h3>Debugger</h3>
    <p><b>WebSocket:</b> {{ wsState }}</p>
    <p><b>Stream lag:</b> {{ redisLag }}</p>
    <p><b>Planner p95:</b> {{ plannerP95 }} s</p>
    <pre>{{ lastEvent }}</pre>
  </aside>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import { useGraphSocket } from '@/hooks/GraphSocket'
import { notify } from '@/bus/notify'
import eventBus from '@/store/eventBus'

const open      = ref(false)
const wsState   = ref('—')
const redisLag  = ref('—')
const plannerP95= ref('—')
const lastEvent = ref('{}')
const { isConnected } = useGraphSocket()

window.addEventListener('keydown', e=>{
  if(e.shiftKey && e.key.toLowerCase()==='d') open.value = !open.value
})

watch(isConnected, (connected) => {
  wsState.value = connected ? 'connected' : 'disconnected'
})

onMounted(()=>{
  eventBus.on('graph', ev => lastEvent.value = JSON.stringify(ev).slice(0,160)+'…')
})

async function poll(){
  try{
    const m = await fetch(`${import.meta.env.VITE_API_URL}/metrics/summary`).then(r=>r.json())
    if (m) {
      redisLag.value = m.redis_lag || '—'
      plannerP95.value = m.planner_p95 ? m.planner_p95.toFixed(2) : '—'
    } else {
      redisLag.value = '—'
      plannerP95.value = '—'
    }
  }catch(e){
    notify.emit('toast',{level:'error',msg:'Metrics API down'})
    redisLag.value = '—'
    plannerP95.value = '—'
  }
  setTimeout(poll, 5000)
}
poll()
</script>

<style>
.debug{position:fixed;right:0;top:50px;width:240px;background:#000c;color:#0f0;padding:8px;font:12px/1.4 monospace;z-index:99}
</style> 