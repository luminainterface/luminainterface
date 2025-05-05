<template>
  <section class="metrics-panel">
    <canvas ref="canvas"></canvas>
    <footer class="metrics-footer">
      <button @click="crawl">Crawl Seed</button>
      <button @click="pause">Pause Crawler</button>
      <button @click="prune">Prune Hub</button>
    </footer>
  </section>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { Chart, LineController, LineElement, TimeScale, LinearScale, PointElement, Legend, Tooltip } from 'chart.js'
import 'chartjs-adapter-date-fns'
import { useToast } from 'vue-toastification'
import bus from '@/hooks/useGraphSocket'
Chart.register(LineController, LineElement, TimeScale, LinearScale, PointElement, Legend, Tooltip)

const canvas = ref(null)
let chart
const toast = useToast()

const fetchMetrics = async () => {
  const base = import.meta.env.VITE_API_URL || 'http://localhost:8201'
  const r = await fetch(`${base}/metrics/summary`).then(r => {
    if (!r.ok) throw new Error(`HTTP error! status: ${r.status}`)
    return r.json()
  })
  const t = Date.now()
  chart.data.datasets[0].data.push({x:t,y:r.nodes})
  chart.data.datasets[1].data.push({x:t,y:r.edges})
  chart.data.datasets[2].data.push({x:t,y:r.fractal_dimension})
  chart.update('none')
}

const poll = async () => {
  try {
    await fetchMetrics()
  } catch (error) {
    console.error('Failed to fetch metrics:', error)
    toast.warn('Metrics API unreachable')
  } finally {
    setTimeout(poll, 5000)
  }
}

onMounted(async () => {
  chart = new Chart(canvas.value.getContext('2d'), {
    type: 'line',
    data: { datasets:[
      {label:'Nodes', data:[], parsing:false},
      {label:'Edges', data:[], parsing:false},
      {label:'Fractal Dim', data:[], parsing:false, yAxisID:'y1'}
    ]},
    options:{ scales:{ x:{type:'time'}, y:{beginAtZero:true}, y1:{position:'right'}}}
  })
  poll()
})

async function crawl(){
  const seed = prompt('Seed topic?')
  if(!seed) return
  const ok = await fetch(`${import.meta.env.VITE_API_URL}/tasks`,{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({crawl:[seed],hops:1,max_nodes:40})
  }).then(r=>r.ok)
  toast[ok?'success':'error'](ok?'Task accepted':'Failed')
  if(ok) bus.emit('confetti')
}
async function pause(){
  const ok = await fetch(`${import.meta.env.VITE_API_URL}/tasks`,{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({pause:true})
  }).then(r=>r.ok)
  toast[ok?'success':'error'](ok?'Paused crawler':'Failed')
  if(ok) bus.emit('confetti')
}
async function prune(){
  const ok = await fetch(`${import.meta.env.VITE_API_URL}/tasks`,{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({prune:'hub'})
  }).then(r=>r.ok)
  toast[ok?'success':'error'](ok?'Pruned hub':'Failed')
  if(ok) bus.emit('confetti')
}
</script>

<style>
.metrics-footer{display:flex;gap:8px;margin-top:8px}
.metrics-footer button{background:#222;color:#fff;border:none;padding:6px 12px;border-radius:4px;cursor:pointer}
.metrics-footer button:hover{background:#444}
</style> 