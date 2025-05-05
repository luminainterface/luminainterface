<template>
  <div class="terminal">
    <select v-model="filter">
      <option value="">All</option>
      <option value="[PLNR]">Planner</option>
      <option value="[EDGE]">Edges</option>
      <option value="[ERR]">Errors</option>
    </select>
    <pre ref="logEl">{{ lines.filter(l=>!filter||l.startsWith(filter)).join('\n') }}</pre>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import bus from '@/hooks/useGraphSocket'

const lines = ref([])
const filter = ref("")

onMounted(() => {
  bus.on('planner', l => pushLine('[PLNR] ' + l.msg))
  bus.on('graph',   e => pushLine(`[EDGE] ${e.payload.source}→${e.payload.target}`))
  bus.on('error',   e => pushLine(`[ERR] ${e.message}`))
})

function pushLine (text) {
  lines.value.push(text)
  if (lines.value.length > 200) lines.value.shift()
}
</script>

<style>
.terminal { height: 200px;background:#000;color:#0f0;overflow:auto;font:12px/1.4 monospace;padding:4px;}
.terminal select { margin-bottom: 4px; background: #222; color: #0f0; border: none; padding: 2px 6px; }
</style> 