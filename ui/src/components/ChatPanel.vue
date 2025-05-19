<template>
  <section class="chat-panel">
    <div class="history" ref="scroll">
      <div v-for="m in messages" :key="m.id" class="msg">
        <b>{{ m.role }}:</b> {{ m.text }}
      </div>
    </div>
    <form @submit.prevent="send">
      <input v-model="draft" placeholder="Ask MasterChat…" />
    </form>
  </section>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import bus from '@/hooks/useGraphSocket'

const draft = ref('')
const messages = ref([])

async function send () {
  if (!draft.value) return
  messages.value.push({ id: Date.now(), role: 'you', text: draft.value })
  try {
    const base = import.meta.env.VITE_CHAT_URL || 'http://localhost:8301'
    const resp = await fetch(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: draft.value })
    })
    if (!resp.ok) {
      throw new Error(`HTTP error! status: ${resp.status}`)
    }
  } catch (e) {
    bus.emit('error', { message: `Failed to send message: ${e.message}` })
  }
  draft.value = ''
}

onMounted(() => {
  // stream assistant replies (optional)
  bus.on('assistant', txt => messages.value.push({ id: Date.now(), role: 'ai', text: txt }))
})
</script> 