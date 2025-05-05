<template>
  <div id="layout">
    <HealthOverlay />
    <DebugPanel />
    <Confetti />
    <nav>
      <span class="logo">Lumina</span>
      <RouterLink to="/chat">Chat</RouterLink>
      <RouterLink to="/graph">Graph</RouterLink>
      <button @click="toggleMetrics">Metrics</button>
    </nav>

    <div id="main">
      <ChatPanel />
      <router-view class="graph-slot" />
      <MetricsPanel v-if="showMetrics" />
    </div>

    <TerminalConsole />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import ChatPanel from '@/components/ChatPanel.vue'
import MetricsPanel from '@/components/MetricsPanel.vue'
import TerminalConsole from '@/components/TerminalConsole.vue'
import HealthOverlay from '@/components/HealthOverlay.vue'
import DebugPanel from '@/components/DebugPanel.vue'
import Confetti from '@/components/Confetti.vue'

const showMetrics = ref(false)
const toggleMetrics = () => (showMetrics.value = !showMetrics.value)
</script> 