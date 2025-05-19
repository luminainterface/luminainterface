<template>
  <div class="dashboard-grid">
    <HealthOverlay />
    <DebugPanel />
    <div class="main-visuals">
      <FractalView />
      <MetricsPanel />
    </div>
    <div class="chat-section">
      <h2>Master Chat</h2>
      <ChatPanel />
      <h2>Dual Chat</h2>
      <ChatPanel />
    </div>
  </div>
</template>

<script setup>
import ChatPanel from '../components/ChatPanel.vue'
import MetricsPanel from '../components/MetricsPanel.vue'
import FractalView from '../components/FractalView.vue'
import HealthOverlay from '../components/HealthOverlay.vue'
import DebugPanel from '../components/DebugPanel.vue'
</script>

<style scoped>
.dashboard-grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  grid-template-rows: auto 1fr;
  gap: 24px;
  min-height: 100vh;
  padding: 24px;
  background: #f7fafc;
}
.main-visuals {
  grid-column: 1 / 2;
  grid-row: 1 / 3;
  display: flex;
  flex-direction: column;
  gap: 24px;
}
.chat-section {
  grid-column: 2 / 3;
  grid-row: 1 / 3;
  display: flex;
  flex-direction: column;
  gap: 16px;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  padding: 16px;
}
.chat-section h2 {
  margin: 0 0 8px 0;
  font-size: 1.1rem;
  color: #333;
}
</style> 