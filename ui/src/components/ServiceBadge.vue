<template>
  <span
    :class="{
      'bg-green-500': svc.status === 'ok',
      'bg-red-600': svc.status === 'down',
      'bg-yellow-500': svc.status === 'warn',
      'bg-gray-600': svc.status === 'unknown'
    }"
    class="px-2 py-0.5 rounded-full text-xs font-medium text-white"
  >
    {{ svc.name }}
  </span>
</template>

<script setup lang="ts">
defineProps<{
  svc: {
    name: string
    status: 'ok' | 'down' | 'warn' | 'unknown'
  }
}>()
</script> 