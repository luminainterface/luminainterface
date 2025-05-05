<template>
  <div ref="containerRef" class="absolute inset-0 pointer-events-none"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import * as PIXI from 'pixi.js'
import eventBus, { EventType, EventPayload } from '../store/eventBus'

interface Spark {
  id: string
  graphics: PIXI.Graphics
  startTime: number
  source: { x: number; y: number }
  target: { x: number; y: number }
}

const containerRef = ref<HTMLDivElement | null>(null)
const app = ref<PIXI.Application | null>(null)
const sparks = ref<Map<string, Spark>>(new Map())
const sparkContainer = ref<PIXI.Container | null>(null)

const initializePixi = () => {
  if (!containerRef.value) return

  app.value = new PIXI.Application({
    background: 'transparent',
    resizeTo: containerRef.value,
    antialias: true
  })

  containerRef.value.appendChild(app.value.view as HTMLCanvasElement)
  sparkContainer.value = new PIXI.Container()
  app.value.stage.addChild(sparkContainer.value)

  // Start animation loop
  app.value.ticker.add(animate)
}

const createSpark = (source: { x: number; y: number }, target: { x: number; y: number }) => {
  if (!sparkContainer.value) return

  const id = Math.random().toString(36).substr(2, 9)
  const graphics = new PIXI.Graphics()
  sparkContainer.value.addChild(graphics)

  const spark: Spark = {
    id,
    graphics,
    startTime: Date.now(),
    source,
    target
  }

  sparks.value.set(id, spark)
}

const animate = () => {
  const now = Date.now()
  const duration = 4000 // 4 seconds

  sparks.value.forEach((spark, id) => {
    const age = now - spark.startTime
    if (age >= duration) {
      spark.graphics.destroy()
      sparks.value.delete(id)
      return
    }

    const progress = age / duration
    const alpha = 1 - progress

    // Calculate current position using quadratic bezier
    const midX = (spark.source.x + spark.target.x) / 2
    const midY = (spark.source.y + spark.target.y) / 2 - 50 // Arc height

    const t = progress
    const x = Math.pow(1 - t, 2) * spark.source.x + 
              2 * (1 - t) * t * midX + 
              Math.pow(t, 2) * spark.target.x
    const y = Math.pow(1 - t, 2) * spark.source.y + 
              2 * (1 - t) * t * midY + 
              Math.pow(t, 2) * spark.target.y

    // Draw spark
    spark.graphics.clear()
    spark.graphics.lineStyle(2, 0x60A5FA, alpha)
    spark.graphics.moveTo(spark.source.x, spark.source.y)
    spark.graphics.quadraticCurveTo(midX, midY, x, y)
  })
}

const handleEdgeAdd = (payload: EventPayload) => {
  if (!payload.edge) return

  // Get node positions from the current view
  const sourceNode = document.querySelector(`[data-node-id="${payload.edge.source}"]`)
  const targetNode = document.querySelector(`[data-node-id="${payload.edge.target}"]`)

  if (sourceNode && targetNode) {
    const sourceRect = sourceNode.getBoundingClientRect()
    const targetRect = targetNode.getBoundingClientRect()

    const source = {
      x: sourceRect.left + sourceRect.width / 2,
      y: sourceRect.top + sourceRect.height / 2
    }

    const target = {
      x: targetRect.left + targetRect.width / 2,
      y: targetRect.top + targetRect.height / 2
    }

    createSpark(source, target)
  }
}

onMounted(() => {
  initializePixi()
  eventBus.on('edge.add', handleEdgeAdd)
})

onUnmounted(() => {
  if (app.value) {
    app.value.destroy(true)
  }
  eventBus.off('edge.add', handleEdgeAdd)
})
</script> 