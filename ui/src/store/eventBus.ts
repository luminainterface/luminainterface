import mitt from 'mitt'

export type EventType = 
  | 'edge.add'
  | 'node.add'
  | 'metric.update'
  | 'health.update'
  | 'graph.update'
  | 'subgraph.load'
  | 'view.change'

export type EventPayload = {
  edge?: {
    source: string
    target: string
    timestamp: number
  }
  node?: {
    id: string
    type: string
    timestamp: number
  }
  metric?: {
    type: string
    value: number
    timestamp: number
  }
  health?: {
    service: string
    status: 'healthy' | 'warning' | 'error'
    latency?: number
  }
  graph?: {
    id: string
    nodes: number
    edges: number
  }
}

const emitter = mitt<Record<EventType, EventPayload>>()

export const eventBus = {
  emit: emitter.emit,
  on: emitter.on,
  off: emitter.off
}

export default eventBus 