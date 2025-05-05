import { mount } from '@vue/test-utils'
import { describe, it, expect, beforeEach } from 'vitest'
import MetricsPanel from '@/components/MetricsPanel.vue'
import eventBus, { EventPayload } from '@/store/eventBus'

describe('MetricsPanel', () => {
  let wrapper: ReturnType<typeof mount>

  beforeEach(() => {
    wrapper = mount(MetricsPanel)
  })

  it('toggles visibility on M key press', async () => {
    expect(wrapper.isVisible()).toBe(false)
    
    await wrapper.trigger('keydown', { key: 'm' })
    expect(wrapper.isVisible()).toBe(true)
    
    await wrapper.trigger('keydown', { key: 'm' })
    expect(wrapper.isVisible()).toBe(false)
  })

  it('updates metrics on metric update', async () => {
    const mockMetricUpdate: EventPayload = {
      metric: {
        type: 'nodes_per_second',
        value: 150,
        timestamp: Date.now()
      }
    }

    eventBus.emit('metric.update', mockMetricUpdate)
    await wrapper.vm.$nextTick()

    const nodesPerSecond = wrapper.find('[data-test="nodes-per-second"]')
    expect(nodesPerSecond.text()).toBe('150.0')
  })

  it('formats latency values correctly', async () => {
    const mockLatencyUpdate: EventPayload = {
      metric: {
        type: 'latency',
        service: 'API',
        value: 2500,
        timestamp: Date.now()
      }
    }

    eventBus.emit('metric.update', mockLatencyUpdate)
    await wrapper.vm.$nextTick()

    const apiLatency = wrapper.find('[data-test="api-latency"]')
    expect(apiLatency.text()).toBe('2.5s')
    expect(apiLatency.classes()).toContain('text-yellow-500')
  })

  it('handles multiple metric updates', async () => {
    const updates: EventPayload[] = [
      {
        metric: {
          type: 'nodes_per_second',
          value: 100,
          timestamp: Date.now()
        }
      },
      {
        metric: {
          type: 'edges_per_second',
          value: 200,
          timestamp: Date.now()
        }
      }
    ]

    for (const update of updates) {
      eventBus.emit('metric.update', update)
    }
    await wrapper.vm.$nextTick()

    const nodesPerSecond = wrapper.find('[data-test="nodes-per-second"]')
    const edgesPerSecond = wrapper.find('[data-test="edges-per-second"]')
    
    expect(nodesPerSecond.text()).toBe('100.0')
    expect(edgesPerSecond.text()).toBe('200.0')
  })
}) 