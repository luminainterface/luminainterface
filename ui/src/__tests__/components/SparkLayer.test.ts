import { mount } from '@vue/test-utils'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import SparkLayer from '@/components/SparkLayer.vue'
import eventBus, { EventPayload } from '@/store/eventBus'

vi.mock('pixi.js', () => ({
  Application: vi.fn().mockImplementation(() => ({
    view: document.createElement('canvas'),
    stage: {
      addChild: vi.fn()
    },
    ticker: {
      add: vi.fn()
    },
    destroy: vi.fn()
  })),
  Container: vi.fn().mockImplementation(() => ({
    addChild: vi.fn()
  })),
  Graphics: vi.fn().mockImplementation(() => ({
    clear: vi.fn(),
    lineStyle: vi.fn(),
    moveTo: vi.fn(),
    quadraticCurveTo: vi.fn(),
    destroy: vi.fn()
  }))
}))

describe('SparkLayer', () => {
  let wrapper: ReturnType<typeof mount>

  beforeEach(() => {
    wrapper = mount(SparkLayer)
  })

  it('initializes PIXI application', () => {
    expect(wrapper.find('.spark-container').exists()).toBe(true)
  })

  it('creates spark on edge add event', async () => {
    const mockEdgeAdd: EventPayload = {
      edge: {
        source: 'node1',
        target: 'node2',
        timestamp: Date.now()
      }
    }

    // Mock DOM elements
    const sourceNode = document.createElement('div')
    sourceNode.setAttribute('data-node-id', 'node1')
    sourceNode.getBoundingClientRect = () => ({
      left: 100,
      top: 100,
      width: 50,
      height: 50
    } as DOMRect)

    const targetNode = document.createElement('div')
    targetNode.setAttribute('data-node-id', 'node2')
    targetNode.getBoundingClientRect = () => ({
      left: 300,
      top: 300,
      width: 50,
      height: 50
    } as DOMRect)

    document.body.appendChild(sourceNode)
    document.body.appendChild(targetNode)

    eventBus.emit('edge.add', mockEdgeAdd)
    await wrapper.vm.$nextTick()

    expect(wrapper.vm.sparks.size).toBe(1)
  })

  it('cleans up on unmount', () => {
    const destroySpy = vi.spyOn(wrapper.vm.app, 'destroy')
    wrapper.unmount()
    expect(destroySpy).toHaveBeenCalled()
  })

  it('handles window resize', async () => {
    const resizeSpy = vi.spyOn(wrapper.vm.app, 'resizeTo')
    window.dispatchEvent(new Event('resize'))
    await wrapper.vm.$nextTick()
    expect(resizeSpy).toHaveBeenCalled()
  })
}) 