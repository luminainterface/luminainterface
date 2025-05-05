import { mount } from '@vue/test-utils'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import PipelineViz from '@/components/PipelineViz.vue'
import { useHealthStore } from '@/stores/health'

vi.mock('@/stores/health', () => ({
  useHealthStore: () => ({
    services: [
      { name: 'API', status: 'ok' },
      { name: 'Event-Mux', status: 'ok' },
      { name: 'Redis', status: 'ok' }
    ],
    startPolling: vi.fn(),
    stopPolling: vi.fn()
  })
}))

describe('PipelineViz', () => {
  let wrapper: ReturnType<typeof mount>

  beforeEach(() => {
    wrapper = mount(PipelineViz)
  })

  it('renders all pipeline steps', () => {
    const steps = wrapper.findAll('.rounded-full')
    expect(steps).toHaveLength(3)
    expect(steps[0].text()).toBe('Crawl')
    expect(steps[1].text()).toBe('Summarise')
    expect(steps[2].text()).toBe('QA')
  })

  it('renders service badges', () => {
    const badges = wrapper.findAllComponents({ name: 'ServiceBadge' })
    expect(badges).toHaveLength(3)
  })

  it('updates step status on SSE event', async () => {
    const event = new MessageEvent('message', {
      data: JSON.stringify({
        agent: 'CrawlAgent',
        status: 'start'
      })
    })

    window.dispatchEvent(event)
    await wrapper.vm.$nextTick()

    const crawlStep = wrapper.find('.rounded-full:first-child')
    expect(crawlStep.classes()).toContain('bg-yellow-500')
  })

  it('shows retry button on error', async () => {
    const event = new MessageEvent('message', {
      data: JSON.stringify({
        agent: 'CrawlAgent',
        status: 'error'
      })
    })

    window.dispatchEvent(event)
    await wrapper.vm.$nextTick()

    expect(wrapper.find('button').exists()).toBe(true)
    expect(wrapper.find('button').text()).toBe('Retry Failed Step')
  })

  it('starts and stops health polling', () => {
    const healthStore = useHealthStore()
    expect(healthStore.startPolling).toHaveBeenCalled()

    wrapper.unmount()
    expect(healthStore.stopPolling).toHaveBeenCalled()
  })
}) 