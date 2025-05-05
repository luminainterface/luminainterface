import { mount } from '@vue/test-utils'
import { describe, it, expect, beforeEach } from 'vitest'
import HealthOverlay from '@/components/HealthOverlay.vue'
import eventBus, { EventPayload } from '@/store/eventBus'

describe('HealthOverlay', () => {
  let wrapper: ReturnType<typeof mount>

  beforeEach(() => {
    wrapper = mount(HealthOverlay)
  })

  it('renders all services', () => {
    const services = wrapper.findAll('.service')
    expect(services).toHaveLength(6)
    expect(services[0].text()).toContain('API')
    expect(services[1].text()).toContain('Event-Mux')
    expect(services[2].text()).toContain('Redis')
  })

  it('updates service status on health update', async () => {
    const mockHealthUpdate: EventPayload = {
      health: {
        service: 'API',
        status: 'warning',
        latency: 1500
      }
    }

    eventBus.emit('health.update', mockHealthUpdate)
    await wrapper.vm.$nextTick()

    const apiService = wrapper.find('.service:first-child')
    const statusIndicator = apiService.find('.status-indicator')
    expect(statusIndicator.classes()).toContain('bg-yellow-500')
    expect(apiService.text()).toContain('1.5s')
  })

  it('handles error status correctly', async () => {
    const mockHealthUpdate: EventPayload = {
      health: {
        service: 'Redis',
        status: 'error',
        latency: 3500
      }
    }

    eventBus.emit('health.update', mockHealthUpdate)
    await wrapper.vm.$nextTick()

    const redisService = wrapper.find('.service:nth-child(3)')
    const statusIndicator = redisService.find('.status-indicator')
    expect(statusIndicator.classes()).toContain('bg-red-500')
    expect(redisService.text()).toContain('3.5s')
  })
}) 