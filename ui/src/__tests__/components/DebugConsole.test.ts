import { mount } from '@vue/test-utils'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import DebugConsole from '@/components/DebugConsole.vue'
import { useDebugStore } from '@/stores/debug'

describe('DebugConsole', () => {
  let wrapper: ReturnType<typeof mount>
  let debugStore: ReturnType<typeof useDebugStore>

  beforeEach(() => {
    setActivePinia(createPinia())
    debugStore = useDebugStore()
    wrapper = mount(DebugConsole)
  })

  it('renders all channel tabs', () => {
    const tabs = wrapper.findAll('.channel-tab')
    expect(tabs).toHaveLength(4)
    expect(tabs[0].text()).toBe('Pipeline')
    expect(tabs[1].text()).toBe('Service')
    expect(tabs[2].text()).toBe('Network')
    expect(tabs[3].text()).toBe('Browser')
  })

  it('toggles dock state', async () => {
    const dockButton = wrapper.find('.dock-button')
    expect(wrapper.classes()).toContain('docked')
    
    await dockButton.trigger('click')
    expect(wrapper.classes()).not.toContain('docked')
    
    await dockButton.trigger('click')
    expect(wrapper.classes()).toContain('docked')
  })

  it('filters logs by selected channel', async () => {
    // Add test logs
    debugStore.addLog({
      channel: 'Pipeline',
      level: 'info',
      message: 'Pipeline log'
    })
    debugStore.addLog({
      channel: 'Service',
      level: 'info',
      message: 'Service log'
    })

    // Check Pipeline channel
    const pipelineTab = wrapper.find('.channel-tab:first-child')
    await pipelineTab.trigger('click')
    const pipelineLogs = wrapper.findAll('.log-entry')
    expect(pipelineLogs).toHaveLength(1)
    expect(pipelineLogs[0].text()).toContain('Pipeline log')

    // Check Service channel
    const serviceTab = wrapper.find('.channel-tab:nth-child(2)')
    await serviceTab.trigger('click')
    const serviceLogs = wrapper.findAll('.log-entry')
    expect(serviceLogs).toHaveLength(1)
    expect(serviceLogs[0].text()).toContain('Service log')
  })

  it('handles log levels correctly', async () => {
    debugStore.addLog({
      channel: 'Pipeline',
      level: 'info',
      message: 'Info log'
    })
    debugStore.addLog({
      channel: 'Pipeline',
      level: 'warning',
      message: 'Warning log'
    })
    debugStore.addLog({
      channel: 'Pipeline',
      level: 'error',
      message: 'Error log'
    })

    const logs = wrapper.findAll('.log-entry')
    expect(logs[0].classes()).toContain('text-blue-500')
    expect(logs[1].classes()).toContain('text-yellow-500')
    expect(logs[2].classes()).toContain('text-red-500')
  })

  it('clears logs', async () => {
    debugStore.addLog({
      channel: 'Pipeline',
      level: 'info',
      message: 'Test log'
    })

    const clearButton = wrapper.find('.clear-button')
    await clearButton.trigger('click')

    const logs = wrapper.findAll('.log-entry')
    expect(logs).toHaveLength(0)
  })

  it('copies log message to clipboard', async () => {
    const mockClipboard = {
      writeText: vi.fn()
    }
    Object.assign(navigator, {
      clipboard: mockClipboard
    })

    debugStore.addLog({
      channel: 'Pipeline',
      level: 'info',
      message: 'Test log',
      metadata: { test: 'data' }
    })

    const copyButton = wrapper.find('.copy-button')
    await copyButton.trigger('click')

    expect(mockClipboard.writeText).toHaveBeenCalledWith(
      expect.stringContaining('Test log')
    )
  })

  it('handles auto-scroll', async () => {
    const autoScrollButton = wrapper.find('.auto-scroll-button')
    const logContent = wrapper.find('.log-content')
    
    // Enable auto-scroll
    await autoScrollButton.trigger('click')
    expect(wrapper.vm.autoScroll).toBe(true)

    // Add logs and check scroll
    for (let i = 0; i < 10; i++) {
      debugStore.addLog({
        channel: 'Pipeline',
        level: 'info',
        message: `Log ${i}`
      })
    }

    await wrapper.vm.$nextTick()
    expect(logContent.element.scrollTop).toBe(logContent.element.scrollHeight)
  })
}) 