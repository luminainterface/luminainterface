import { mount } from '@vue/test-utils'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import SubgraphView from '@/components/SubgraphView.vue'
import axios from 'axios'

vi.mock('axios')
const mockedAxios = axios as jest.Mocked<typeof axios>

describe('SubgraphView', () => {
  let wrapper: ReturnType<typeof mount>

  beforeEach(() => {
    wrapper = mount(SubgraphView, {
      props: {
        subgraphId: 'test-subgraph'
      }
    })
  })

  it('renders canvas element', () => {
    const canvas = wrapper.find('canvas')
    expect(canvas.exists()).toBe(true)
  })

  it('shows loading state while fetching data', async () => {
    const mockSubgraph = {
      nodes: [
        { id: 'node1', label: 'Node 1', type: 'type1' },
        { id: 'node2', label: 'Node 2', type: 'type2' }
      ],
      links: [
        { source: 'node1', target: 'node2', type: 'link1' }
      ]
    }

    mockedAxios.get.mockImplementation(() => new Promise(resolve => {
      setTimeout(() => resolve({ data: mockSubgraph }), 100)
    }))

    expect(wrapper.find('.loading').exists()).toBe(true)
    await wrapper.vm.$nextTick()
    expect(wrapper.find('.loading').exists()).toBe(false)
  })

  it('fetches subgraph data on mount', async () => {
    const mockSubgraph = {
      nodes: [
        { id: 'node1', label: 'Node 1', type: 'type1' },
        { id: 'node2', label: 'Node 2', type: 'type2' }
      ],
      links: [
        { source: 'node1', target: 'node2', type: 'link1' }
      ]
    }

    mockedAxios.get.mockResolvedValueOnce({ data: mockSubgraph })
    await wrapper.vm.$nextTick()

    expect(mockedAxios.get).toHaveBeenCalledWith('/api/subgraph/test-subgraph')
  })

  it('handles fetch error gracefully', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    mockedAxios.get.mockRejectedValueOnce(new Error('Network error'))

    await wrapper.vm.$nextTick()

    expect(consoleSpy).toHaveBeenCalled()
    consoleSpy.mockRestore()
  })

  it('updates simulation on window resize', async () => {
    const mockSubgraph = {
      nodes: [
        { id: 'node1', label: 'Node 1', type: 'type1' }
      ],
      links: []
    }

    mockedAxios.get.mockResolvedValueOnce({ data: mockSubgraph })
    await wrapper.vm.$nextTick()

    const simulationSpy = vi.spyOn(wrapper.vm.simulation, 'force')
    window.dispatchEvent(new Event('resize'))
    await wrapper.vm.$nextTick()

    expect(simulationSpy).toHaveBeenCalled()
  })
}) 