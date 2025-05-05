import { mount } from '@vue/test-utils'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import FractalView from '@/components/FractalView.vue'
import axios from 'axios'

vi.mock('axios')
const mockedAxios = axios as jest.Mocked<typeof axios>

describe('FractalView', () => {
  let wrapper: ReturnType<typeof mount>

  beforeEach(() => {
    wrapper = mount(FractalView)
  })

  it('renders SVG element', () => {
    const svg = wrapper.find('svg')
    expect(svg.exists()).toBe(true)
  })

  it('fetches hierarchy data on mount', async () => {
    const mockHierarchy = {
      id: 'root',
      name: 'Root',
      children: [
        {
          id: 'child1',
          name: 'Child 1',
          value: 10
        }
      ]
    }

    mockedAxios.get.mockResolvedValueOnce({ data: mockHierarchy })
    await wrapper.vm.$nextTick()

    expect(mockedAxios.get).toHaveBeenCalledWith('/api/hierarchy')
  })

  it('handles fetch error gracefully', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    mockedAxios.get.mockRejectedValueOnce(new Error('Network error'))

    await wrapper.vm.$nextTick()

    expect(consoleSpy).toHaveBeenCalled()
    consoleSpy.mockRestore()
  })

  it('navigates through breadcrumbs', async () => {
    const mockHierarchy = {
      id: 'root',
      name: 'Root',
      children: [
        {
          id: 'child1',
          name: 'Child 1',
          children: [
            {
              id: 'grandchild1',
              name: 'Grandchild 1',
              value: 5
            }
          ]
        }
      ]
    }

    mockedAxios.get.mockResolvedValueOnce({ data: mockHierarchy })
    await wrapper.vm.$nextTick()

    const breadcrumbs = wrapper.findAll('.breadcrumb')
    expect(breadcrumbs).toHaveLength(1)
    expect(breadcrumbs[0].text()).toContain('Root')
  })

  it('handles window resize', async () => {
    const renderSpy = vi.spyOn(wrapper.vm, 'renderHierarchy')
    
    window.dispatchEvent(new Event('resize'))
    await wrapper.vm.$nextTick()

    expect(renderSpy).toHaveBeenCalled()
  })
}) 