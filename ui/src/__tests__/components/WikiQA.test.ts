import { mount } from '@vue/test-utils'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import WikiQA from '@/components/WikiQA.vue'
import axios from 'axios'

vi.mock('axios')
const mockedAxios = axios as jest.Mocked<typeof axios>

describe('WikiQA', () => {
  let wrapper: ReturnType<typeof mount>

  beforeEach(() => {
    wrapper = mount(WikiQA)
  })

  it('renders input and button', () => {
    expect(wrapper.find('input').exists()).toBe(true)
    expect(wrapper.find('button').exists()).toBe(true)
    expect(wrapper.find('button').text()).toContain('Ask Wiki')
  })

  it('submits question on enter', async () => {
    const input = wrapper.find('input')
    await input.setValue('What is Alan Turing known for?')
    await input.trigger('keyup.enter')

    expect(mockedAxios.post).toHaveBeenCalledWith('/api/masterchat/plan', {
      mode: 'wiki_qa',
      question: 'What is Alan Turing known for?'
    })
  })

  it('submits question on button click', async () => {
    const input = wrapper.find('input')
    const button = wrapper.find('button')
    
    await input.setValue('What is Alan Turing known for?')
    await button.trigger('click')

    expect(mockedAxios.post).toHaveBeenCalledWith('/api/masterchat/plan', {
      mode: 'wiki_qa',
      question: 'What is Alan Turing known for?'
    })
  })

  it('displays loading state while fetching', async () => {
    const input = wrapper.find('input')
    await input.setValue('What is Alan Turing known for?')
    await input.trigger('keyup.enter')

    expect(wrapper.find('.loading-spinner').exists()).toBe(true)
  })

  it('displays answer when received', async () => {
    const mockAnswer = {
      message: 'Alan Turing was a pioneering computer scientist and mathematician.'
    }

    mockedAxios.post.mockResolvedValueOnce({ data: mockAnswer })

    const input = wrapper.find('input')
    await input.setValue('What is Alan Turing known for?')
    await input.trigger('keyup.enter')
    await wrapper.vm.$nextTick()

    expect(wrapper.find('.answer-bubble').text()).toContain(mockAnswer.message)
  })

  it('displays error message on failure', async () => {
    mockedAxios.post.mockRejectedValueOnce(new Error('Network error'))

    const input = wrapper.find('input')
    await input.setValue('What is Alan Turing known for?')
    await input.trigger('keyup.enter')
    await wrapper.vm.$nextTick()

    expect(wrapper.find('.log-entry.error').exists()).toBe(true)
    expect(wrapper.find('.log-entry.error').text()).toContain('Error: Failed to get answer')
  })

  it('clears logs when clear button is clicked', async () => {
    // First add some logs
    await wrapper.setData({
      logs: [
        { timestamp: Date.now(), message: 'Test log', type: 'info' }
      ]
    })

    const clearButton = wrapper.find('.logs-header button')
    await clearButton.trigger('click')

    expect(wrapper.vm.logs).toHaveLength(0)
  })
}) 