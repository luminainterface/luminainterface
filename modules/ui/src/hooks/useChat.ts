import { useCallback } from 'react';
import { create } from 'zustand';
import { v4 as uuidv4 } from 'uuid';
import { Message, ChatState } from '@/types/chat';

const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isStreaming: false,
  error: null,
}));

export function useChat() {
  const { messages, isStreaming, error } = useChatStore();

  const sendMessage = useCallback(async (text: string) => {
    const userMessage: Message = {
      id: uuidv4(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    useChatStore.setState((state) => ({
      messages: [...state.messages, userMessage],
      isStreaming: true,
      error: null,
    }));

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: text }),
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const data = await response.json();
      const assistantMessage: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: data.content,
        code: data.code,
        language: data.language,
        confidence: data.confidence,
        sources: data.sources,
        timestamp: Date.now(),
      };

      useChatStore.setState((state) => ({
        messages: [...state.messages, assistantMessage],
        isStreaming: false,
      }));
    } catch (err) {
      useChatStore.setState({
        isStreaming: false,
        error: err instanceof Error ? err.message : 'An error occurred',
      });
    }
  }, []);

  const clearMessages = useCallback(() => {
    useChatStore.setState({ messages: [], error: null });
  }, []);

  return {
    messages,
    isStreaming,
    error,
    sendMessage,
    clearMessages,
  };
} 