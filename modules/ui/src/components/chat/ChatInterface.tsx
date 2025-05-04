import { useState, useRef, useEffect } from 'react';
import { useChat } from '@/hooks/useChat';
import { Message } from '@/types/chat';
import { Button } from '@/components/ui/Button';
import { Markdown } from '@/components/ui/Markdown';
import { CodeBlock } from '@/components/ui/CodeBlock';
import { ConfidenceScore } from '@/components/ui/ConfidenceScore';
import { SourceAttribution } from '@/components/ui/SourceAttribution';

interface ChatInterfaceProps {
  className?: string;
}

export function ChatInterface({ className }: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { messages, sendMessage, isStreaming } = useChat();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    const message = input.trim();
    setInput('');
    await sendMessage(message);
  };

  const renderMessage = (message: Message) => {
    const isUser = message.role === 'user';

    return (
      <div
        key={message.id}
        className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
      >
        <div
          className={`max-w-[80%] rounded-lg p-4 ${
            isUser
              ? 'bg-primary-500 text-white'
              : 'bg-gray-100 dark:bg-gray-800'
          }`}
        >
          <Markdown content={message.content} />
          
          {message.code && (
            <div className="mt-2">
              <CodeBlock
                code={message.code}
                language={message.language || 'text'}
              />
            </div>
          )}

          {!isUser && message.confidence && (
            <div className="mt-2">
              <ConfidenceScore score={message.confidence} />
            </div>
          )}

          {!isUser && message.sources && message.sources.length > 0 && (
            <div className="mt-2">
              <SourceAttribution sources={message.sources} />
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className={`flex flex-col h-full ${className}`}>
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map(renderMessage)}
        <div ref={messagesEndRef} />
      </div>

      <form
        onSubmit={handleSubmit}
        className="border-t border-gray-200 dark:border-gray-700 p-4"
      >
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
            disabled={isStreaming}
          />
          <Button
            type="submit"
            disabled={!input.trim() || isStreaming}
            className="px-6"
          >
            {isStreaming ? 'Sending...' : 'Send'}
          </Button>
        </div>
      </form>
    </div>
  );
} 