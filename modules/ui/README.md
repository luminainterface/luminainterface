# Lumina UI Module

## Overview
The Lumina UI module provides a modern, responsive interface for interacting with the Lumina knowledge system. Built with Next.js for optimal performance and developer experience.

## Features
- Real-time chat interface with streaming responses
- Knowledge graph visualization
- Module dashboard for system monitoring
- Plugin management interface
- Dark/light theme support
- Responsive design for all devices

## Architecture

### Core Components
1. **Chat Interface**
   - Real-time message streaming
   - Markdown rendering
   - Code syntax highlighting
   - Source attribution display
   - Confidence scoring visualization

2. **Knowledge Graph**
   - Force-directed graph layout
   - Real-time node updates
   - Interactive node exploration
   - Topic clustering visualization
   - Search and filter capabilities

3. **Module Dashboard**
   - System health metrics
   - Plugin status monitoring
   - Resource usage graphs
   - Log viewer
   - Configuration editor

4. **Plugin Manager**
   - Module discovery
   - Installation interface
   - Configuration UI
   - Status monitoring
   - Update management

### Technical Stack
- **Framework**: Next.js 14
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Graph Visualization**: D3.js
- **WebSocket**: Socket.IO
- **API Client**: TanStack Query
- **UI Components**: shadcn/ui

## Getting Started

### Prerequisites
- Node.js 18+
- pnpm 8+

### Installation
```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev

# Build for production
pnpm build

# Start production server
pnpm start
```

### Environment Variables
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:7210
NEXT_PUBLIC_GRAPH_URL=http://localhost:9080
```

## Development

### Project Structure
```
ui/
├── src/
│   ├── app/                 # Next.js app directory
│   ├── components/          # Reusable components
│   │   ├── chat/           # Chat interface
│   │   ├── graph/          # Knowledge graph
│   │   ├── dashboard/      # System dashboard
│   │   └── plugins/        # Plugin management
│   ├── hooks/              # Custom React hooks
│   ├── lib/                # Utility functions
│   ├── store/              # Zustand stores
│   └── types/              # TypeScript types
├── public/                 # Static assets
└── styles/                 # Global styles
```

### Key Components

#### Chat Interface
```typescript
// components/chat/ChatWindow.tsx
import { useChat } from '@/hooks/useChat';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';

export function ChatWindow() {
  const { messages, sendMessage, isStreaming } = useChat();

  return (
    <div className="flex flex-col h-full">
      <MessageList messages={messages} />
      <ChatInput onSend={sendMessage} disabled={isStreaming} />
    </div>
  );
}
```

#### Knowledge Graph
```typescript
// components/graph/KnowledgeGraph.tsx
import { useGraph } from '@/hooks/useGraph';
import { ForceGraph } from './ForceGraph';
import { GraphControls } from './GraphControls';

export function KnowledgeGraph() {
  const { nodes, links, updateLayout } = useGraph();

  return (
    <div className="relative h-full">
      <ForceGraph nodes={nodes} links={links} />
      <GraphControls onLayoutChange={updateLayout} />
    </div>
  );
}
```

#### Module Dashboard
```typescript
// components/dashboard/SystemMetrics.tsx
import { useMetrics } from '@/hooks/useMetrics';
import { MetricCard } from './MetricCard';
import { LineChart } from './LineChart';

export function SystemMetrics() {
  const { metrics, history } = useMetrics();

  return (
    <div className="grid grid-cols-3 gap-4">
      {metrics.map(metric => (
        <MetricCard key={metric.id} {...metric} />
      ))}
      <LineChart data={history} />
    </div>
  );
}
```

### State Management
```typescript
// store/chatStore.ts
import create from 'zustand';

interface ChatState {
  messages: Message[];
  isStreaming: boolean;
  sendMessage: (text: string) => Promise<void>;
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isStreaming: false,
  sendMessage: async (text) => {
    set({ isStreaming: true });
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      set(state => ({
        messages: [...state.messages, data],
        isStreaming: false,
      }));
    } catch (error) {
      set({ isStreaming: false });
      console.error('Failed to send message:', error);
    }
  },
}));
```

### WebSocket Integration
```typescript
// hooks/useWebSocket.ts
import { useEffect, useRef } from 'react';
import { io, Socket } from 'socket.io-client';

export function useWebSocket(url: string) {
  const socket = useRef<Socket>();

  useEffect(() => {
    socket.current = io(url);
    
    socket.current.on('connect', () => {
      console.log('WebSocket connected');
    });

    socket.current.on('graph_update', (data) => {
      // Handle graph updates
    });

    return () => {
      socket.current?.disconnect();
    };
  }, [url]);

  return socket.current;
}
```

## Styling

### Theme Configuration
```typescript
// styles/theme.ts
export const theme = {
  colors: {
    primary: {
      50: '#f0f9ff',
      100: '#e0f2fe',
      // ...
    },
    // ...
  },
  // ...
};
```

### Component Styling
```typescript
// components/ui/Button.tsx
import { cn } from '@/lib/utils';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
}

export function Button({
  className,
  variant = 'primary',
  size = 'md',
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        'rounded-md font-medium transition-colors',
        {
          'bg-primary-500 text-white hover:bg-primary-600': variant === 'primary',
          'bg-secondary-500 text-white hover:bg-secondary-600': variant === 'secondary',
          'hover:bg-gray-100': variant === 'ghost',
        },
        {
          'px-3 py-1.5 text-sm': size === 'sm',
          'px-4 py-2': size === 'md',
          'px-6 py-3 text-lg': size === 'lg',
        },
        className
      )}
      {...props}
    />
  );
}
```

## Testing

### Component Testing
```typescript
// components/chat/__tests__/ChatWindow.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { ChatWindow } from '../ChatWindow';

describe('ChatWindow', () => {
  it('renders chat interface', () => {
    render(<ChatWindow />);
    expect(screen.getByRole('textbox')).toBeInTheDocument();
  });

  it('sends message on submit', async () => {
    render(<ChatWindow />);
    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'Hello' } });
    fireEvent.click(screen.getByRole('button'));
    expect(await screen.findByText('Hello')).toBeInTheDocument();
  });
});
```

## Deployment

### Build Process
```bash
# Build the application
pnpm build

# Export static files
pnpm export

# Deploy to production
pnpm deploy
```

### Docker Support
```dockerfile
# Dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY . .
RUN pnpm install
RUN pnpm build

FROM node:18-alpine AS runner
WORKDIR /app
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
RUN pnpm install --prod
EXPOSE 3000
CMD ["pnpm", "start"]
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License - see LICENSE file for details 