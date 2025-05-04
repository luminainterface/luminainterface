import { ChatInterface } from '@/components/chat/ChatInterface';
import { Sidebar } from '@/components/layout/Sidebar';
import { ThemeToggle } from '@/components/ThemeToggle';

export default function Home() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <header className="h-16 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between px-4">
          <h1 className="text-xl font-semibold">Lumina Chat</h1>
          <ThemeToggle />
        </header>
        <main className="flex-1 overflow-hidden">
          <ChatInterface />
        </main>
      </div>
    </div>
  );
} 