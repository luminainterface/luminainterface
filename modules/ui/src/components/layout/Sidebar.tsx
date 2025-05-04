import { useState } from 'react';
import { useTheme } from '@/components/ThemeProvider';
import { Button } from '@/components/ui/Button';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  MessageSquare,
  Settings,
  BookOpen,
  Brain,
  Menu,
  X,
  LayoutDashboard,
} from 'lucide-react';

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(true);
  const { theme, setTheme } = useTheme();
  const pathname = usePathname();

  const toggleSidebar = () => {
    setIsOpen(!isOpen);
  };

  const isActive = (path: string) => pathname === path;

  return (
    <>
      <Button
        variant="ghost"
        size="icon"
        className="fixed top-4 left-4 z-50 md:hidden"
        onClick={toggleSidebar}
      >
        {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
      </Button>

      <aside
        className={`fixed inset-y-0 left-0 z-40 w-64 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 transform transition-transform duration-200 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        } md:translate-x-0`}
      >
        <div className="flex flex-col h-full">
          <div className="p-4 border-b border-gray-200 dark:border-gray-800">
            <h2 className="text-lg font-semibold">Lumina</h2>
          </div>

          <nav className="flex-1 p-4 space-y-2">
            <Link href="/dashboard">
              <Button
                variant={isActive('/dashboard') ? 'default' : 'ghost'}
                className="w-full justify-start"
              >
                <LayoutDashboard className="mr-2 h-5 w-5" />
                Dashboard
              </Button>
            </Link>
            <Link href="/">
              <Button
                variant={isActive('/') ? 'default' : 'ghost'}
                className="w-full justify-start"
              >
                <MessageSquare className="mr-2 h-5 w-5" />
                Chat
              </Button>
            </Link>
            <Link href="/knowledge">
              <Button
                variant={isActive('/knowledge') ? 'default' : 'ghost'}
                className="w-full justify-start"
              >
                <BookOpen className="mr-2 h-5 w-5" />
                Knowledge Base
              </Button>
            </Link>
            <Link href="/learning">
              <Button
                variant={isActive('/learning') ? 'default' : 'ghost'}
                className="w-full justify-start"
              >
                <Brain className="mr-2 h-5 w-5" />
                Learning
              </Button>
            </Link>
          </nav>

          <div className="p-4 border-t border-gray-200 dark:border-gray-800">
            <Link href="/settings">
              <Button
                variant={isActive('/settings') ? 'default' : 'ghost'}
                className="w-full justify-start"
              >
                <Settings className="mr-2 h-5 w-5" />
                Settings
              </Button>
            </Link>
          </div>
        </div>
      </aside>
    </>
  );
} 