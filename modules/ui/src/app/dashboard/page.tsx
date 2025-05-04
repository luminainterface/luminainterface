import { ModuleDashboard } from '@/components/dashboard/ModuleDashboard';
import { MetricsPanel } from '@/components/dashboard/MetricsPanel';
import { Sidebar } from '@/components/layout/Sidebar';
import { ThemeToggle } from '@/components/ThemeToggle';

export default function DashboardPage() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <header className="h-16 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between px-4">
          <h1 className="text-xl font-semibold">System Dashboard</h1>
          <ThemeToggle />
        </header>
        <main className="flex-1 overflow-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
            <div className="lg:col-span-2">
              <ModuleDashboard />
            </div>
            <div className="lg:col-span-2">
              <MetricsPanel />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
} 