import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/Button';
import { cn } from '@/lib/utils';
import { Server, AlertCircle, CheckCircle2 } from 'lucide-react';

interface Service {
  id: string;
  name: string;
  status: 'healthy' | 'degraded' | 'down';
  lastSeen: string;
  version: string;
  endpoints: string[];
}

export function ModuleDashboard() {
  const [services, setServices] = useState<Service[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchServices = async () => {
      try {
        const response = await fetch('/api/modules');
        if (!response.ok) throw new Error('Failed to fetch services');
        const data = await response.json();
        setServices(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load services');
      } finally {
        setLoading(false);
      }
    };

    fetchServices();
    const interval = setInterval(fetchServices, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: Service['status']) => {
    switch (status) {
      case 'healthy':
        return 'text-green-500';
      case 'degraded':
        return 'text-yellow-500';
      case 'down':
        return 'text-red-500';
    }
  };

  const getStatusIcon = (status: Service['status']) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle2 className="h-5 w-5" />;
      case 'degraded':
        return <AlertCircle className="h-5 w-5" />;
      case 'down':
        return <Server className="h-5 w-5" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-red-500">{error}</div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <h2 className="text-2xl font-semibold mb-6">System Modules</h2>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {services.map((service) => (
          <div
            key={service.id}
            className="bg-white dark:bg-gray-800 rounded-lg shadow p-4"
          >
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-lg font-medium">{service.name}</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  v{service.version}
                </p>
              </div>
              <div className={cn('flex items-center gap-2', getStatusColor(service.status))}>
                {getStatusIcon(service.status)}
                <span className="text-sm font-medium capitalize">{service.status}</span>
              </div>
            </div>
            <div className="mt-4">
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Last seen: {new Date(service.lastSeen).toLocaleString()}
              </p>
              <div className="mt-2">
                <h4 className="text-sm font-medium mb-1">Endpoints:</h4>
                <div className="space-y-1">
                  {service.endpoints.map((endpoint) => (
                    <div
                      key={endpoint}
                      className="text-sm text-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-gray-700 px-2 py-1 rounded"
                    >
                      {endpoint}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
} 