import { useState } from 'react';
import { Source } from '@/types/chat';
import { Button } from './Button';
import { cn } from '@/lib/utils';
import { ExternalLink, ChevronDown, ChevronUp } from 'lucide-react';

interface SourceAttributionProps {
  sources: Source[];
  className?: string;
}

export function SourceAttribution({ sources, className }: SourceAttributionProps) {
  const [expanded, setExpanded] = useState(false);

  if (sources.length === 0) return null;

  return (
    <div className={cn('text-sm', className)}>
      <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
        <span>Sources:</span>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setExpanded(!expanded)}
          className="p-0 h-auto"
        >
          {expanded ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </Button>
      </div>

      {expanded && (
        <div className="mt-2 space-y-2">
          {sources.map((source) => (
            <div
              key={source.id}
              className="p-2 rounded-md bg-gray-50 dark:bg-gray-800"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900 dark:text-gray-100">
                    {source.title}
                  </h4>
                  <p className="mt-1 text-gray-600 dark:text-gray-300">
                    {source.snippet}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  asChild
                  className="flex-shrink-0"
                >
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                  >
                    <ExternalLink className="h-4 w-4" />
                  </a>
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
} 