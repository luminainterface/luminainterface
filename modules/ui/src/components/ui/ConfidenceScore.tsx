import { cn } from '@/lib/utils';

interface ConfidenceScoreProps {
  score: number;
  className?: string;
}

export function ConfidenceScore({ score, className }: ConfidenceScoreProps) {
  const getColor = (score: number) => {
    if (score >= 0.8) return 'text-green-500';
    if (score >= 0.6) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getLabel = (score: number) => {
    if (score >= 0.8) return 'High';
    if (score >= 0.6) return 'Medium';
    return 'Low';
  };

  return (
    <div className={cn('flex items-center gap-2 text-sm', className)}>
      <span className="text-gray-500 dark:text-gray-400">Confidence:</span>
      <span className={getColor(score)}>
        {getLabel(score)} ({Math.round(score * 100)}%)
      </span>
      <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className={cn('h-full transition-all duration-300', {
            'bg-green-500': score >= 0.8,
            'bg-yellow-500': score >= 0.6 && score < 0.8,
            'bg-red-500': score < 0.6,
          })}
          style={{ width: `${score * 100}%` }}
        />
      </div>
    </div>
  );
} 