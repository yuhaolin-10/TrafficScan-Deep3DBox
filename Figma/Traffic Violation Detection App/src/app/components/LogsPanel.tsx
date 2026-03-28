import { useEffect, useRef } from 'react';
import { Terminal, Info, AlertCircle, CheckCircle } from 'lucide-react';
import { ScrollArea } from '@/app/components/ui/scroll-area';

export interface LogEntry {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'success' | 'error';
  message: string;
}

interface LogsPanelProps {
  logs: LogEntry[];
}

export function LogsPanel({ logs }: LogsPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const getLogIcon = (level: LogEntry['level']) => {
    switch (level) {
      case 'info':
        return <Info className="w-3 h-3 text-blue-400" />;
      case 'warning':
        return <AlertCircle className="w-3 h-3 text-orange-400" />;
      case 'success':
        return <CheckCircle className="w-3 h-3 text-green-400" />;
      case 'error':
        return <AlertCircle className="w-3 h-3 text-red-400" />;
    }
  };

  const getLogColor = (level: LogEntry['level']) => {
    switch (level) {
      case 'info':
        return 'text-blue-400';
      case 'warning':
        return 'text-orange-400';
      case 'success':
        return 'text-green-400';
      case 'error':
        return 'text-red-400';
    }
  };

  return (
    <div className="h-48 bg-[#0a0a0a] border-t border-[#2a2a2a] flex flex-col">
      <div className="h-10 bg-[#1a1a1a] border-b border-[#2a2a2a] flex items-center px-4 gap-2">
        <Terminal className="w-4 h-4 text-gray-500" />
        <span className="text-xs text-gray-400 font-mono">PROCESSING LOGS</span>
      </div>
      
      <ScrollArea className="flex-1">
        <div ref={scrollRef} className="p-2 font-mono text-xs">
          {logs.map((log) => (
            <div key={log.id} className="flex items-start gap-2 py-1 hover:bg-[#1a1a1a] px-2 -mx-2">
              <span className="text-gray-600 flex-shrink-0">{log.timestamp}</span>
              <div className="flex-shrink-0 mt-0.5">
                {getLogIcon(log.level)}
              </div>
              <span className={getLogColor(log.level)}>{log.message}</span>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
