import { Play, Settings, Database, Activity, Circle } from 'lucide-react';
import { Button } from '@/app/components/ui/button';

interface ToolbarProps {
  onRunAll: () => void;
  processingCount: number;
}

export function Toolbar({ onRunAll, processingCount }: ToolbarProps) {
  return (
    <div className="h-14 bg-[#1a1a1a] border-b border-[#2a2a2a] flex items-center justify-between px-4">
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <div className="text-sm font-mono text-white">TVD-SYSTEM</div>
          <div className="text-xs text-gray-500">v2.4.1</div>
        </div>
        
        <div className="h-6 w-px bg-[#2a2a2a]" />
        
        <div className="flex items-center gap-2">
          <Circle className="w-2 h-2 fill-green-500 text-green-500" />
          <span className="text-xs text-gray-400">Model Loaded</span>
          <span className="text-xs text-gray-500 font-mono">YOLOv8-Traffic-v3.2</span>
        </div>
      </div>
      
      <div className="flex items-center gap-2">
        {processingCount > 0 && (
          <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-500/10 border border-blue-500/30 rounded">
            <Activity className="w-3.5 h-3.5 text-blue-400 animate-pulse" />
            <span className="text-xs text-blue-400 font-mono">{processingCount} processing</span>
          </div>
        )}
        
        <Button
          onClick={onRunAll}
          className="bg-blue-600 hover:bg-blue-700 text-white gap-2 h-9"
        >
          <Play className="w-4 h-4" />
          Run All
        </Button>
        
        <div className="w-px h-6 bg-[#2a2a2a]" />
        
        <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white">
          <Database className="w-4 h-4" />
        </Button>
        
        <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white">
          <Settings className="w-4 h-4" />
        </Button>
      </div>
    </div>
  );
}
