import { useState } from 'react';
import { Upload, Clock, CheckCircle2, XCircle, Loader2, Camera } from 'lucide-react';
import { ScrollArea } from '@/app/components/ui/scroll-area';

export interface WorkspaceFile {
  id: string;
  filename: string;
  camera: string;
  timestamp: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  thumbnail: string;
  violations?: number;
}

interface WorkspacePanelProps {
  files: WorkspaceFile[];
  selectedFileId: string | null;
  onSelectFile: (id: string) => void;
}

export function WorkspacePanel({ files, selectedFileId, onSelectFile }: WorkspacePanelProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    // Handle file drop
  };

  const getStatusIcon = (status: WorkspaceFile['status']) => {
    switch (status) {
      case 'pending':
        return <Clock className="w-3.5 h-3.5 text-gray-500" />;
      case 'processing':
        return <Loader2 className="w-3.5 h-3.5 text-blue-400 animate-spin" />;
      case 'completed':
        return <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />;
      case 'error':
        return <XCircle className="w-3.5 h-3.5 text-red-500" />;
    }
  };

  const getStatusColor = (status: WorkspaceFile['status']) => {
    switch (status) {
      case 'pending':
        return 'text-gray-400';
      case 'processing':
        return 'text-blue-400';
      case 'completed':
        return 'text-green-400';
      case 'error':
        return 'text-red-400';
    }
  };

  return (
    <div className="w-80 bg-[#1a1a1a] border-r border-[#2a2a2a] flex flex-col">
      <div className="p-4 border-b border-[#2a2a2a]">
        <h2 className="text-sm text-gray-400 mb-3">WORKSPACE</h2>
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`
            border-2 border-dashed rounded p-4 text-center transition-colors
            ${isDragging ? 'border-blue-500 bg-blue-500/5' : 'border-[#2a2a2a] bg-[#0f0f0f]'}
          `}
        >
          <Upload className="w-5 h-5 text-gray-500 mx-auto mb-2" />
          <p className="text-xs text-gray-500">Drop images here</p>
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-2">
          {files.map((file) => (
            <button
              key={file.id}
              onClick={() => onSelectFile(file.id)}
              className={`
                w-full p-2 rounded mb-1 text-left transition-colors
                ${selectedFileId === file.id ? 'bg-blue-500/20 border border-blue-500/50' : 'hover:bg-[#2a2a2a]'}
              `}
            >
              <div className="flex gap-2">
                <div className="w-16 h-16 bg-[#0f0f0f] rounded overflow-hidden flex-shrink-0">
                  <img
                    src={file.thumbnail}
                    alt={file.filename}
                    className="w-full h-full object-cover"
                  />
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-1 mb-1">
                    <div className="font-mono text-xs text-gray-300 truncate">
                      {file.filename}
                    </div>
                    {getStatusIcon(file.status)}
                  </div>
                  
                  <div className="flex items-center gap-1 mb-1">
                    <Camera className="w-3 h-3 text-gray-600" />
                    <span className="text-xs text-gray-500">{file.camera}</span>
                  </div>
                  
                  <div className="text-xs text-gray-600 font-mono mb-1">
                    {file.timestamp}
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className={`text-xs font-mono ${getStatusColor(file.status)}`}>
                      {file.status.toUpperCase()}
                    </span>
                    {file.violations !== undefined && file.status === 'completed' && (
                      <span className="text-xs text-orange-400">
                        {file.violations} violation{file.violations !== 1 ? 's' : ''}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>
      </ScrollArea>
      
      <div className="p-3 border-t border-[#2a2a2a] text-xs text-gray-500 font-mono">
        {files.length} file{files.length !== 1 ? 's' : ''} loaded
      </div>
    </div>
  );
}
