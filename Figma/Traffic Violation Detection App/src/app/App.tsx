import { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/app/components/ui/tabs';
import { Toolbar } from '@/app/components/Toolbar';
import { WorkspacePanel, WorkspaceFile } from '@/app/components/WorkspacePanel';
import { ImageViewer } from '@/app/components/ImageViewer';
import { LogsPanel, LogEntry } from '@/app/components/LogsPanel';
import { ViolationsTable, Violation } from '@/app/components/ViolationsTable';
import { HistoricalRecords } from '@/app/components/HistoricalRecords';

// Mock data for demonstration
const mockFiles: WorkspaceFile[] = [
  {
    id: 'file-1',
    filename: 'CAM-A-20260115-143022.jpg',
    camera: 'CAM-A-NORTH',
    timestamp: '2026-01-15 14:30:22',
    status: 'completed',
    thumbnail: 'https://images.unsplash.com/photo-1733149086985-7e607db85843?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxoaWdod2F5JTIwdHJhZmZpYyUyMHN1cnZlaWxsYW5jZXxlbnwxfHx8fDE3Njg0NjUzODB8MA&ixlib=rb-4.1.0&q=80&w=400',
    violations: 2,
  },
  {
    id: 'file-2',
    filename: 'CAM-B-20260115-143045.jpg',
    camera: 'CAM-B-SOUTH',
    timestamp: '2026-01-15 14:30:45',
    status: 'processing',
    thumbnail: 'https://images.unsplash.com/photo-1691635188006-78cfe07dadcc?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxoaWdod2F5JTIwY2FycyUyMGFlcmlhbHxlbnwxfHx8fDE3Njg0NjUzODB8MA&ixlib=rb-4.1.0&q=80&w=400',
  },
  {
    id: 'file-3',
    filename: 'CAM-A-20260115-143103.jpg',
    camera: 'CAM-A-NORTH',
    timestamp: '2026-01-15 14:31:03',
    status: 'pending',
    thumbnail: 'https://images.unsplash.com/photo-1766910094822-0da58433f557?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx0cmFmZmljJTIwY2FtZXJhJTIwdmlld3xlbnwxfHx8fDE3Njg0NjUzODB8MA&ixlib=rb-4.1.0&q=80&w=400',
  },
  {
    id: 'file-4',
    filename: 'CAM-C-20260115-143125.jpg',
    camera: 'CAM-C-EAST',
    timestamp: '2026-01-15 14:31:25',
    status: 'completed',
    thumbnail: 'https://images.unsplash.com/photo-1707905772191-b1f1098c4657?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxyb2FkJTIwdmVoaWNsZXMlMjBvdmVyaGVhZHxlbnwxfHx8fDE3Njg0NjUzODF8MA&ixlib=rb-4.1.0&q=80&w=400',
    violations: 0,
  },
  {
    id: 'file-5',
    filename: 'CAM-A-20260115-143200.jpg',
    camera: 'CAM-A-NORTH',
    timestamp: '2026-01-15 14:32:00',
    status: 'error',
    thumbnail: 'https://images.unsplash.com/photo-1733149086985-7e607db85843?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxoaWdod2F5JTIwdHJhZmZpYyUyMHN1cnZlaWxsYW5jZXxlbnwxfHx8fDE3Njg0NjUzODB8MA&ixlib=rb-4.1.0&q=80&w=400',
  },
];

const mockDetections = [
  {
    id: 'V-001',
    type: 'sedan',
    confidence: 0.94,
    bounds: { x: 450, y: 580, width: 180, height: 120 },
    isViolation: false,
    distance: 45,
  },
  {
    id: 'V-002',
    type: 'suv',
    confidence: 0.89,
    bounds: { x: 720, y: 520, width: 200, height: 140 },
    isViolation: false,
    distance: 52,
  },
  {
    id: 'V-003',
    type: 'truck',
    confidence: 0.92,
    bounds: { x: 1050, y: 480, width: 220, height: 160 },
    isViolation: false,
    distance: 68,
  },
  {
    id: 'V-004',
    type: 'sedan',
    confidence: 0.91,
    bounds: { x: 1420, y: 620, width: 160, height: 110 },
    isViolation: true,
    distance: 38,
  },
  {
    id: 'V-005',
    type: 'motorcycle',
    confidence: 0.87,
    bounds: { x: 1480, y: 720, width: 90, height: 85 },
    isViolation: true,
    distance: 28,
  },
];

const mockViolations: Violation[] = [
  {
    id: 'V-004',
    vehicleId: 'V-004',
    type: 'Emergency Lane Violation',
    timestamp: '2026-01-15 14:30:22',
    confidence: 0.91,
    camera: 'CAM-A-NORTH',
    confirmed: false,
  },
  {
    id: 'V-005',
    vehicleId: 'V-005',
    type: 'Emergency Lane Violation',
    timestamp: '2026-01-15 14:30:22',
    confidence: 0.87,
    camera: 'CAM-A-NORTH',
    confirmed: false,
  },
];

const initialLogs: LogEntry[] = [
  {
    id: '1',
    timestamp: '14:30:18',
    level: 'info',
    message: 'System initialized. Model: YOLOv8-Traffic-v3.2',
  },
  {
    id: '2',
    timestamp: '14:30:19',
    level: 'info',
    message: 'GPU acceleration enabled (CUDA 12.1)',
  },
  {
    id: '3',
    timestamp: '14:30:22',
    level: 'info',
    message: 'Processing: CAM-A-20260115-143022.jpg',
  },
  {
    id: '4',
    timestamp: '14:30:23',
    level: 'success',
    message: 'Detected 5 vehicles in frame',
  },
  {
    id: '5',
    timestamp: '14:30:24',
    level: 'warning',
    message: 'Emergency lane violation detected: V-004 (confidence: 91%)',
  },
  {
    id: '6',
    timestamp: '14:30:24',
    level: 'warning',
    message: 'Emergency lane violation detected: V-005 (confidence: 87%)',
  },
  {
    id: '7',
    timestamp: '14:30:25',
    level: 'success',
    message: 'Analysis complete: 2 violations found',
  },
];

const mockHistoricalRecords = [
  {
    id: 'hist-1',
    filename: 'CAM-A-20260115-120000.jpg',
    camera: 'CAM-A-NORTH',
    timestamp: '2026-01-15 12:00:00',
    violations: 1,
    originalImage: 'https://images.unsplash.com/photo-1733149086985-7e607db85843?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=800',
    processedImage: 'https://images.unsplash.com/photo-1733149086985-7e607db85843?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=800',
  },
  {
    id: 'hist-2',
    filename: 'CAM-B-20260115-113000.jpg',
    camera: 'CAM-B-SOUTH',
    timestamp: '2026-01-15 11:30:00',
    violations: 0,
    originalImage: 'https://images.unsplash.com/photo-1691635188006-78cfe07dadcc?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=800',
    processedImage: 'https://images.unsplash.com/photo-1691635188006-78cfe07dadcc?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=800',
  },
  {
    id: 'hist-3',
    filename: 'CAM-A-20260115-110000.jpg',
    camera: 'CAM-A-NORTH',
    timestamp: '2026-01-15 11:00:00',
    violations: 3,
    originalImage: 'https://images.unsplash.com/photo-1766910094822-0da58433f557?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=800',
    processedImage: 'https://images.unsplash.com/photo-1766910094822-0da58433f557?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=800',
  },
  {
    id: 'hist-4',
    filename: 'CAM-C-20260115-103000.jpg',
    camera: 'CAM-C-EAST',
    timestamp: '2026-01-15 10:30:00',
    violations: 0,
    originalImage: 'https://images.unsplash.com/photo-1707905772191-b1f1098c4657?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=800',
    processedImage: 'https://images.unsplash.com/photo-1707905772191-b1f1098c4657?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=800',
  },
];

export default function App() {
  const [selectedFileId, setSelectedFileId] = useState<string | null>('file-1');
  const [selectedViolationId, setSelectedViolationId] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>(initialLogs);
  const [violations, setViolations] = useState<Violation[]>(mockViolations);
  const [activeView, setActiveView] = useState<'analysis' | 'history'>('analysis');

  // Apply dark theme
  useEffect(() => {
    document.documentElement.classList.add('dark');
  }, []);

  const selectedFile = mockFiles.find(f => f.id === selectedFileId);
  const processingCount = mockFiles.filter(f => f.status === 'processing').length;

  const handleRunAll = () => {
    const newLog: LogEntry = {
      id: String(logs.length + 1),
      timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }).slice(0, 5),
      level: 'info',
      message: 'Batch processing started for pending files',
    };
    setLogs([...logs, newLog]);
  };

  const handleConfirmViolation = (id: string, confirmed: boolean) => {
    setViolations(violations.map(v =>
      v.id === id ? { ...v, confirmed } : v
    ));
  };

  return (
    <div className="h-screen flex flex-col bg-[#0f0f0f] text-gray-300">
      <Toolbar onRunAll={handleRunAll} processingCount={processingCount} />
      
      <div className="flex-1 flex overflow-hidden">
        {activeView === 'analysis' ? (
          <>
            <WorkspacePanel
              files={mockFiles}
              selectedFileId={selectedFileId}
              onSelectFile={setSelectedFileId}
            />
            
            <div className="flex-1 flex flex-col overflow-hidden">
              <Tabs defaultValue="viewer" className="flex-1 flex flex-col">
                <div className="h-10 bg-[#1a1a1a] border-b border-[#2a2a2a] px-4">
                  <TabsList className="h-10 bg-transparent border-0 p-0 gap-0">
                    <TabsTrigger
                      value="viewer"
                      className="data-[state=active]:bg-[#2a2a2a] data-[state=active]:text-white text-gray-400 rounded-none border-b-2 border-transparent data-[state=active]:border-blue-500 px-4"
                    >
                      IMAGE ANALYSIS
                    </TabsTrigger>
                    <TabsTrigger
                      value="violations"
                      className="data-[state=active]:bg-[#2a2a2a] data-[state=active]:text-white text-gray-400 rounded-none border-b-2 border-transparent data-[state=active]:border-blue-500 px-4"
                    >
                      VIOLATIONS
                    </TabsTrigger>
                    <TabsTrigger
                      value="history"
                      className="data-[state=active]:bg-[#2a2a2a] data-[state=active]:text-white text-gray-400 rounded-none border-b-2 border-transparent data-[state=active]:border-blue-500 px-4"
                      onClick={() => setActiveView('history')}
                    >
                      HISTORICAL RECORDS
                    </TabsTrigger>
                  </TabsList>
                </div>
                
                <TabsContent value="viewer" className="flex-1 m-0">
                  {selectedFile && selectedFile.status === 'completed' ? (
                    <ImageViewer
                      imageUrl={selectedFile.thumbnail.replace('w=400', 'w=1920')}
                      detections={mockDetections}
                      selectedViolationId={selectedViolationId}
                    />
                  ) : (
                    <div className="flex-1 flex items-center justify-center">
                      <div className="text-center">
                        <p className="text-sm text-gray-500">
                          {selectedFile?.status === 'processing' && 'Processing...'}
                          {selectedFile?.status === 'pending' && 'Click "Run All" to process this file'}
                          {selectedFile?.status === 'error' && 'Error processing this file'}
                          {!selectedFile && 'Select a file to view'}
                        </p>
                      </div>
                    </div>
                  )}
                </TabsContent>
                
                <TabsContent value="violations" className="flex-1 m-0">
                  <ViolationsTable
                    violations={violations}
                    selectedViolationId={selectedViolationId}
                    onSelectViolation={setSelectedViolationId}
                    onConfirmViolation={handleConfirmViolation}
                  />
                </TabsContent>
              </Tabs>
              
              <LogsPanel logs={logs} />
            </div>
          </>
        ) : (
          <div className="flex-1">
            <div className="h-10 bg-[#1a1a1a] border-b border-[#2a2a2a] px-4 flex items-center">
              <button
                onClick={() => setActiveView('analysis')}
                className="text-xs text-gray-400 hover:text-white font-mono"
              >
                ← BACK TO ANALYSIS
              </button>
            </div>
            <div className="h-[calc(100%-2.5rem)]">
              <HistoricalRecords records={mockHistoricalRecords} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}