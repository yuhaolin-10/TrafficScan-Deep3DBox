import { useState } from 'react';
import { Calendar, Camera, Filter, Download, FileText, Image as ImageIcon } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Input } from '@/app/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/app/components/ui/select';
import { ScrollArea } from '@/app/components/ui/scroll-area';

interface HistoricalRecord {
  id: string;
  filename: string;
  camera: string;
  timestamp: string;
  violations: number;
  originalImage: string;
  processedImage: string;
}

interface HistoricalRecordsProps {
  records: HistoricalRecord[];
}

export function HistoricalRecords({ records }: HistoricalRecordsProps) {
  const [selectedRecord, setSelectedRecord] = useState<HistoricalRecord | null>(null);
  const [cameraFilter, setCameraFilter] = useState<string>('all');
  const [violationFilter, setViolationFilter] = useState<string>('all');

  const filteredRecords = records.filter(record => {
    if (cameraFilter !== 'all' && record.camera !== cameraFilter) return false;
    if (violationFilter === 'violations-only' && record.violations === 0) return false;
    return true;
  });

  const cameras = Array.from(new Set(records.map(r => r.camera)));

  return (
    <div className="flex h-full">
      {/* Records List */}
      <div className="w-96 bg-[#1a1a1a] border-r border-[#2a2a2a] flex flex-col">
        <div className="p-4 border-b border-[#2a2a2a]">
          <h2 className="text-sm text-gray-400 mb-4 font-mono">HISTORICAL RECORDS</h2>
          
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Calendar className="w-4 h-4 text-gray-500" />
              <Input
                type="date"
                className="flex-1 bg-[#0f0f0f] border-[#2a2a2a] text-gray-300 text-xs"
                defaultValue="2026-01-15"
              />
            </div>
            
            <div className="flex items-center gap-2">
              <Camera className="w-4 h-4 text-gray-500" />
              <Select value={cameraFilter} onValueChange={setCameraFilter}>
                <SelectTrigger className="flex-1 bg-[#0f0f0f] border-[#2a2a2a] text-gray-300 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Cameras</SelectItem>
                  {cameras.map(camera => (
                    <SelectItem key={camera} value={camera}>{camera}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-gray-500" />
              <Select value={violationFilter} onValueChange={setViolationFilter}>
                <SelectTrigger className="flex-1 bg-[#0f0f0f] border-[#2a2a2a] text-gray-300 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Records</SelectItem>
                  <SelectItem value="violations-only">Violations Only</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        <ScrollArea className="flex-1">
          <div className="p-2">
            {filteredRecords.map((record) => (
              <button
                key={record.id}
                onClick={() => setSelectedRecord(record)}
                className={`
                  w-full p-3 rounded mb-1 text-left transition-colors
                  ${selectedRecord?.id === record.id
                    ? 'bg-blue-500/20 border border-blue-500/50'
                    : 'hover:bg-[#2a2a2a]'
                  }
                `}
              >
                <div className="flex gap-2 mb-2">
                  <div className="w-20 h-20 bg-[#0f0f0f] rounded overflow-hidden flex-shrink-0">
                    <img
                      src={record.originalImage}
                      alt={record.filename}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="font-mono text-xs text-gray-300 truncate mb-1">
                      {record.filename}
                    </div>
                    <div className="text-xs text-gray-500 mb-1">
                      {record.camera}
                    </div>
                    <div className="text-xs text-gray-600 font-mono">
                      {record.timestamp}
                    </div>
                  </div>
                </div>
                
                {record.violations > 0 && (
                  <div className="text-xs text-orange-400 font-mono">
                    {record.violations} violation{record.violations !== 1 ? 's' : ''}
                  </div>
                )}
              </button>
            ))}
          </div>
        </ScrollArea>
        
        <div className="p-3 border-t border-[#2a2a2a]">
          <div className="flex gap-2">
            <Button className="flex-1 bg-[#2a2a2a] hover:bg-[#3a3a3a] text-gray-300 gap-2 text-xs">
              <FileText className="w-3.5 h-3.5" />
              Export CSV
            </Button>
            <Button className="flex-1 bg-[#2a2a2a] hover:bg-[#3a3a3a] text-gray-300 gap-2 text-xs">
              <Download className="w-3.5 h-3.5" />
              Export Images
            </Button>
          </div>
        </div>
      </div>

      {/* Comparison View */}
      <div className="flex-1 bg-[#0f0f0f] flex flex-col">
        {selectedRecord ? (
          <>
            <div className="h-12 bg-[#1a1a1a] border-b border-[#2a2a2a] flex items-center px-4">
              <span className="text-xs text-gray-400 font-mono">
                COMPARISON VIEW - {selectedRecord.filename}
              </span>
            </div>
            
            <div className="flex-1 overflow-auto">
              <div className="grid grid-cols-2 gap-4 p-4 h-full">
                <div className="flex flex-col">
                  <div className="text-xs text-gray-400 font-mono mb-2 flex items-center gap-2">
                    <ImageIcon className="w-3.5 h-3.5" />
                    ORIGINAL IMAGE
                  </div>
                  <div className="flex-1 bg-[#1a1a1a] rounded border border-[#2a2a2a] overflow-hidden">
                    <img
                      src={selectedRecord.originalImage}
                      alt="Original"
                      className="w-full h-full object-contain"
                    />
                  </div>
                </div>
                
                <div className="flex flex-col">
                  <div className="text-xs text-gray-400 font-mono mb-2 flex items-center gap-2">
                    <ImageIcon className="w-3.5 h-3.5" />
                    PROCESSED IMAGE
                  </div>
                  <div className="flex-1 bg-[#1a1a1a] rounded border border-[#2a2a2a] overflow-hidden">
                    <img
                      src={selectedRecord.processedImage}
                      alt="Processed"
                      className="w-full h-full object-contain"
                    />
                  </div>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <ImageIcon className="w-12 h-12 text-gray-700 mx-auto mb-3" />
              <p className="text-sm text-gray-500">Select a record to view comparison</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
