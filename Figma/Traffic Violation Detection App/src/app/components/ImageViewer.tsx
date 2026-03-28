import { useState } from 'react';
import { ZoomIn, ZoomOut, Maximize2, Eye, Box, Navigation, AlertTriangle } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Switch } from '@/app/components/ui/switch';
import { Label } from '@/app/components/ui/label';

interface Detection {
  id: string;
  type: string;
  confidence: number;
  bounds: { x: number; y: number; width: number; height: number };
  isViolation: boolean;
  distance?: number;
}

interface ImageViewerProps {
  imageUrl: string;
  detections: Detection[];
  selectedViolationId: string | null;
}

export function ImageViewer({ imageUrl, detections, selectedViolationId }: ImageViewerProps) {
  const [zoom, setZoom] = useState(100);
  const [show2DBoxes, setShow2DBoxes] = useState(true);
  const [show3DBoxes, setShow3DBoxes] = useState(true);
  const [showLanes, setShowLanes] = useState(true);
  const [showViolations, setShowViolations] = useState(true);

  return (
    <div className="flex-1 bg-[#0f0f0f] flex flex-col">
      {/* Controls Bar */}
      <div className="h-12 bg-[#1a1a1a] border-b border-[#2a2a2a] flex items-center justify-between px-4">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-gray-400 hover:text-white"
              onClick={() => setZoom(Math.max(50, zoom - 10))}
            >
              <ZoomOut className="w-4 h-4" />
            </Button>
            <span className="text-xs font-mono text-gray-400 w-12 text-center">
              {zoom}%
            </span>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-gray-400 hover:text-white"
              onClick={() => setZoom(Math.min(200, zoom + 10))}
            >
              <ZoomIn className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-gray-400 hover:text-white"
              onClick={() => setZoom(100)}
            >
              <Maximize2 className="w-4 h-4" />
            </Button>
          </div>
          
          <div className="h-4 w-px bg-[#2a2a2a]" />
          
          <div className="text-xs text-gray-500 font-mono">
            {detections.length} objects detected
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Switch
              id="2d-boxes"
              checked={show2DBoxes}
              onCheckedChange={setShow2DBoxes}
            />
            <Label htmlFor="2d-boxes" className="text-xs text-gray-400 cursor-pointer">
              2D Boxes
            </Label>
          </div>
          
          <div className="flex items-center gap-2">
            <Switch
              id="3d-boxes"
              checked={show3DBoxes}
              onCheckedChange={setShow3DBoxes}
            />
            <Label htmlFor="3d-boxes" className="text-xs text-gray-400 cursor-pointer">
              3D Boxes
            </Label>
          </div>
          
          <div className="flex items-center gap-2">
            <Switch
              id="lanes"
              checked={showLanes}
              onCheckedChange={setShowLanes}
            />
            <Label htmlFor="lanes" className="text-xs text-gray-400 cursor-pointer">
              Lanes
            </Label>
          </div>
          
          <div className="flex items-center gap-2">
            <Switch
              id="violations"
              checked={showViolations}
              onCheckedChange={setShowViolations}
            />
            <Label htmlFor="violations" className="text-xs text-gray-400 cursor-pointer">
              Violations
            </Label>
          </div>
        </div>
      </div>

      {/* Image Display Area */}
      <div className="flex-1 overflow-auto relative">
        <div className="p-8">
          <div
            className="relative mx-auto"
            style={{
              width: `${zoom}%`,
              maxWidth: '1920px',
            }}
          >
            <img
              src={imageUrl}
              alt="Traffic surveillance"
              className="w-full h-auto"
            />
            
            {/* Overlay Canvas for Detections */}
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none"
              viewBox="0 0 1920 1080"
              preserveAspectRatio="none"
            >
              {/* Lane markings */}
              {showLanes && (
                <g>
                  <line
                    x1="400"
                    y1="1080"
                    x2="600"
                    y2="400"
                    stroke="#3b82f6"
                    strokeWidth="2"
                    strokeDasharray="10,5"
                    opacity="0.4"
                  />
                  <line
                    x1="800"
                    y1="1080"
                    x2="900"
                    y2="400"
                    stroke="#3b82f6"
                    strokeWidth="2"
                    strokeDasharray="10,5"
                    opacity="0.4"
                  />
                  <line
                    x1="1200"
                    y1="1080"
                    x2="1200"
                    y2="400"
                    stroke="#3b82f6"
                    strokeWidth="2"
                    strokeDasharray="10,5"
                    opacity="0.4"
                  />
                  <line
                    x1="1500"
                    y1="1080"
                    x2="1400"
                    y2="400"
                    stroke="#f59e0b"
                    strokeWidth="3"
                    opacity="0.5"
                  />
                  <text
                    x="1450"
                    y="900"
                    fill="#f59e0b"
                    fontSize="14"
                    opacity="0.7"
                  >
                    EMERGENCY LANE
                  </text>
                </g>
              )}
              
              {/* Detection boxes */}
              {detections.map((detection) => {
                const isSelected = selectedViolationId === detection.id;
                const strokeColor = detection.isViolation
                  ? (showViolations ? '#ef4444' : '#3b82f6')
                  : '#3b82f6';
                const strokeWidth = isSelected ? 3 : 2;
                const opacity = isSelected ? 1 : 0.7;
                
                return (
                  <g key={detection.id}>
                    {/* 2D Bounding Box */}
                    {show2DBoxes && (
                      <rect
                        x={detection.bounds.x}
                        y={detection.bounds.y}
                        width={detection.bounds.width}
                        height={detection.bounds.height}
                        fill="none"
                        stroke={strokeColor}
                        strokeWidth={strokeWidth}
                        opacity={opacity}
                      />
                    )}
                    
                    {/* 3D Wireframe Box */}
                    {show3DBoxes && (
                      <g opacity={opacity * 0.6}>
                        <polygon
                          points={`
                            ${detection.bounds.x + 10},${detection.bounds.y + 10}
                            ${detection.bounds.x + detection.bounds.width - 10},${detection.bounds.y + 10}
                            ${detection.bounds.x + detection.bounds.width},${detection.bounds.y + detection.bounds.height - 20}
                            ${detection.bounds.x},${detection.bounds.y + detection.bounds.height - 20}
                          `}
                          fill="none"
                          stroke={strokeColor}
                          strokeWidth="1"
                          strokeDasharray="3,2"
                        />
                        <line
                          x1={detection.bounds.x}
                          y1={detection.bounds.y}
                          x2={detection.bounds.x + 10}
                          y2={detection.bounds.y + 10}
                          stroke={strokeColor}
                          strokeWidth="1"
                        />
                        <line
                          x1={detection.bounds.x + detection.bounds.width}
                          y1={detection.bounds.y}
                          x2={detection.bounds.x + detection.bounds.width - 10}
                          y2={detection.bounds.y + 10}
                          stroke={strokeColor}
                          strokeWidth="1"
                        />
                      </g>
                    )}
                    
                    {/* Label */}
                    <g>
                      <rect
                        x={detection.bounds.x}
                        y={detection.bounds.y - 22}
                        width={120}
                        height={20}
                        fill="#000000"
                        opacity="0.8"
                      />
                      <text
                        x={detection.bounds.x + 4}
                        y={detection.bounds.y - 8}
                        fill={strokeColor}
                        fontSize="12"
                        fontFamily="monospace"
                      >
                        {detection.id} | {(detection.confidence * 100).toFixed(1)}%
                      </text>
                    </g>
                    
                    {/* Distance label */}
                    {detection.distance && (
                      <text
                        x={detection.bounds.x + detection.bounds.width / 2}
                        y={detection.bounds.y + detection.bounds.height + 16}
                        fill="#ffffff"
                        fontSize="11"
                        fontFamily="monospace"
                        textAnchor="middle"
                        opacity="0.7"
                      >
                        {detection.distance}m
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-black/80 border border-[#2a2a2a] rounded p-3">
        <div className="text-xs text-gray-400 mb-2 font-mono">LEGEND</div>
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <div className="w-4 h-1 bg-blue-500" />
            <span className="text-xs text-gray-400">Normal Vehicle</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-1 bg-red-500" />
            <span className="text-xs text-gray-400">Violation Detected</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-1 bg-orange-500" />
            <span className="text-xs text-gray-400">Emergency Lane</span>
          </div>
        </div>
      </div>
    </div>
  );
}
