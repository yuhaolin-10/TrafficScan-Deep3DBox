import { Checkbox } from '@/app/components/ui/checkbox';
import { ScrollArea } from '@/app/components/ui/scroll-area';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/app/components/ui/table';

export interface Violation {
  id: string;
  vehicleId: string;
  type: string;
  timestamp: string;
  confidence: number;
  camera: string;
  confirmed: boolean;
}

interface ViolationsTableProps {
  violations: Violation[];
  selectedViolationId: string | null;
  onSelectViolation: (id: string) => void;
  onConfirmViolation: (id: string, confirmed: boolean) => void;
}

export function ViolationsTable({
  violations,
  selectedViolationId,
  onSelectViolation,
  onConfirmViolation,
}: ViolationsTableProps) {
  return (
    <div className="flex flex-col h-full">
      <div className="h-10 bg-[#1a1a1a] border-b border-[#2a2a2a] flex items-center px-4">
        <span className="text-xs text-gray-400 font-mono">VIOLATION RECORDS</span>
        <span className="ml-2 text-xs text-orange-400 font-mono">
          {violations.length} detected
        </span>
      </div>
      
      <ScrollArea className="flex-1">
        <Table>
          <TableHeader>
            <TableRow className="border-[#2a2a2a] hover:bg-transparent">
              <TableHead className="text-gray-500 font-mono text-xs">Vehicle ID</TableHead>
              <TableHead className="text-gray-500 font-mono text-xs">Violation Type</TableHead>
              <TableHead className="text-gray-500 font-mono text-xs">Timestamp</TableHead>
              <TableHead className="text-gray-500 font-mono text-xs">Confidence</TableHead>
              <TableHead className="text-gray-500 font-mono text-xs">Camera</TableHead>
              <TableHead className="text-gray-500 font-mono text-xs text-center">Confirmed</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {violations.map((violation) => (
              <TableRow
                key={violation.id}
                onClick={() => onSelectViolation(violation.id)}
                className={`
                  border-[#2a2a2a] cursor-pointer transition-colors font-mono text-xs
                  ${selectedViolationId === violation.id
                    ? 'bg-blue-500/20 hover:bg-blue-500/20'
                    : 'hover:bg-[#1a1a1a]'
                  }
                `}
              >
                <TableCell className="text-gray-300">{violation.vehicleId}</TableCell>
                <TableCell className="text-orange-400">{violation.type}</TableCell>
                <TableCell className="text-gray-400">{violation.timestamp}</TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-[#2a2a2a] h-1.5 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${
                          violation.confidence >= 0.9
                            ? 'bg-green-500'
                            : violation.confidence >= 0.7
                            ? 'bg-orange-500'
                            : 'bg-red-500'
                        }`}
                        style={{ width: `${violation.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-gray-400 w-10 text-right">
                      {(violation.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </TableCell>
                <TableCell className="text-gray-400">{violation.camera}</TableCell>
                <TableCell>
                  <div className="flex justify-center" onClick={(e) => e.stopPropagation()}>
                    <Checkbox
                      checked={violation.confirmed}
                      onCheckedChange={(checked) => 
                        onConfirmViolation(violation.id, checked as boolean)
                      }
                    />
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </ScrollArea>
    </div>
  );
}
