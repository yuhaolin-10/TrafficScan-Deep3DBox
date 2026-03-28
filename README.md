# TrafficScan Deep3DBox UI

This folder is a new integration project built from:

- `D:\college\GraduationProject\testProject\TrafficScan_UI`
- `D:\college\GraduationProject\testProject\deep3dbox_independent_test`

## Goal

Use the existing Qt UI project as the main application shell, then gradually replace the current 3D vehicle geometry path with the Deep3DBox-based pipeline.

## Current State

- UI base has been copied from `TrafficScan_UI`
- Deep3DBox demo checkpoint has been copied into `external/deep3dbox_demo_model`
- Independent Deep3DBox test runner has been migrated to:
  - `src/tools/run_deep3dbox_illegal_test.py`

## Suggested Next Steps

1. Keep lane segmentation and GUI from the current UI project.
2. Wrap Deep3DBox into a detector class under `src/core/`.
3. Replace the current `VehicleDetector3D` call path step by step.
4. After single-image validation is stable, wire it into `ProcessingWorker` and the GUI batch flow.

## Quick Run

Deep3DBox independent verification:

```powershell
& 'D:\Anaconda3\shell\condabin\conda-hook.ps1'
conda activate depth-probe
python D:\college\GraduationProject\testProject\TrafficScan_Deep3DBox_UI\src\tools\run_deep3dbox_illegal_test.py
```

Qt UI:

```powershell
python D:\college\GraduationProject\testProject\TrafficScan_Deep3DBox_UI\src\run_gui_auto_batch.py
```
