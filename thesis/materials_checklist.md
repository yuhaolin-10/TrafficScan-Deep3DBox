# Thesis Materials Checklist

## Figures worth preparing

- System overall architecture diagram
- End-to-end processing flowchart
- Lane segmentation result examples
- YOLO vehicle detection result examples
- 3D bounding box projection examples
- Region rule configuration screenshot
- GUI main interface screenshot
- Database E-R diagram or table relation diagram
- Wrong-way / no-parking / no-non-motor example screenshots

## Tables worth preparing

- Hardware and software environment table
- Dataset composition table
- Training parameter table
- Detection and segmentation metric table
- Functional test case table
- Runtime efficiency table
- Error case summary table

## Metrics to fill in later

- Vehicle detection: Precision, Recall, mAP@0.5, mAP@0.5:0.95
- Lane segmentation: IoU, mIoU, Pixel Accuracy
- 3D estimation: orientation error, localization error, or qualitative effect comparison
- Violation recognition: accuracy, recall, false alarm rate, missed detection rate
- System efficiency: average image processing time, video FPS, database write latency

## Project evidence that can be cited in the thesis

- `src/core/lane_segmenter.py`
- `src/core/vehicle_detector_deep3dbox.py`
- `src/core/object_tracker.py`
- `src/core/violation_checker.py`
- `src/services/pipeline.py`
- `src/services/video_pipeline.py`
- `src/services/region_rule_engine.py`
- `src/services/database_manager.py`
- `src/gui/main_window.py`

## Places where real data should replace placeholders

- Abstract quantitative results
- Chapter 5 training parameter details
- Chapter 7 experiment result tables
- Final conclusion section
- Reference list
