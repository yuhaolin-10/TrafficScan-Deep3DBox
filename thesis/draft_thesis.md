# Design and Implementation of a Vehicle Violation Recognition System Based on YOLO, Lane Segmentation, and 3D Object Detection

## Abstract

With the continuous growth of urban traffic volume, traditional manual monitoring methods have become increasingly insufficient in terms of efficiency, real-time response, and large-scale deployment cost. To improve the automation level of intelligent traffic supervision, this project designs and implements a vehicle violation recognition system oriented to traffic scenes. The system takes YOLO-based vehicle detection as the core visual perception entry, introduces lane segmentation to obtain road region information, and combines monocular 3D bounding box estimation to enrich the spatial description of vehicles. On this basis, a region rule engine is constructed to support violation reasoning for multiple scenarios such as emergency lane occupation, no parking, no non-motor vehicle entry, and wrong-way driving. At the engineering level, the project also integrates video processing, object tracking, plate recognition, graphical user interface, and SQLite-based result persistence, forming a complete system from perception to decision and presentation.

According to the current implementation, the system has completed the integration of image and video processing pipelines, manual and automatic lane region acquisition, multi-frame track association, region-based rule judgment, visualization rendering, and structured storage of analysis results. Experimental results can later be supplemented with real metrics such as vehicle detection precision, lane segmentation IoU, and end-to-end violation recognition accuracy. The completed system provides a practical reference for the application of deep learning and intelligent vision technology in traffic violation analysis.

Keywords: YOLO, lane segmentation, monocular 3D object detection, Deep3DBox, vehicle violation recognition, intelligent transportation

## Abstract in English

To improve the automation capability of intelligent traffic supervision, this project designs and implements a vehicle violation recognition system for complex road scenes. The system uses YOLO-based vehicle detection as the perception backbone, lane segmentation for road region extraction, and monocular 3D bounding box estimation to provide richer spatial cues for vehicle analysis. A region rule engine is further introduced to determine different traffic violations, including emergency lane occupation, no parking, no non-motor vehicle entry, and wrong-way driving. In engineering practice, the system integrates image and video processing, object tracking, plate recognition, a Qt-based graphical interface, and SQLite result persistence, forming a complete solution from visual perception to violation decision and result management.

The current project has completed the core functional pipeline and can be extended with real experimental indicators in the final thesis, such as mAP, IoU, processing speed, and violation recognition accuracy. The work demonstrates the feasibility of combining deep learning perception modules with engineering-oriented rule reasoning for traffic scene analysis.

## 1. Introduction

### 1.1 Research Background

In recent years, road traffic management has gradually shifted from manual experience-driven supervision to data-driven intelligent supervision. In urban expressways, ring roads, and major arterial roads, traffic violations such as emergency lane occupation, illegal parking, non-motor vehicle intrusion, and wrong-way driving not only affect traffic order, but may also cause severe congestion and safety accidents. Traditional supervision methods rely heavily on manual inspection, fixed-function equipment, or post-event review, which often suffer from delayed response, high labor cost, and limited scalability.

At the same time, the rapid development of deep learning has significantly improved the performance of computer vision systems in target detection, semantic segmentation, and scene understanding. The YOLO family has become a mainstream technical route for real-time traffic target detection due to its balance between speed and accuracy. In addition, lane segmentation can provide explicit road structure information, and monocular 3D detection can compensate for the lack of spatial understanding in conventional 2D vision systems. Therefore, building a violation recognition system around multi-module visual fusion has both theoretical value and engineering significance.

### 1.2 Research Significance

The significance of this topic is mainly reflected in three aspects. First, from the application perspective, the system can improve the automation level of traffic violation discovery and reduce the dependence on manual review. Second, from the technical perspective, the project explores the fusion of target detection, segmentation, monocular 3D estimation, tracking, and rule reasoning, which is more aligned with real intelligent transportation scenarios than using a single classification model. Third, from the engineering perspective, the work implements a full workflow including perception, analysis, visualization, and persistence, which is suitable for graduation design research on algorithm plus system implementation.

### 1.3 Research Status and Existing Problems

Existing traffic vision systems mostly focus on one of the following directions: vehicle detection, plate recognition, traffic sign recognition, or behavior analysis under specific scenarios. Many studies can achieve good performance on independent tasks, but practical deployment often faces several problems.

First, relying only on 2D bounding boxes is often insufficient for judging complex violations. For example, determining whether a vehicle truly occupies a special lane requires not only detecting the vehicle, but also calculating its overlap with lane regions. Second, traffic videos are dynamic scenes, and single-frame decisions are vulnerable to noise, occlusion, and temporary errors. Therefore, multi-frame tracking and temporal accumulation are necessary. Third, some violation behaviors, such as wrong-way driving and long-time parking, cannot be reliably judged by one frame alone and require explicit rule modeling. Finally, many algorithm studies stop at model comparison, while fewer works complete the engineering closed loop of interface interaction, result storage, and batch analysis.

### 1.4 Main Work of This Thesis

Based on the above problems, this thesis completes the following work around the current project:

1. A YOLO-based vehicle perception module is constructed to detect vehicles and related road users in traffic scenes.
2. A lane segmentation module is introduced to extract lane area masks and polygon contours for subsequent region analysis.
3. A monocular 3D box estimation method based on Deep3DBox is integrated to supplement vehicle orientation, dimensions, and projected ground footprint.
4. A violation determination strategy based on geometric overlap and a region rule engine is designed to support multiple violation types.
5. A complete system with image processing, video processing, GUI interaction, result rendering, and database persistence is implemented.

### 1.5 Thesis Organization

Chapter 1 introduces the research background, significance, current status, and main work. Chapter 2 presents the related theories and key technologies. Chapter 3 analyzes system requirements. Chapter 4 gives the overall system design. Chapter 5 elaborates the key algorithm design. Chapter 6 introduces the concrete implementation of the system. Chapter 7 presents experiment and test analysis. Chapter 8 summarizes the work and discusses future directions.

## 2. Related Theories and Key Technologies

### 2.1 YOLO-Based Target Detection

YOLO is a one-stage target detection framework that directly maps the input image to target category and location, thereby avoiding the high computational overhead of multi-stage proposal generation. In this project, the YOLO model is used as the front-end detector for vehicle perception. According to the current engineering configuration, the vehicle detection model path is `src/models/yolo11l.pt`, which indicates that a YOLO11 large model is used as the main 2D detection backbone. The detector outputs bounding boxes, confidences, and category labels, providing the basis for subsequent 3D estimation and rule analysis.

Compared with traditional methods, YOLO has two advantages in this project. One is stronger real-time capability, which is suitable for interactive GUI and video processing. The other is better engineering adaptability, since the detector can be directly integrated with Ultralytics inference interfaces and can be extended to multiple categories such as car, truck, bus, bicycle, motorcycle, and person.

### 2.2 Lane Segmentation

Lane segmentation aims to extract lane or drivable area regions from road images at the pixel or polygon level. In this project, `src/core/lane_segmenter.py` loads a trained YOLO segmentation model and outputs two types of results: a binary lane mask and polygon contours. This design is useful in engineering practice because the mask is convenient for pixel overlap calculation, while the polygon is convenient for visualization, geometric reasoning, and later manual editing.

The lane segmentation module provides explicit scene structure information. Without lane segmentation, the system can only detect the vehicle itself and cannot determine whether the detected vehicle is located inside a restricted lane or special road region. Therefore, the lane segmentation result is one of the key foundations for violation determination.

### 2.3 Monocular 3D Object Detection and Deep3DBox

Conventional 2D detection only provides planar image coordinates, while many traffic analysis tasks require richer spatial understanding, such as vehicle orientation, coarse depth relation, and footprint projection on the road plane. For this reason, the project introduces a monocular 3D box estimation method based on Deep3DBox. In the current codebase, `src/core/vehicle_detector_3d.py` has been deprecated and redirected to `src/core/vehicle_detector_deep3dbox.py`, showing that the project has shifted from an older 3D geometry path to a Deep3DBox-based pipeline.

The Deep3DBox module uses the 2D detection result as input, crops the object patch, predicts orientation and dimension offsets, and then solves the 3D location through projection geometry constraints. This process allows the system to estimate the 3D bounding box and derive useful information such as 2D projected corners, ground footprint, and yaw angle. For traffic violation judgment, the footprint is especially important because it gives a better approximation of the vehicle-road contact area than a plain 2D rectangle.

### 2.4 Object Tracking and Temporal Association

Traffic violation recognition in video scenes cannot rely solely on frame-level results. The project therefore introduces an `ObjectTracker` in `src/core/object_tracker.py` to associate detections across frames. The tracker uses spatial distance, bounding box geometry, and vehicle type compatibility to maintain track identities and accumulate hit counts, history anchors, and motion information.

Temporal association supports several key functions. First, it reduces the instability caused by occasional missed detections. Second, it allows the system to determine long-duration events such as illegal parking. Third, it provides motion direction information for wrong-way analysis. Therefore, tracking is the bridge between perception output and higher-level rule reasoning.

### 2.5 Rule-Based Violation Reasoning

A central feature of this project is that it does not treat all violations as a single end-to-end classification problem. Instead, it combines visual perception with explicit traffic rule reasoning. The project contains two layers of violation determination.

The first layer is geometric overlap judgment, implemented by `src/core/violation_checker.py`, which calculates the overlap ratio between the vehicle footprint and lane mask. This is mainly suitable for scenarios such as emergency lane occupation.

The second layer is region rule reasoning, implemented by `src/services/region_rule_engine.py`. The engine currently supports multiple rule types, including no parking, no non-motor vehicle entry, and no wrong-way driving. It uses polygon regions, allowed direction lines, overlap ratio, consecutive frame thresholds, and speed constraints to determine whether a track triggers a violation event. Compared with one-shot classification, this strategy is more explainable and more suitable for engineering customization.

### 2.6 Engineering Technologies

From the system engineering perspective, the project is built mainly with Python, OpenCV, Ultralytics YOLO, TensorFlow compatibility support for Deep3DBox, PyTorch runtime support, Qt GUI components, and SQLite database persistence. `src/gui/main_window.py` shows that the system provides a complete desktop GUI for media import, preview, region drawing, rule configuration, running analysis, and result viewing. `src/services/database_manager.py` implements structured storage for images, records, lane segments, detections, and review results, enabling later query and management.

## 3. System Requirements Analysis

### 3.1 System Goals

The goal of the system is to build an integrated traffic scene analysis platform that can process road images and videos, automatically detect vehicles and road regions, recognize potential traffic violations, display results visually, and persist data for later review. Compared with a pure algorithm demo, this system aims to deliver a more complete engineering workflow.

### 3.2 Functional Requirements

The system should meet the following functional requirements:

1. It should support batch image analysis and video analysis.
2. It should support automatic lane segmentation and manual region override.
3. It should support vehicle target detection and 3D box estimation.
4. It should judge lane occupation and region-based violations.
5. It should display processing results visually in the GUI.
6. It should save structured result data into the database.
7. It should provide basic support for plate recognition and track-level association.

### 3.3 Non-Functional Requirements

In addition to functional completeness, the system should also satisfy certain non-functional goals. It should have acceptable runtime efficiency for image and moderate-frame-rate video processing. It should maintain stable execution under batch processing conditions. It should have a clear modular structure for future replacement of models. Finally, the result output should be understandable and operable for users, which requires good interface interaction and readable visualization.

### 3.4 Feasibility Analysis

The feasibility of this project is reflected in three aspects. In terms of technology, mature open-source frameworks such as YOLO, OpenCV, and Qt provide stable tool support. In terms of engineering, the modular architecture of the current codebase already separates detection, segmentation, rule reasoning, rendering, and persistence. In terms of implementation scope, the project chooses a realistic route of algorithm integration plus rule design rather than attempting to build all models from scratch, which is suitable for the scale of a graduation design.

## 4. Overall System Design

### 4.1 Overall Architecture

The system can be divided into four layers: data input layer, perception layer, decision layer, and application layer.

The data input layer accepts images and videos from the workspace. The perception layer includes lane segmentation, YOLO vehicle detection, 3D box estimation, plate recognition, and object tracking. The decision layer includes footprint overlap calculation, region rule reasoning, and traffic counting. The application layer includes result rendering, GUI interaction, and database persistence.

### 4.2 Main Module Division

According to the current project structure, the main modules are as follows:

- `src/core/`: perception and geometry modules, such as lane segmentation, 3D detection, tracking, and plate recognition
- `src/services/`: process orchestration, rule engine, rendering, scene profile, video pipeline, and database management
- `src/gui/`: user interface, worker threads, workspace panel, viewer panel, and rule configuration panel
- `src/tests/`: unit tests for core logic and service logic

This modular division reflects a clear engineering boundary and helps later maintenance and extension.

### 4.3 Processing Flow

For image processing, the system first reads the image, then obtains lane polygons by automatic detection or manual input. After that, the vehicle detector predicts the target bounding boxes and 3D geometry. The violation checker calculates the overlap between the vehicle footprint and lane mask. The renderer overlays the lane area, detection boxes, labels, and violation information on the original image. Finally, the database manager stores the result.

For video processing, the system extends the image pipeline by adding frame iteration, tracking, track-level OCR gating, region rule reasoning, count line statistics, and progress callbacks. This design makes the system suitable for dynamic traffic scenes instead of only static images.

### 4.4 Database Design

The database manager uses SQLite and defines tables such as `images`, `records`, `lane_segments`, `detections`, `violation_types`, and `reviews`. This design supports at least three important engineering goals. First, it separates original media from analysis records. Second, it stores lane geometry and detection geometry in structured form. Third, it reserves a review table for later human confirmation or rejection of suspicious events. Such a design is beneficial if the project is later extended toward a semi-automatic traffic review platform.

## 5. Key Algorithm Design

### 5.1 YOLO Vehicle Detection Method

The project uses YOLO as the front-end detector for road targets. In practice, the detector outputs bounding boxes, confidence scores, and category labels. These results are not only used for visualization, but also serve as the input for later 3D box estimation. In this design, 2D detection is the starting point of the entire perception chain. If the front-end detector fails, later spatial reasoning and violation judgment will also be affected.

To improve engineering usability, the detector is organized as a reusable class. This encapsulation allows the model path and confidence threshold to be maintained independently from the GUI and service layer. In the final thesis, real training settings and quantitative metrics such as Precision, Recall, and mAP should be supplemented here.

### 5.2 Lane Segmentation Method

The lane segmentation module loads a trained YOLO segmentation model and predicts the road special lane region. The code shows that the module collects polygon contours from the segmentation output and fills them into a binary mask. This dual representation is important. The mask supports pixel-level area calculation, while the polygon supports contour drawing, region editing, and geometric storage.

Compared with using manually drawn regions only, automatic lane segmentation improves scalability and reduces per-scene manual configuration cost. At the same time, the current pipeline still keeps the ability to accept manual lane overrides, which is a practical engineering compromise. It ensures that the system can still work when automatic segmentation is unstable or when the user wants to define special analysis regions explicitly.

### 5.3 Monocular 3D Bounding Box Estimation

The 3D module integrates Deep3DBox with YOLO detections. After obtaining 2D boxes, the system crops object patches, predicts orientation bins, dimension offsets, and confidence, and then computes the 3D location under projection constraints. The final output includes projected 2D corners, footprint polygons, object dimensions, and yaw information.

From the violation recognition perspective, the most valuable output is the ground-contact footprint approximation. When the system determines whether a vehicle occupies a lane or enters a restricted region, using the footprint is more reasonable than using a raw rectangular box, because the footprint better matches the actual road contact region of the vehicle.

### 5.4 Footprint-Lane Overlap Determination

The overlap-based violation checker is a direct and interpretable method. It rasterizes the vehicle footprint polygon and computes the intersection with the lane mask. The ratio between the overlap area and the vehicle footprint area is then used as the violation score. When the ratio exceeds the threshold, the system marks the vehicle as violating.

This method is suitable for emergency lane occupation and similar region intrusion scenarios. Its advantage lies in high interpretability and low implementation complexity. Its limitation is that it still depends on the quality of both lane segmentation and footprint estimation. Therefore, threshold tuning and error analysis should be discussed in the experiment chapter.

### 5.5 Region Rule Engine

The region rule engine is one of the most distinctive modules in the current project. It abstracts each scene region into polygon coordinates, optional direction lines, and rule bindings. Different rules then define their own parameters, such as minimum consecutive frames, minimum stop duration, speed threshold, and overlap ratio threshold.

For example, no parking can be triggered when the tracked object remains nearly stationary in a region for enough time. No non-motor vehicle entry can be triggered when a bicycle, motorcycle, or person remains inside a restricted region for enough frames. Wrong-way driving can be determined by comparing the motion direction of the track with the allowed direction line and verifying that the target sufficiently overlaps the designated region. This design makes the system explainable, configurable, and extensible.

### 5.6 Multi-Module Fusion Strategy

The project does not simply stack multiple algorithms together. Instead, it forms a multi-stage fusion pipeline:

1. YOLO provides category-level target perception.
2. Lane segmentation provides road structure information.
3. Deep3DBox provides geometric and spatial cues.
4. Tracking provides temporal continuity.
5. Rule reasoning converts visual evidence into explicit violation events.

This layered fusion strategy is more suitable for traffic violation analysis than a single black-box classifier because it allows each stage to be independently optimized and explained.

## 6. System Implementation

### 6.1 Development Environment

The project is implemented in Python. The main libraries include OpenCV for image processing, Ultralytics for YOLO inference, TensorFlow compatibility modules for Deep3DBox loading, NumPy for geometric computation, Qt for GUI construction, and SQLite for lightweight structured storage. According to the code structure, the system supports both script entry execution and GUI-based interactive analysis.

### 6.2 Image Processing Pipeline

In the image pipeline, `process_frame` in `src/services/pipeline.py` is the central orchestration function. It first determines whether manual lane polygons exist. If yes, it uses manual polygons to build the lane mask. Otherwise, it calls the lane detector for automatic lane extraction. Then the vehicle detector is invoked with lane polygon context. For each detected vehicle, the system obtains the footprint, 2D corners, category, confidence, and yaw information, performs overlap-based violation judgment, optionally runs plate recognition, and finally packs all results into a structured payload.

This implementation reflects a strong engineering style because each stage is explicit, debuggable, and replaceable. It also makes the thesis easier to write, since the data flow is clear from input to output.

### 6.3 Video Processing and Tracking Implementation

The video pipeline in `src/services/video_pipeline.py` extends the image pipeline with frame iteration, tracking, counting, event aggregation, and progress callbacks. Object tracking allows the system to retain track identities and compute motion trajectories. This is the basis for multi-frame violation reasoning. The video pipeline also includes track-level plate OCR gating, which avoids running plate recognition on every single frame and improves runtime efficiency.

In practical terms, the video pipeline is what turns the project from a static image demo into a more realistic traffic scene analysis system.

### 6.4 GUI Implementation

The GUI is implemented based on Qt, with `src/gui/main_window.py` as the main entry point. The interface includes a workspace panel, viewer panel, and rule configuration panel. It supports file import, preview display, result interaction, manual region drawing, direction setting, analysis start and stop, and progress feedback.

The GUI is important for the graduation project because it demonstrates the engineering completeness of the system. Users can not only run the algorithm, but also configure scene regions, inspect results, and interact with the analysis process. This improves both usability and presentation value.

### 6.5 Result Persistence Implementation

The database module stores analysis records in SQLite. The database manager first records the original media, then creates running records, and finally saves lane polygons, detections, violation categories, and review information after processing is completed. This mechanism makes the system results traceable and supports later extension toward query, filtering, and audit functions.

### 6.6 Testing Support

The current project includes test files for lane geometry, object tracking, region rule engine, traffic counting, video pipeline, and violation checking. This indicates that the project is not only a functional demo but also has a certain degree of testability and modular verification. In the thesis, this can be reflected as an advantage in software engineering quality.

## 7. Experiment and Test Analysis

### 7.1 Experimental Setup

The final version of the thesis should supplement the hardware configuration, software environment, model versions, and dataset composition in a table. Suggested items include CPU, GPU, memory, Python version, major dependency versions, image resolution, dataset size, and train-test split ratio.

### 7.2 Vehicle Detection Analysis

The vehicle detection experiment should report Precision, Recall, mAP@0.5, and optionally mAP@0.5:0.95. If the project used a custom dataset or transferred a pretrained model, this section should also describe the annotation categories, training epochs, confidence threshold, and representative detection examples. At present, the actual metric values should be replaced later with your own experiment results.

Suggested placeholder sentence:

"On the self-built traffic scene dataset, the YOLO-based detector achieved a Precision of [to be filled], a Recall of [to be filled], and an mAP@0.5 of [to be filled], indicating that the model can effectively identify the main vehicle categories in road scenes."

### 7.3 Lane Segmentation Analysis

The lane segmentation experiment should report indicators such as IoU, mIoU, and Pixel Accuracy, or at least provide a qualitative comparison through visual examples. The most important point in this section is not only whether the segmentation mask looks complete, but whether the contour can provide stable support for later overlap-based violation determination.

Suggested placeholder sentence:

"The lane segmentation module can stably extract the special lane region under most daytime road scenes, while failure cases are mainly concentrated in severe shadow interference, worn lane boundaries, and dense occlusion."

### 7.4 3D Box Estimation Analysis

Because the current 3D module is used mainly for spatial assistance and footprint estimation, this section can be written from two angles. One is quantitative evaluation if you have a benchmark. The other, which is more practical for a graduation design, is qualitative effect evaluation. You can compare whether the 3D projected box aligns with the vehicle body and whether the footprint is more reasonable than a plain 2D rectangle for region occupancy judgment.

Suggested placeholder sentence:

"Compared with pure 2D box-based overlap judgment, the introduction of monocular 3D box estimation improves the geometric rationality of vehicle-ground contact region approximation and enhances the interpretability of the violation determination process."

### 7.5 Violation Recognition Effect

The final experiment should evaluate the complete end-to-end effect of the system. Recommended indicators include violation recognition accuracy, recall, false alarm rate, and missed detection rate. You can also divide the analysis by violation type, such as emergency lane occupation, no parking, no non-motor vehicle entry, and wrong-way driving. If data volume is limited, a confusion-style summary table and several representative cases are already sufficient for a graduation thesis.

### 7.6 Functional Testing

Functional testing can be described around the following items:

1. Whether the system can import images and videos correctly
2. Whether lane segmentation runs normally
3. Whether vehicle targets and 3D boxes are rendered correctly
4. Whether region rules can be configured and applied
5. Whether processing results can be stored in the database
6. Whether the GUI can display progress and result details correctly

### 7.7 Limitations and Error Analysis

Although the current system has completed the major functional chain, there are still several limitations. First, the lane segmentation effect may degrade under poor illumination, severe occlusion, or complex road markings. Second, monocular 3D estimation is still sensitive to camera perspective assumptions and front-end detection quality. Third, rule-based reasoning depends on the correctness of region configuration and threshold settings. Fourth, plate recognition stability in small or blurred targets still needs improvement. These limitations should be described honestly in the thesis, because they also provide a natural basis for the future work section.

## 8. Conclusion and Future Work

This thesis designs and implements a vehicle violation recognition system oriented to traffic scenes. The system takes YOLO-based vehicle detection as the perception foundation, uses lane segmentation to provide structured road region information, introduces Deep3DBox-based monocular 3D estimation to enrich spatial understanding, and combines geometric overlap judgment with a region rule engine to complete violation reasoning. In addition, the system integrates object tracking, plate recognition, GUI interaction, and SQLite-based result persistence, forming a relatively complete engineering workflow.

Compared with schemes that only focus on one visual task, the system proposed in this project has stronger application orientation and modular explainability. It demonstrates that an algorithm plus engineering route is feasible for graduation design in intelligent transportation and computer vision.

Future work can be carried out in several directions. First, the lane segmentation model can be optimized using a larger and more diverse dataset. Second, the 3D module can be further improved with camera calibration or depth estimation assistance. Third, the rule engine can be expanded to more traffic scenarios such as red-light running, lane changing over solid lines, and illegal U-turns. Fourth, the database and GUI can be extended toward full case retrieval and review workflows. Finally, the project can be deployed toward a lightweight service or edge-computing form for more realistic application testing.

## References Placeholder

This draft intentionally leaves the formal reference list to a later step after you finish CNKI and related literature selection. When you are ready, this section can be replaced with a standard bibliography according to your school formatting rules.
