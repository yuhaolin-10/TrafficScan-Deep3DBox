# M2 计划：数据流转设计（以 SQLite Schema 为准）

M2 的目标：把当前 UI-only 的“Run”流程，逐步替换成真实后端处理，并把数据稳定地落到数据库里，保证后续 M3（渲染联动）和 M4（历史查询/复核）不会推翻数据模型。

本文只讲清楚一件事：**数据从哪里来，经过哪些状态，最终存到哪里**。

---

## 1. 数据对象（你给定的数据库设计）

### 1.1 表与含义

一套最小闭环的口径（先定死，避免实现时反复改）：
- images：原图实体（唯一），以 `content_hash` 去重
- records：一次处理 run（可多次），承载本次处理状态/产物
- lane_segments：某次 run 的车道结果（多边形/可选 mask）
- detections：某次 run 的车辆/违章实例（核心是 3D 底面 footprint，同时保留 2D 框）
- violation_types：违章类型字典（code → name）
- reviews：对某个 detection 的人工复核结果（通过/驳回 + 备注）

### 1.2 Schema（字段表）

#### images（原图实体，唯一）

| 字段 | 类型 | 约束/示例 | 含义 |
| --- | --- | --- | --- |
| id | INTEGER | 主键，自增 | image 的内部 ID |
| content_hash | TEXT | 非空；唯一；示例：`sha256(hex)` | 原图内容哈希（稳定去重键） |
| original_path | TEXT | 非空；示例：`D:\\...\\a.jpg` | 最近一次导入/定位到该图的路径（可变） |
| imported_at | TEXT | 非空；建议 ISO8601 | 首次入库时间（或最近导入时间） |

#### records（一次处理记录 / 一次 run）

| 字段 | 类型 | 约束/示例 | 含义 |
| --- | --- | --- | --- |
| id | INTEGER | 主键，自增 | record/run 的唯一 ID |
| image_id | INTEGER | 非空；外键→`images.id` | 本次 run 针对哪张原图 |
| status | TEXT | 非空；`running/done/failed` | 本次 run 的状态 |
| created_at | TEXT | 非空；建议 ISO8601 | 创建该 run 的时间（排队/启动入口） |
| processed_at | TEXT | 可空；建议 ISO8601 | 本次 run 完成时间（done/failed 时写） |
| processed_path | TEXT | 可空；示例：`...\\a_processed.jpg` | 本次 run 生成的结果图路径（done 时写） |
| error_message | TEXT | 可空 | 失败原因摘要（failed 时写） |

#### lane_segments（某次 run 的车道结果）

| 字段 | 类型 | 约束/示例 | 含义 |
| --- | --- | --- | --- |
| id | INTEGER | 主键，自增 | lane 结果的唯一 ID |
| record_id | INTEGER | 非空；外键→`records.id` | 该车道结果属于哪一次 run |
| lane_type_code | TEXT | 非空；示例：`"EMERGENCY_LANE"` | 车道类型码（为后续扩展预留） |
| polygons_json | TEXT | 非空；示例：`[[[x,y],...], ...]` | 车道多边形列表（矢量结果，可还原 mask） |
| mask_path | TEXT | 可空；示例：`...\\lane_mask.png` | 可选：保存的 mask 文件路径（回放/调试） |
| created_at | TEXT | 非空；建议 ISO8601 | 写入时间 |

#### violation_types（违章类型字典）

| 字段 | 类型 | 约束/示例 | 含义 |
| --- | --- | --- | --- |
| id | INTEGER | 主键，自增 | 违章类型 ID |
| code | TEXT | 唯一；示例：`"EMERGENCY_LANE"` | 稳定的机器可读类型码（用于程序/模型对齐） |
| name | TEXT | 示例：`"Emergency Lane Occupation"` | 用户可读的类型名（用于 UI 展示） |

#### detections（某次 run 的车辆/违章实例）

| 字段 | 类型 | 约束/示例 | 含义 |
| --- | --- | --- | --- |
| id | INTEGER | 主键，自增 | detection 的唯一 ID |
| record_id | INTEGER | 非空；外键→`records.id` | 该 detection 属于哪一次 run |
| vehicle_type | TEXT | 非空；示例：`"car"` | 车辆类别（来自检测模型） |
| confidence | REAL | 非空；建议 `0~1`；示例：`0.87` | 置信度（越大越可信） |
| bbox_x1 | REAL | 示例：`120.5` | 2D 框 xmin（像素，基于原图） |
| bbox_y1 | REAL | 示例：`88.0` | 2D 框 ymin（像素） |
| bbox_x2 | REAL | 示例：`360.5` | 2D 框 xmax（像素） |
| bbox_y2 | REAL | 示例：`248.0` | 2D 框 ymax（像素） |
| footprint_json | TEXT | 非空；示例：`[[x,y],[x,y],[x,y],[x,y]]` | 3D 框底面在图像上的 4 点多边形 |
| corners_json | TEXT | 可空；示例：`[[x,y],... x8]` | 3D 框 8 个角点的 2D 投影（用于回放绘制） |
| is_violating | INTEGER | 非空；`0/1` | 是否判定为违章 |
| violation_ratio | REAL | 非空；示例：`0.42` | 侵占比例（intersection / footprint_area） |
| violation_type_id | INTEGER | 可空；外键→`violation_types.id` | 违章类型（仅违章时填） |

#### reviews（人工复核记录）

| 字段 | 类型 | 约束/示例 | 含义 |
| --- | --- | --- | --- |
| id | INTEGER | 主键，自增 | review 的唯一 ID |
| detection_id | INTEGER | 外键；指向 `detections.id` | 复核针对哪条 detection |
| result | TEXT | `pass/reject` | 复核结论：通过/驳回 |
| reviewer | TEXT | 可空；示例：`"alice"` | 复核人 |
| reviewed_at | TEXT | 可空；建议 ISO8601 | 复核时间 |
| comment | TEXT | 可空 | 备注（例如驳回原因） |

---

## 2. M2 要解决的“数据流转”问题

M2 不追求做复杂 UI，而是要让下面三件事跑通：

1) UI 触发一次处理（Run Selected / Run All）
2) 后端处理输出结构化结果（detections 等）
3) 状态与结果能稳定落库，并能从库里读出来回显

---

## 3. 关键口径（先统一，后面才不会反复改）

### 3.1 status 的含义（以 records 为准）
- records.status 是“单次 run 级”的状态：
  - running：已经创建 run，正在处理中
  - done：处理成功，`processed_path` 可用（detections/lane_segments 也已落库）
  - failed：处理失败，`error_message` 有值（可无 detections）

### 3.2 processed_path 是什么？
- 建议定义为“可回放的结果图路径”（渲染叠加后的图片）。
- M2 可以先做到：跑完后端 → 生成一张结果图 → 写入 processed_path。

### 3.3 lane_segments 的口径
- lane_segments 保存模型输出的车道多边形（polygons），必要时可从 polygons 还原 mask。
- mask_path 属于可选缓存项：不影响闭环，只影响回放性能/调试便利性。

---

## 4. 数据流转主链路（从 UI 到数据库）

下面以“用户选中一张图片并点击 Run Selected”为例。

### 4.1 UI 触发
- 输入：选中的 original_path（图片路径）
- UI 立即做两件事：
  1) 把 UI 状态置为 RUNNING（避免用户觉得没响应）
  2) 启动后台任务（必须不阻塞 UI）

### 4.2 后台处理（Pipeline）
- 输入：original_path
- 输出：一个“结果包”（不要太复杂，至少包含）
  - content_hash（用于 images 去重）
  - width/height（可选）
  - processed_path（生成/保存的结果图路径）
  - lane_polygons（车道多边形列表，用于生成/复现 lane_mask）
  - detections 列表（每条至少包含：vehicle_type、confidence、bbox_2d、footprint_2d、corners_2d、is_violating、violation_ratio）
  - error（如果失败，给错误信息）

### 4.3 落库（写入顺序）
建议顺序（目的：中途失败也能定位到是哪张图的问题）：

1) 写入/更新 images（以 content_hash 去重）
   - 用 `content_hash` 查询 images；不存在则插入
   - 可选：更新 original_path 为最近一次导入路径
   - 成功后拿到 image_id
2) 创建 records（先写 running，保证失败也有落点）
   - 插入 records(image_id, status='running', created_at=now)
   - 成功后拿到 record_id
3) 写入 lane_segments（属于本次 run）
   - 插入 lane_segments(record_id, lane_type_code, polygons_json, mask_path?, created_at=now)
4) 写入 detections（属于本次 run）
   - 若有违章类型：先保证 violation_types 里存在对应 code（不存在就插入）
   - 再插入 detections(record_id, vehicle_type, confidence, bbox_*, footprint_json, corners_json?, is_violating, violation_ratio, violation_type_id?)
5) 更新 records（结束态）
   - 成功：status='done', processed_at=now, processed_path=...
   - 失败：status='failed', processed_at=now, error_message=...

### 4.4 UI 回显
- 成功：右侧结果面板（暂不改形态也可以）能显示 detections 相关字段
- 失败：日志显示 error；records.status 显示 failed（若 UI 暂未接 DB，可先用 UI 内存状态模拟）

---

## 5. 从数据库读回数据（为“历史/回放”做准备）

M2 的“读回”建议只做最小闭环：
- 给定 image_id / content_hash / original_path
  - 查 images 拿到 image_id
  - 查 records（取最新一条 status='done' 的记录）拿到 processed_path、processed_at、record_id
  - 查 lane_segments 拿到本次 run 的车道多边形（必要时还原 lane_mask）
  - 查 detections（按 record_id）+ violation_types 拿到本次 run 的结果列表
  - 查 reviews（可选）拿到人工复核结果（review → detection → record → image 可追溯）

这样你就能验证：写进去的数据能读出来，字段口径没有歧义。

---

## 6. 失败与异常（要写进数据流）

M2 至少要把这几类情况想清楚（UI/日志/DB 各自怎么表现）：
- 图片读不了（路径不存在/格式不支持）
- 模型加载失败（权重缺失/环境问题）
- 推理成功但无 detection（detections=空）
- 推理成功但 processed_path 生成失败（磁盘权限/目录不存在）

建议策略（简单版）：
- 无 detection：依然算“处理成功”，只是 detections 为空
- 处理失败：不写 detections；processed_path 也不更新；UI 标 FAILED 并打印 error

---

## 7. 为后续阶段预留（不在 M2 强做）

### 7.1 需要但可暂缓的字段
- images.captured_at：图片拍摄时间（EXIF/视频时间戳）
- images.processed_at：检测完成时间
- images.status：单图状态（pending/running/done/failed）

### 7.2 M3/M4 会用到什么
- M3（渲染联动）：detections 的 bbox + type + confidence 是核心输入
- M4（历史/复核）：images/records/lane_segments/detections/reviews 能完整支撑检索与审核

---

## 8. M2 验收标准（数据流转视角）

- 能按 content_hash 去重写入 images（同一张图多次导入不会重复插入）
- Run Selected/All 后：
  - 每次 run 都会创建 records（running→done/failed）
  - lane_segments 与 detections 能按 record_id 写入并能读回（包括空结果）
  - processed_path 能回放最新一次 done 的结果图
  - UI 能展示“处理成功/失败”的明确反馈（状态 + 日志）
