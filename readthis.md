目标：我需要快速知道每个脚本用途，并用正确命令跑不同 pipeline 和整体对比。

脚本用途：
- video.sh: 把 DAVIS 序列帧合成 mp4，输出到 outputs/origins。
- comparison.sh: 跑 baseline + 所有 pipelines（DAVIS 序列），并计算/收集指标。
- eval_comparison.sh: 只收集指标并生成对比 markdown（在 pipelines 跑完之后）。
- pipeline_vggt4d/vggt4d.sh: VGGT4D 动态掩码 + ProPainter 修复（视频或 DAVIS）。
- pipeline_vggt4dsam3/vggt4dsam3.sh: VGGT4D + SAM3 精修 + ProPainter（视频或 DAVIS）。
- pipeline_vggt4dsam3sd/vggt4dsam3sd.sh: VGGT4D + SAM3 + SD 关键帧 + ProPainter（视频或 DAVIS）。
- pipeline_vggt4dsam3_diffueraser/vggt4dsam3_diffueraser.sh: VGGT4D + SAM3 + DiffuEraser（视频或 DAVIS）。
- pipeline_yoloopt/yoloopt.sh: YOLO 首帧掩码 + 光流 + ProPainter（视频或 DAVIS）。
- pipeline_yolosam2/yolosam2.sh: YOLO 首帧掩码 + SAM2 VOS + ProPainter（视频或 DAVIS）。

常用命令（DAVIS 模式）：
（默认值来自脚本内的 *_ENV 变量，可在命令前覆盖）

pipeline_yoloopt：YOLO_ENV=sam2，PROPAINTER_ENV=propainter，DAVIS_ENV=davis
``` python
bash pipeline_yoloopt/yoloopt.sh --davis_seq bmx-trees --davis_input_root ./DAVIS --davis_gt_root ./DAVIS --davis_task unsupervised
```

pipeline_yolosam2：SAM2_ENV=sam2，YOLO_ENV=sam2，PROPAINTER_ENV=propainter，DAVIS_ENV=davis
``` python
bash pipeline_yolosam2/yolosam2.sh --davis_seq bmx-trees --davis_input_root ./DAVIS --davis_gt_root ./DAVIS --davis_task unsupervised
```

pipeline_vggt4d：VGGT_ENV=vggt，PREPROCESS_ENV=sam2，PROPAINTER_ENV=propainter，DAVIS_ENV=davis
``` python
bash pipeline_vggt4d/vggt4d.sh --davis_seq bmx-trees --davis_input_root ./DAVIS --davis_gt_root ./DAVIS --davis_task unsupervised --dyn_threshold_scale 1.0
```

pipeline_vggt4dsam3：SAM3_ENV=sam3，VGGT_ENV=vggt，PREPROCESS_ENV=sam3，PROPAINTER_ENV=propainter，DAVIS_ENV=davis
``` python
bash pipelines/vggt4dsam3/vggt4dsam3.sh --davis_seq bmx-trees --davis_input_root data/DAVIS --davis_gt_root data/DAVIS --davis_task unsupervised --dyn_threshold_scale 0.7
```

pipeline_vggt4dsam3sd：SAM3_ENV=sam3，VGGT_ENV=vggt，PREPROCESS_ENV=sam3，PROPAINTER_ENV=propainter，DAVIS_ENV=davis
``` python
bash pipelines/vggt4dsam3sd/vggt4dsam3sd.sh --davis_seq bmx-trees --davis_input_root data/DAVIS --davis_gt_root data/DAVIS --davis_task unsupervised --dyn_threshold_scale 0.7
```

pipeline_vggt4dsam3_diffueraser：SAM3_ENV=sam3，VGGT_ENV=vggt，PREPROCESS_ENV=sam3，DIFFUERASER_ENV=diffueraser，DAVIS_ENV=davis
``` python
bash pipelines/vggt4dsam3_diffueraser/vggt4dsam3_diffueraser.sh --davis_seq bmx-trees --davis_input_root data/DAVIS --davis_gt_root data/DAVIS --davis_task unsupervised --dyn_threshold_scale 0.7
```

常用命令（视频模式）：
（默认值来自脚本内的 *_ENV 变量，可在命令前覆盖）

pipeline_yoloopt：YOLO_ENV=sam2，PROPAINTER_ENV=propainter，DAVIS_ENV=davis
``` python
bash pipeline_yoloopt/yoloopt.sh --video /path/to/video.mp4
```
pipeline_yolosam2：SAM2_ENV=sam2，YOLO_ENV=sam2，PROPAINTER_ENV=propainter，DAVIS_ENV=davis
``` python
bash pipeline_yolosam2/yolosam2.sh --video /path/to/video.mp4
```
pipeline_vggt4d：VGGT_ENV=vggt，PREPROCESS_ENV=sam2，PROPAINTER_ENV=propainter，DAVIS_ENV=davis
``` python
bash pipeline_vggt4d/vggt4d.sh --video /path/to/video.mp4
```
pipeline_vggt4dsam3：SAM3_ENV=sam3，VGGT_ENV=vggt，PREPROCESS_ENV=sam3，PROPAINTER_ENV=propainter，DAVIS_ENV=davis
``` python
bash pipeline_vggt4dsam3/vggt4dsam3.sh --video /path/to/video.mp4 --dyn_threshold_scale 0.7
```
pipeline_vggt4dsam3sd：SAM3_ENV=sam3，VGGT_ENV=vggt，PREPROCESS_ENV=sam3，PROPAINTER_ENV=propainter，DAVIS_ENV=davis
``` python
bash pipeline_vggt4dsam3sd/vggt4dsam3sd.sh --video /path/to/video.mp4 --dyn_threshold_scale 0.7
```
pipeline_vggt4dsam3_diffueraser：SAM3_ENV=sam3，VGGT_ENV=vggt，PREPROCESS_ENV=sam3，DIFFUERASER_ENV=diffueraser，DAVIS_ENV=davis
``` python
bash pipeline_vggt4dsam3_diffueraser/vggt4dsam3_diffueraser.sh --video /path/to/video.mp4 --dyn_threshold_scale 0.7
```

一键对比（DAVIS）：
baseline（comparison.sh 内调用）：sam2；其余 pipeline 使用各自脚本里的默认 *_ENV
``` python
bash comparison.sh --davis_seq bmx-trees --davis_input_root ./DAVIS --davis_gt_root ./DAVIS --davis_task unsupervised
```

只做指标汇总（在 pipeline 跑完之后）：
eval_comparison.sh：PROPAINTER_ENV=propainter
``` python
bash eval_comparison.sh --davis_seq bmx-trees --davis_input_root ./DAVIS --davis_gt_root ./DAVIS
```