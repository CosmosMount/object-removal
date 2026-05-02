# Project Layout

This repo is organized by purpose. New locations are the canonical paths; root-level symlinks keep old commands working.

## Key Folders
- data/
  - DAVIS/
  - inputs/
  - videos/
- pipelines/
  - vggt4d/
  - vggt4dsam3/
  - vggt4dsam3sd/
  - vggt4dsam3_diffueraser/
  - yoloopt/
  - yolosam2/
- scripts/
  - comparison.sh
  - eval_comparison.sh
  - video.sh
- external/
  - ProPainter/
  - VGGT4D/
  - DiffuEraser/
  - sam2/
  - sam3/
  - davis2017-evaluation/
- outputs/
- baseline/

## Root Symlinks (Compatibility)
These names remain in the repo root as symlinks:
- DAVIS -> data/DAVIS
- inputs -> data/inputs
- videos -> data/videos
- pipeline_vggt4d -> pipelines/vggt4d
- pipeline_vggt4dsam3 -> pipelines/vggt4dsam3
- pipeline_vggt4dsam3sd -> pipelines/vggt4dsam3sd
- pipeline_vggt4dsam3_diffueraser -> pipelines/vggt4dsam3_diffueraser
- pipeline_yoloopt -> pipelines/yoloopt
- pipeline_yolosam2 -> pipelines/yolosam2
- ProPainter -> external/ProPainter
- VGGT4D -> external/VGGT4D
- DiffuEraser -> external/DiffuEraser
- sam2 -> external/sam2
- sam3 -> external/sam3
- davis2017-evaluation -> external/davis2017-evaluation
- comparison.sh -> scripts/comparison.sh
- eval_comparison.sh -> scripts/eval_comparison.sh
- video.sh -> scripts/video.sh

## Notes
- Use the pipelines/ and scripts/ paths in new docs.
- Existing commands using root-level paths still work via symlinks.
