#!/usr/bin/env python3
import argparse
import os
import shutil

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Preprocess a video into frames and an initial YOLO mask for SAM2.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--root_dir", required=True)
	parser.add_argument("--video", required=True, help="Path to input video file")
	parser.add_argument(
		"--classes",
		nargs="+",
		type=int,
		default=[0, 1, 2, 3, 5, 7],
		help="YOLO class IDs to detect for first-frame mask",
	)
	return parser.parse_args()


def ensure_empty_dir(path: str) -> None:
	if os.path.isdir(path):
		shutil.rmtree(path)
	os.makedirs(path, exist_ok=True)


def split_video_to_frames(video_path: str, frame_dir: str) -> tuple[np.ndarray, int]:
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video: {video_path}")

	first_frame = None
	frame_idx = 0
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		if first_frame is None:
			first_frame = frame.copy()
		out_name = f"{frame_idx:05d}.jpg"
		out_path = os.path.join(frame_dir, out_name)
		cv2.imwrite(out_path, frame)
		frame_idx += 1

	cap.release()

	if frame_idx == 0 or first_frame is None:
		raise RuntimeError(f"No frames decoded from video: {video_path}")
	return first_frame, frame_idx


def build_first_mask(frame: np.ndarray, classes: list[int], model_path: str) -> np.ndarray:
    from ultralytics import YOLO

    model = YOLO(model_path)
    h, w = frame.shape[:2]
    
    # 建议加上 conf=0.5 参数来过滤低置信度噪点
    result = model(frame, classes=classes, conf=0.5, verbose=False)[0]

    # 创建一个与原图尺寸相同、单通道的全黑背景图片 (数据类型必须是 uint8)
    img = np.zeros((h, w), dtype=np.uint8)

    # 提取所有目标的边界框，并将它们作为实心白色矩形画在黑底上
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            # 获取边界框坐标 (x_min, y_min, x_max, y_max)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # OpenCV 画图需要坐标为整数
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 在 img 上画实心矩形：颜色 255 (纯白), 线宽 -1 (表示填充实心)
            cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
            
    # 返回这张绘制好的图像
    return img


def main() -> None:
	args = parse_args()
	root_dir = os.path.abspath(args.root_dir)
	video_path = os.path.abspath(args.video)

	if not os.path.isfile(video_path):
		raise FileNotFoundError(f"Video file not found: {video_path}")

	video_name = os.path.splitext(os.path.basename(video_path))[0]
	inputs_dir = os.path.join(root_dir, "inputs")
	frame_dir = os.path.join(inputs_dir, video_name)
	mask_dir = os.path.join(inputs_dir, f"{video_name}_mask")

	ensure_empty_dir(frame_dir)
	ensure_empty_dir(mask_dir)

	first_frame, frame_count = split_video_to_frames(video_path, frame_dir)

	yolo_ckpt = os.path.join(root_dir, "baseline", "yolov8n-seg.pt")
	if not os.path.isfile(yolo_ckpt):
		yolo_ckpt = "yolov8n-seg.pt"

	mask = build_first_mask(first_frame, args.classes, yolo_ckpt)
	cv2.imwrite(os.path.join(mask_dir, "00000.png"), mask)

	print(f"video_name: {video_name}")
	print(f"frame_dir: {frame_dir}")
	print(f"mask_dir: {mask_dir}")
	print(f"frame_count: {frame_count}")


if __name__ == "__main__":
	main()
