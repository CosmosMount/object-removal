#!/usr/bin/env python3
import argparse
import os
import sys

DAVIS_ROOT = os.path.join(os.path.dirname(__file__), "..", "DAVIS")
INPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "inputs")


def list_davis_sequences(davis_root):
    paths = []
    for subdir in ["JPEGImages/480p", "JPEGImages", ""]:
        seq_dir = os.path.join(davis_root, subdir)
        if os.path.isdir(seq_dir):
            paths = [d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))]
            if paths:
                break
    return sorted(paths)


def list_local_videos():
    videos = []
    for ext in ["*.mp4", "*.avi", "*.mkv", "*.mov"]:
        import glob
        videos.extend(glob.glob(os.path.join(INPUTS_DIR, ext)))
        videos.extend(glob.glob(os.path.join("videos", ext)))
    return sorted(set(os.path.abspath(v) for v in videos))


def select_video(videos):
    if not videos:
        print("No local videos found.")
        return None
    print("\nAvailable local videos:")
    for i, v in enumerate(videos, 1):
        print(f"  [{i}] {os.path.basename(v)}")
    print(f"  [0] Enter custom path")
    try:
        choice = int(input("\nSelect video number: "))
    except (ValueError, EOFError):
        return None
    if choice == 0:
        return input("Enter video path: ").strip()
    if 1 <= choice <= len(videos):
        return videos[choice - 1]
    return None


def select_davis_sequence(sequences):
    if not sequences:
        print("No DAVIS sequences found.")
        return None
    print("\nAvailable DAVIS sequences:")
    cols = 4
    for i in range(0, len(sequences), cols):
        row = sequences[i:i+cols]
        print("  " + "  ".join(f"[{i+j+1:2d}] {s:<15}" for j, s in enumerate(row)))
    print(f"  [0] Enter custom sequence name")
    try:
        choice = int(input("\nSelect sequence number: "))
    except (ValueError, EOFError):
        return None
    if choice == 0:
        return input("Enter sequence name: ").strip()
    if 1 <= choice <= len(sequences):
        return sequences[choice - 1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Select video or DAVIS sequence for baseline processing")
    parser.add_argument("--video", help="Directly specify video path")
    parser.add_argument("--davis_seq", help="Directly specify DAVIS sequence name")
    parser.add_argument("--davis_root", default=DAVIS_ROOT, help="DAVIS dataset root")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list_videos", action="store_true", help="List available local videos")
    group.add_argument("--list_davis", action="store_true", help="List available DAVIS sequences")
    args = parser.parse_args()

    if args.list_videos:
        videos = list_local_videos()
        if not videos:
            print("No local videos found.")
        else:
            print("\nAvailable local videos:")
            for v in videos:
                print(f"  {v}")
        return

    if args.list_davis:
        sequences = list_davis_sequences(args.davis_root)
        if not sequences:
            print("No DAVIS sequences found.")
        else:
            print(f"\nAvailable DAVIS sequences in {args.davis_root}:")
            cols = 4
            for i in range(0, len(sequences), cols):
                row = sequences[i:i+cols]
                print("  " + "  ".join(f"[{i+j+1:2d}] {s:<15}" for j, s in enumerate(row)))
        return

    video_path = None
    davis_seq = None

    if args.video:
        video_path = args.video
    elif args.davis_seq:
        davis_seq = args.davis_seq
    else:
        print("Select input source:")
        print("  [1] Local video file")
        print("  [2] DAVIS dataset sequence")
        try:
            choice = input("\nChoice [1/2]: ").strip()
        except (ValueError, EOFError):
            choice = ""
        
        if choice == "1":
            videos = list_local_videos()
            if videos:
                video_path = select_video(videos)
            else:
                video_path = input("Enter video path: ").strip()
        elif choice == "2":
            sequences = list_davis_sequences(args.davis_root)
            if sequences:
                davis_seq = select_davis_sequence(sequences)
            else:
                davis_seq = input("Enter DAVIS sequence name: ").strip()
        else:
            print("Invalid choice.")
            sys.exit(1)

    if video_path:
        if not os.path.exists(video_path):
            print(f"Error: Video not found: {video_path}")
            sys.exit(1)
        print(f"\nSelected video: {video_path}")
        print(f"Run with: python baseline/baseline.py --video {video_path}")
        print(f"Example output: python baseline/baseline.py --video {video_path} --output output/baseline_{os.path.splitext(os.path.basename(video_path))[0]}")
    elif davis_seq:
        seq_dir = os.path.join(args.davis_root, "JPEGImages/480p", davis_seq)
        if not os.path.isdir(seq_dir):
            seq_dir = os.path.join(args.davis_root, "JPEGImages", davis_seq)
        if not os.path.isdir(seq_dir):
            seq_dir = os.path.join(args.davis_root, davis_seq)
        if not os.path.isdir(seq_dir):
            print(f"Error: DAVIS sequence '{davis_seq}' not found.")
            sys.exit(1)
        print(f"\nSelected DAVIS sequence: {davis_seq}")
        print(f"  Frames dir: {seq_dir}")
        print(f"Run with: python baseline/baseline.py --video {seq_dir}")
        print(f"Example output: python baseline/baseline.py --video {seq_dir} --output output/baseline_{davis_seq}")
    else:
        print("No input specified.")
        sys.exit(1)


if __name__ == "__main__":
    main()
