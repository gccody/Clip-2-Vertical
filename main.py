import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue
from threading import Thread
from moviepy import VideoFileClip
from tqdm import tqdm
import multiprocessing

class VideoProcessor:
    def __init__(self, resize_percent=0.7, blur_size=99):
        self.resize_percent = resize_percent
        self.blur_size = blur_size
        self.num_workers = multiprocessing.cpu_count()

    def process_frame(self, frame, target_width, target_height):
        # Calculate dimensions once
        orig_height, orig_width = frame.shape[:2]
        resized_width = int(orig_width * self.resize_percent)
        resized_height = int(orig_height * self.resize_percent)
        aspect_ratio = orig_width / orig_height
        target_aspect = target_width / target_height
        
        # Calculate background dimensions
        if aspect_ratio > target_aspect:
            bg_height = target_height
            bg_width = int(bg_height * aspect_ratio)
        else:
            bg_width = target_width
            bg_height = int(bg_width / aspect_ratio)
        
        # Process background (blur)
        bg_frame = cv2.resize(frame, (bg_width, bg_height), interpolation=cv2.INTER_LINEAR)
        bg_frame = cv2.GaussianBlur(bg_frame, (self.blur_size, self.blur_size), 0)
        
        # Create canvas and calculate offsets
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_offset_bg = (target_height - bg_height) // 2
        x_offset_bg = (target_width - bg_width) // 2
        
        # Place background
        y_start_bg = max(0, y_offset_bg)
        y_end_bg = min(target_height, y_offset_bg + bg_height)
        x_start_bg = max(0, x_offset_bg)
        x_end_bg = min(target_width, x_offset_bg + bg_width)
        bg_y_start = max(0, -y_offset_bg)
        bg_y_end = bg_y_start + (y_end_bg - y_start_bg)
        bg_x_start = max(0, -x_offset_bg)
        bg_x_end = bg_x_start + (x_end_bg - x_start_bg)
        
        canvas[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = bg_frame[bg_y_start:bg_y_end, bg_x_start:bg_x_end]
        
        # Process foreground
        resized_frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        
        # Place foreground
        x_offset = (target_width - resized_width) // 2
        y_offset = (target_height - resized_height) // 2
        y_start = max(0, y_offset)
        y_end = min(target_height, y_offset + resized_height)
        x_start = max(0, x_offset)
        x_end = min(target_width, x_offset + resized_width)
        frame_y_start = max(0, -y_offset)
        frame_y_end = frame_y_start + (y_end - y_start)
        frame_x_start = max(0, -x_offset)
        frame_x_end = frame_x_start + (x_end - x_start)
        
        canvas[y_start:y_end, x_start:x_end] = resized_frame[frame_y_start:frame_y_end, frame_x_start:frame_x_end]
        
        return canvas

    def _frame_reader(self, video, frame_queue, total_frames):
        count = 0
        while count < total_frames:
            ret, frame = video.read()
            if not ret:
                break
            frame_queue.put((count, frame))
            count += 1
        frame_queue.put(None)

    def _frame_writer(self, out_video, result_queue, total_frames):
        count = 0
        with tqdm(total=total_frames, unit='frame') as pbar:
            while count < total_frames:
                result = result_queue.get()
                if result is None:
                    break
                _, frame = result
                out_video.write(frame)
                count += 1
                pbar.update(1)

    def process_video(self, input_path: Path) -> Path:
        temp_output = input_path.parent / f"{input_path.stem}_temp.mp4"
        
        # Open video
        video = cv2.VideoCapture(str(input_path))
        
        # Video properties
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        orig_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output parameters
        target_width = orig_height
        target_height = orig_width
        
        # Initialize queues
        frame_queue = queue.Queue(maxsize=self.num_workers*2)
        result_queue = queue.Queue(maxsize=self.num_workers*2)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(temp_output), fourcc, fps, (target_width, target_height), isColor=True)
        
        # Start reader and writer threads
        reader_thread = Thread(target=self._frame_reader, args=(video, frame_queue, length))
        writer_thread = Thread(target=self._frame_writer, args=(out, result_queue, length))
        reader_thread.start()
        writer_thread.start()
        
        # Process frames using thread pool
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            while True:
                item = frame_queue.get()
                if item is None:
                    break
                    
                frame_idx, frame = item
                future = executor.submit(
                    self.process_frame,
                    frame,
                    target_width,
                    target_height,
                )
                result_queue.put((frame_idx, future.result()))
        
        # Signal completion and cleanup
        result_queue.put(None)
        reader_thread.join()
        writer_thread.join()
        video.release()
        out.release()
        cv2.destroyAllWindows()
        
        return temp_output

    @staticmethod
    def merge_audio(input_path: Path, processed_path: Path, output_path: Path):
        original = VideoFileClip(str(input_path))
        if original.audio is None:
          print("No audio in original clip")
          return
        processed = VideoFileClip(str(processed_path))
        
        processed.audio = original.audio
        
        processed.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
        
        original.close()
        processed.close()

def process_single_file(input_path: Path, processor: VideoProcessor):
    """Process a single video file"""
    print(f"Processing {input_path}")
    output_path = input_path.parent / f"{input_path.stem}_vertical.mp4"
    temp_path = processor.process_video(input_path)
    processor.merge_audio(input_path, temp_path, output_path)
    temp_path.unlink()  # Remove temporary file
    print(f"Completed: {output_path}")

def process_directory(directory: Path, processor: VideoProcessor):
    """Process all video files in a directory"""
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
    for file_path in directory.iterdir():
        if file_path.suffix.lower() in video_extensions:
            process_single_file(file_path, processor)

def main():
    parser = argparse.ArgumentParser(description='Convert horizontal videos to vertical format with blur background')
    parser.add_argument('input', type=str, help='Input file or directory path')
    parser.add_argument('--resize', type=float, default=0.7, help='Resize percentage (default: 0.7)')
    parser.add_argument('--blur', type=int, default=99, help='Blur size (must be odd number, default: 99)')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    
    # Create processor instance with custom parameters
    processor = VideoProcessor(
        resize_percent=args.resize,
        blur_size=args.blur
    )
    
    if input_path.is_file():
        process_single_file(input_path, processor)
    elif input_path.is_dir():
        process_directory(input_path, processor)
    else:
        print(f"Error: {input_path} does not exist")

if __name__ == "__main__":
    main()