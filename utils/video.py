"""Video utilities for recording and playback"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from collections import deque
import threading
import queue


class VideoRecorder:
    """
    Thread-safe video recorder with circular buffer
    Supports continuous recording and alert-triggered clips
    """
    
    def __init__(self, output_dir: str = "recordings", 
                 fps: int = 30, 
                 codec: str = 'mp4v',
                 buffer_seconds: int = 10):
        """
        Args:
            output_dir: Directory to save recordings
            fps: Frames per second for output video
            codec: Video codec (mp4v, avc1, etc.)
            buffer_seconds: Number of seconds to buffer before alert
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.fps = fps
        self.codec = cv2.VideoWriter_fourcc(*codec)
        self.buffer_seconds = buffer_seconds
        
        # Circular buffer for pre-alert frames
        self.buffer_size = fps * buffer_seconds
        self.frame_buffer = deque(maxlen=self.buffer_size)
        
        # Current recording state
        self.is_recording = False
        self.current_writer: Optional[cv2.VideoWriter] = None
        self.current_filename: Optional[Path] = None
        self.frames_written = 0
        self.record_until_frame = 0
        
        # Thread-safe queue for frames
        self.frame_queue = queue.Queue(maxsize=100)
        self.recording_thread = None
        self.stop_thread = threading.Event()
        
    def start_background_recording(self):
        """Start background thread for video writing"""
        if self.recording_thread is None or not self.recording_thread.is_alive():
            self.stop_thread.clear()
            self.recording_thread = threading.Thread(target=self._recording_worker)
            self.recording_thread.daemon = True
            self.recording_thread.start()
    
    def stop_background_recording(self):
        """Stop background recording thread"""
        self.stop_thread.set()
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
    
    def _recording_worker(self):
        """Background worker that writes frames to disk"""
        while not self.stop_thread.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                if self.is_recording and self.current_writer:
                    self.current_writer.write(frame)
                    self.frames_written += 1
                    
                    # Check if we should stop recording
                    if self.frames_written >= self.record_until_frame:
                        self._stop_recording()
                
            except queue.Empty:
                continue
    
    def add_frame(self, frame: np.ndarray):
        """
        Add frame to buffer and recording (if active)
        Call this for every frame from your camera
        """
        # Always add to circular buffer
        self.frame_buffer.append(frame.copy())
        
        # If recording, add to queue
        if self.is_recording:
            try:
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                # Drop frame if queue is full
                pass
    
    def start_alert_recording(self, duration_seconds: int = 10, 
                             alert_id: str = "alert") -> Optional[Path]:
        """
        Start recording an alert clip
        Includes buffered frames from before the alert
        
        Args:
            duration_seconds: How long to record after alert
            alert_id: Identifier for the alert (included in filename)
            
        Returns:
            Path to output file or None if failed
        """
        if self.is_recording:
            # Already recording, extend duration
            additional_frames = duration_seconds * self.fps
            self.record_until_frame += additional_frames
            return self.current_filename
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"alert_{alert_id}_{timestamp}.mp4"
        
        # Get frame dimensions from buffer
        if not self.frame_buffer:
            return None
        
        h, w = self.frame_buffer[0].shape[:2]
        
        # Create video writer
        self.current_writer = cv2.VideoWriter(
            str(filename),
            self.codec,
            self.fps,
            (w, h)
        )
        
        if not self.current_writer.isOpened():
            return None
        
        self.current_filename = filename
        self.is_recording = True
        self.frames_written = 0
        
        # Calculate total frames to record
        post_alert_frames = duration_seconds * self.fps
        self.record_until_frame = len(self.frame_buffer) + post_alert_frames
        
        # Write buffered frames (pre-alert)
        for frame in self.frame_buffer:
            self.current_writer.write(frame)
            self.frames_written += 1
        
        return filename
    
    def _stop_recording(self):
        """Stop current recording"""
        if self.current_writer:
            self.current_writer.release()
            self.current_writer = None
        
        if self.current_filename:
            file_size = self.current_filename.stat().st_size / 1024  # KB
            print(f"✓ Saved recording: {self.current_filename.name} ({file_size:.1f} KB)")
        
        self.is_recording = False
        self.current_filename = None
        self.frames_written = 0
        self.record_until_frame = 0
    
    def save_snapshot(self, frame: np.ndarray, prefix: str = "snapshot") -> Path:
        """Save single frame as image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{prefix}_{timestamp}.jpg"
        
        cv2.imwrite(str(filename), frame)
        return filename
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_background_recording()
        
        if self.current_writer:
            self.current_writer.release()
        
        self.frame_buffer.clear()


class VideoPlayer:
    """
    Playback recorded video files with controls
    """
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0
        self.paused = False
    
    def play(self, window_name: str = "Video Playback"):
        """
        Play video with controls:
        - SPACE: Pause/Resume
        - LEFT/RIGHT: Skip backward/forward
        - Q: Quit
        """
        print(f"\nPlaying: {self.video_path.name}")
        print(f"Frames: {self.total_frames}, FPS: {self.fps:.1f}")
        print("\nControls:")
        print("  SPACE - Pause/Resume")
        print("  LEFT  - Skip backward 1 second")
        print("  RIGHT - Skip forward 1 second")
        print("  Q     - Quit\n")
        
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("End of video")
                    break
                
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                # Add progress bar
                self._draw_progress(frame)
                
                cv2.imshow(window_name, frame)
                
                # Calculate wait time based on FPS
                wait_time = int(1000 / self.fps)
            else:
                # When paused, just wait for key
                wait_time = 100
            
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"{'⏸ Paused' if self.paused else '▶ Playing'}")
            elif key == 81:  # Left arrow
                self._skip(-int(self.fps))  # Skip back 1 second
            elif key == 83:  # Right arrow
                self._skip(int(self.fps))   # Skip forward 1 second
        
        self.cleanup()
        cv2.destroyWindow(window_name)
    
    def _skip(self, frames: int):
        """Skip forward/backward by number of frames"""
        new_frame = self.current_frame + frames
        new_frame = max(0, min(new_frame, self.total_frames - 1))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.current_frame = new_frame
        print(f"Skipped to frame {new_frame}/{self.total_frames}")
    
    def _draw_progress(self, frame: np.ndarray):
        """Draw progress bar on frame"""
        h, w = frame.shape[:2]
        
        # Progress bar
        bar_height = 30
        bar_y = h - bar_height
        
        # Background
        cv2.rectangle(frame, (0, bar_y), (w, h), (0, 0, 0), -1)
        
        # Progress
        progress = self.current_frame / self.total_frames
        progress_w = int(w * progress)
        cv2.rectangle(frame, (0, bar_y), (progress_w, h), (0, 255, 0), -1)
        
        # Text
        time_current = self.current_frame / self.fps
        time_total = self.total_frames / self.fps
        
        text = f"{time_current:.1f}s / {time_total:.1f}s"
        if self.paused:
            text = "⏸ " + text
        
        cv2.putText(frame, text, (10, h - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()


class FrameBuffer:
    """
    Simple ring buffer for frames
    Useful for implementing replay features
    """
    
    def __init__(self, max_size: int = 300):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, frame: np.ndarray):
        """Add frame to buffer"""
        self.buffer.append(frame.copy())
    
    def get_last_n(self, n: int) -> list:
        """Get last N frames"""
        n = min(n, len(self.buffer))
        return list(self.buffer)[-n:]
    
    def get_all(self) -> list:
        """Get all frames in buffer"""
        return list(self.buffer)
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


# Example usage functions
def example_recorder():
    """Example: Record alerts with pre-buffer"""
    recorder = VideoRecorder(
        output_dir="recordings",
        fps=30,
        buffer_seconds=5  # Keep 5 seconds before alert
    )
    
    # Start background thread
    recorder.start_background_recording()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add every frame to recorder
            recorder.add_frame(frame)
            
            # Simulate alert trigger
            frame_count += 1
            if frame_count == 150:  # Alert at 5 seconds
                print("🚨 Alert triggered!")
                filename = recorder.start_alert_recording(
                    duration_seconds=10,
                    alert_id="test"
                )
                print(f"Recording to: {filename}")
            
            cv2.imshow('Camera', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recorder.cleanup()


def example_player():
    """Example: Play recorded video"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video.py <video_file>")
        return
    
    player = VideoPlayer(sys.argv[1])
    player.play()


if __name__ == "__main__":
    # Test video player if video file provided
    import sys
    if len(sys.argv) > 1:
        example_player()
    else:
        print("Video utilities module")
        print("\nAvailable classes:")
        print("  - VideoRecorder: Record video with circular buffer")
        print("  - VideoPlayer: Playback with controls")
        print("  - FrameBuffer: Simple frame ring buffer")