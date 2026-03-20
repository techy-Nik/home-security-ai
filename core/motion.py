"""Intelligent motion detection with false positive filtering"""
import cv2
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class MotionResult:
    has_motion: bool
    is_valid: bool  # True if motion is localized, not global lighting change
    fg_mask: np.ndarray
    motion_boxes: List[Tuple[int, int, int, int]]
    motion_percentage: float


class SmartMotionDetector:
    """MOG2 with intelligent filtering"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=config.motion.var_threshold,
            detectShadows=True
        )
        
        # Set learning rate
        self.learning_rate = config.motion.learning_rate
        
        self.min_contour_area = config.motion.min_contour_area
        self.motion_threshold = config.motion.motion_percentage_threshold
        
        # Adaptive thresholding
        self.recent_motion_percentages = []
        self.max_history = 30
        
    def detect_motion(self, frame: np.ndarray) -> MotionResult:
        """
        Detect motion with intelligent filtering
        
        Returns MotionResult with validity check for lighting changes
        """
        h, w = frame.shape[:2]
        total_pixels = h * w
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        
        # Remove shadows (marked as 127)
        fg_mask[fg_mask == 127] = 0
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate motion percentage
        motion_pixels = cv2.countNonZero(fg_mask)
        motion_percentage = motion_pixels / total_pixels
        
        # Track recent motion percentages
        self.recent_motion_percentages.append(motion_percentage)
        if len(self.recent_motion_percentages) > self.max_history:
            self.recent_motion_percentages.pop(0)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter by area
        motion_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_boxes.append((x, y, x + w, y + h))
        
        has_motion = len(motion_boxes) > 0
        
        # Validate motion (filter out global lighting changes)
        is_valid = self._validate_motion(motion_percentage, motion_boxes, total_pixels)
        
        return MotionResult(
            has_motion=has_motion,
            is_valid=is_valid,
            fg_mask=fg_mask,
            motion_boxes=motion_boxes,
            motion_percentage=motion_percentage
        )
    
    def _validate_motion(self, motion_percentage: float, 
                        motion_boxes: List, total_pixels: int) -> bool:
        """
        Determine if motion is valid (localized) or invalid (lighting change)
        
        Returns False if:
        - Motion covers >40% of frame (likely lighting change)
        - Very few large boxes (clouds moving, shadows)
        """
        # Check 1: Global motion (lighting change)
        if motion_percentage > self.motion_threshold:
            self.logger.debug(
                f"Motion rejected: {motion_percentage:.1%} of frame "
                f"(threshold: {self.motion_threshold:.1%})"
            )
            return False
        
        # Check 2: Single massive box covering most of frame
        if len(motion_boxes) == 1:
            x1, y1, x2, y2 = motion_boxes[0]
            box_area = (x2 - x1) * (y2 - y1)
            if box_area / total_pixels > 0.6:
                self.logger.debug("Motion rejected: single box >60% of frame")
                return False
        
        # Check 3: Sudden spike in motion (likely environmental)
        if len(self.recent_motion_percentages) >= 3:
            recent_avg = np.mean(self.recent_motion_percentages[-3:-1])
            if motion_percentage > recent_avg * 3:
                self.logger.debug(
                    f"Motion rejected: spike {motion_percentage:.1%} vs "
                    f"avg {recent_avg:.1%}"
                )
                return False
        
        return True
    
    def reset(self):
        """Reset background model (call when camera moves or major scene change)"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=self.config.motion.var_threshold,
            detectShadows=True
        )
        self.recent_motion_percentages = []
        self.logger.info("Motion detector reset")