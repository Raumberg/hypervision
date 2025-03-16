import numpy as np
import cv2
import timeit
import time
import sys

import keyboard
from typing import Tuple, List, Dict


from libpy.engine import TensorRTEngine
from libpy.aim import MouseController
from libpy.screen import VisionSystem


class Hypervision:
    """Main application class handling detection loop and logic"""
    
    def __init__(self, config: Dict):
        """
        Initialize aimbot application
        
        Args:
            config: Configuration dictionary
        """
        self.cfg = config
        
        self.mouse = MouseController(
            config['screen_width'], 
            config['screen_height']
            )
        self.vision = VisionSystem(
            config['screen_width'], 
            config['screen_height'], 
            config['activation_range']
            )
        self.engine = TensorRTEngine(
            config['model_path']
            )
        
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.class_names = {0: "TARGET", 1: "HEAD"}

        self.last_delta = (0, 0)
        self.smoothing_factor = 0.25  # Start with 25% of calculated delta
        self.aim_threshold = 5  # Pixels threshold for perfect aim

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame for neural network input"""
        input_tensor = cv2.resize(frame, (640, 640))
        input_tensor = input_tensor.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(input_tensor, axis=0)

    def _process_detections(self, output: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[List, List, List]:
        # Pure algorithmic dreams
        output = output.reshape(6, 8400).transpose()
        frame_height, frame_width = frame_shape[:2]

        # Vectorized confidence and class calculations
        conf0 = output[:, 4]
        conf1 = output[:, 5]
        class_ids = np.where(conf0 > conf1, 0, 1)
        confidences = np.maximum(conf0, conf1)
        
        # Filter based on confidence threshold
        mask = confidences > self.cfg['confidence_threshold']
        filtered = output[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Catch edge case early
        if filtered.size == 0:
            return [], [], []

        # Vectorized coordinate scaling and box calculation
        scale_x = frame_width / 640
        scale_y = frame_height / 640
        x_center = filtered[:, 0] * scale_x
        y_center = filtered[:, 1] * scale_y
        width = filtered[:, 2] * scale_x
        height = filtered[:, 3] * scale_y

        x1 = (x_center - width / 2).astype(int)
        y1 = (y_center - height / 2).astype(int)
        x2 = (x_center + width / 2).astype(int)
        y2 = (y_center + height / 2).astype(int)
        boxes = np.column_stack([x1, y1, x2, y2])

        # Apply per-class NMS
        final_boxes = []
        final_confidences = []
        final_class_ids = []

        for class_id in [0, 1]:
            cls_mask = class_ids == class_id
            if not np.any(cls_mask):
                continue
            
            class_boxes = boxes[cls_mask].tolist()
            class_scores = confidences[cls_mask].tolist()
            
            nms_indices = cv2.dnn.NMSBoxes(
                class_boxes, class_scores,
                self.cfg['confidence_threshold'],
                self.cfg['nms_threshold']
            )
            
            if len(nms_indices) > 0:
                for idx in nms_indices.flatten():
                    final_boxes.append(class_boxes[idx])
                    final_confidences.append(class_scores[idx])
                    final_class_ids.append(class_id)

        return final_boxes, final_confidences, final_class_ids

    def _draw_detections(self, frame: np.ndarray, boxes: List, confidences: List, class_ids: List) -> np.ndarray:
        """Draw detection results on frame"""
        # Convert to RGB for display
        frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if boxes:
            # Separate head and body detections
            head_boxes = []
            body_boxes = []
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if cls_id == 1:  # HEAD
                    head_boxes.append((box, conf))
                else:  # TARGET
                    body_boxes.append((box, conf))

            # Draw all body boxes first
            for box, conf in body_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 1)
                label = f"TARGET {conf*100:.1f}%"
                cv2.putText(frame_display, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Process head detections
            if head_boxes:
                # Get best head detection
                head_boxes.sort(key=lambda x: x[1], reverse=True)
                (x1, y1, x2, y2), head_conf = head_boxes[0]
                
                # Draw head box
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Calculate center for head dot
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame_display, (center_x, center_y), 3, (0, 0, 255), -1)
                
                # Head text at bottom of box
                label = f"HEAD {head_conf*100:.1f}%"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_y = y2 + text_size[1] + 2
                cv2.putText(frame_display, label, (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # If no heads, draw best body as main target
            elif body_boxes:
                body_boxes.sort(key=lambda x: x[1], reverse=True)
                (x1, y1, x2, y2), body_conf = body_boxes[0]
                
                # Draw main target box
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Target center dot
                center_x = (x1 + x2) // 2
                center_y = y1 + (y2 - y1) // 5
                cv2.circle(frame_display, (center_x, center_y), 5, (0, 0, 255), -1)
                
                label = f"TARGET {body_conf*100:.1f}%"
                cv2.putText(frame_display, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw FPS counter
        fps_text = f"FPS: {self.fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(frame_display, fps_text, 
                    (frame.shape[1] - text_size[0] - 10, text_size[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame_display
    
    def _update_fps(self):
        """Update FPS counter using exponential moving average"""
        now = time.time()
        if not hasattr(self, '_last_fps_update'):
            self._last_fps_update = now
            self._frame_count = 0
            return
            
        self._frame_count += 1
        elapsed = now - self._last_fps_update
        
        if elapsed >= 1.0:
            self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_update = now

    def run(self):
        """Main detection loop"""
        try:
            while True:
                loop_start = timeit.default_timer()
                
                frame = self.vision.capture_frame()
                H, W = frame.shape[:2]
                input_tensor = self._preprocess_frame(frame)
                
                nn_start = timeit.default_timer()
                outputs = self.engine.infer(input_tensor)
                nn_time = (timeit.default_timer() - nn_start) * 1000
                
                final_boxes, final_confidences, final_class_ids = self._process_detections(
                    outputs[0]['host'].copy(), (H, W)
                )

                if self.cfg['enable_aim'] and final_boxes and keyboard.is_pressed(self.cfg['toggle_button']):
                    head_boxes = []
                    body_boxes = []
                    
                    for box, conf, cls_id in zip(final_boxes, final_confidences, final_class_ids):
                        if cls_id == 1:
                            head_boxes.append((box, conf))
                        else:
                            body_boxes.append((box, conf))
                    
                    # Only proceed if we have detections
                    if not (head_boxes or body_boxes):
                        continue
                    
                    # Prioritize heads
                    if head_boxes:
                        (x1, y1, x2, y2), _ = head_boxes[0]
                        head_center_y = y1 + int((y2 - y1) * 0.25)  # 25% from top
                        target_x = (x1 + x2) // 2
                        target_y = head_center_y
                    else:
                        # Fallback to bodies
                        (x1, y1, x2, y2), _ = body_boxes[0]
                        target_x = (x1 + x2) // 2
                        target_y = y1 + int((y2 - y1) * 0.33)

                    center_x = target_x + self.vision.region[0]
                    center_y = target_y + self.vision.region[1]
                    
                    match self.cfg['mouse']:
                        case 'rel':
                            self.mouse.set_rel_position(center_x, center_y, self.cfg["sensitivity"])
                        case 'abs':
                            self.mouse.set_abs_position(center_x, center_y, self.cfg["sensitivity"])
                        case _:
                            raise ValueError(f"Invalid mouse mode: {self.cfg['mouse']}")

                if self.cfg['display']:
                    display_frame = self._draw_detections(frame, final_boxes, final_confidences, final_class_ids)
                    cv2.imshow("HyperVision", display_frame)

                self._update_fps()
                total_time = (timeit.default_timer() - loop_start) * 1000
                
                sys.stdout.write(
                    f"\rFPS: {self.fps:.1f} | Inference: {nn_time:.1f}ms | Total: {total_time:.1f}ms"
                )
                sys.stdout.flush()

                if (self.cfg['display'] and cv2.waitKey(1) == ord('0')) or keyboard.is_pressed('0'):
                    break

        finally:
            cv2.destroyAllWindows()