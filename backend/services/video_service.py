import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchreid
from torchvision import transforms
from collections import defaultdict
from scipy.spatial import distance
import matplotlib.pyplot as plt
import json
import os
import base64

from backend.models.analyzer import TrackAnalyzer

class VideoProcessor:
    def __init__(self):
        # Initialize models
        self.yolo_model = YOLO("yolov8m.pt")
        self.reid_model = self._initialize_reid_model()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    def _initialize_reid_model(self):
        model = torchreid.models.build_model(name='osnet_x1_0', num_classes=751)
        torchreid.utils.load_pretrained_weights(model, 'osnet_x1_0_imagenet.pth')
        model.eval()
        return model
    
    def get_features(self, roi):
        if roi.size == 0: 
            return None
        roi = cv2.resize(roi, (128, 256), interpolation=cv2.INTER_LANCZOS4)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = torch.from_numpy(roi).float().permute(2, 0, 1) / 255.0
        roi = self.normalize(roi).unsqueeze(0)
        with torch.no_grad():
            return self.reid_model(roi).squeeze().cpu().numpy()
    def video_to_base64(self, video_path: str) -> str:
        """Convert video file to base64 encoded string"""
        with open(video_path, "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read()).decode('utf-8')
        return encoded_string
    
    def process_video(self, video_path: str, output_folder: str, 
                     desired_fps: int = 10, confidence_threshold: float = 0.3, 
                     iou_threshold: float = 0.4):
        """Process video and return analysis results"""
        os.makedirs(output_folder, exist_ok=True)

        # Initialize tracker and analyzer
        tracker = DeepSort(max_age=15, n_init=3, max_cosine_distance=0.4, nn_budget=50)
        analyzer = TrackAnalyzer()

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video metadata
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps

        if duration < 1:
            original_fps = 125
            duration = total_frames / original_fps

        frame_skip = max(1, int(round(original_fps / desired_fps)))
        target_frame_count = int(desired_fps * duration)

        # Set up output video
        output_video_path = os.path.join(output_folder, "output-behavior-analysis.mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                            desired_fps, (512, 384))

        frame_counter = 0
        processed_frames = 0

        interaction_colors = {
            "Handshake": (0, 255, 0),
            "Pushing": (0, 0, 255),
            "Wrestling": (255, 0, 0),
            "Close Proximity": (255, 255, 0),
            "Interaction": (255, 0, 255)
        }

        while cap.isOpened() and processed_frames < target_frame_count:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_counter % frame_skip != 0:
                frame_counter += 1
                continue

            # Process frame
            frame = cv2.resize(frame, (512, 384))

            # Person detection
            results = self.yolo_model(frame, classes=[0], 
                                    conf=confidence_threshold, 
                                    iou=iou_threshold)
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = frame[y1:y2, x1:x2]
                    features = self.get_features(roi)
                    if features is not None:
                        detections.append(([x1, y1, x2 - x1, y2 - y1], box.conf[0], features))

            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame)

            # Update analyzer
            analyzer.update_tracks(tracks, processed_frames)

            # Draw tracks and interactions
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                track_id = track.track_id

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                info_text = f"ID: {track_id} ({x2 - x1}x{y2 - y1})"
                cv2.putText(frame, info_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw interaction lines
            for id1, id2, interaction_type, confidence in analyzer.get_interactions():
                center1 = analyzer._box_center(analyzer.track_history[id1]['boxes'][-1])
                center2 = analyzer._box_center(analyzer.track_history[id2]['boxes'][-1])

                color = interaction_colors.get(interaction_type, (255, 255, 255))
                cv2.line(frame, center1, center2, color, 2)
                mid_point = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
                cv2.putText(frame, f"{interaction_type} {confidence * 100:.0f}%", mid_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            out.write(frame)
            processed_frames += 1
            frame_counter += 1

        cap.release()
        out.release()

        # Generate behavior report
        behavior_report = analyzer.get_behavior_analysis()
        report_path = os.path.join(output_folder, "behavior_report.json")
        with open(report_path, 'w') as f:
            json.dump(behavior_report, f, indent=2)

        # Generate interaction plots
        plot_folder = os.path.join(output_folder, 'interaction_plots')
        os.makedirs(plot_folder, exist_ok=True)
        analyzer.generate_interaction_plots(output_folder)

        return {
            "output_video_path": output_video_path,
            "report_path": report_path,
            "plots_path": plot_folder,
            "processed_frames": processed_frames
        }