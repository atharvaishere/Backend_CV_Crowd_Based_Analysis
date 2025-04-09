from collections import defaultdict
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import os

class TrackAnalyzer:
    def __init__(self, frame_width=512, frame_height=384):
        self.track_history = defaultdict(lambda: {
            'boxes': [], 'velocity': [], 'aspect_ratio': [],
            'interaction_partners': defaultdict(list), 'motion_pattern': []
        })
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.interaction_queue = []
        self.handshake_params = {
            'max_distance': 0.10 * frame_width,
            'min_duration': 5,
            'max_velocity': 2.0,
            'min_overlap': 0.15,
            'aspect_stability': 0.1
        }

        self.min_interaction_frames = 5
        self.handshake_max_dist = 0.10 * frame_width
        self.push_vel_threshold = 5.0
        self.wrestle_min_frames = 5
        self.wrestle_overlap_threshold = 0.3
        self.velocity_spike_threshold = 8.0

    def update_tracks(self, tracks, frame_time):
        """Update tracks with new detections"""
        current_tracks = {}
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            width, height = x2 - x1, y2 - y1
            aspect_ratio = width / (height + 1e-6)

            history = self.track_history[track_id]
            history['boxes'].append((x1, y1, x2, y2))
            history['aspect_ratio'].append(aspect_ratio)

            if len(history['boxes']) > 1:
                prev_center = self._box_center(history['boxes'][-2])
                curr_center = self._box_center(history['boxes'][-1])
                dx, dy = curr_center[0] - prev_center[0], curr_center[1] - prev_center[1]
                if len(history['velocity']) >= 5:
                    history['velocity'].pop(0)
                history['velocity'].append((dx, dy))

            current_tracks[track_id] = (x1, y1, x2, y2)

        self._detect_interactions(current_tracks, frame_time)

    def _detect_interactions(self, current_tracks, frame_time):
        """Detect interactions between tracks"""
        ids = list(current_tracks.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                box1, box2 = current_tracks[id1], current_tracks[id2]

                interaction = {
                    'id1': id1,
                    'id2': id2,
                    'time': frame_time,
                    'distance': distance.euclidean(self._box_center(box1), self._box_center(box2)),
                    'overlap': self._bbox_overlap(box1, box2),
                    'rel_velocity': self._relative_velocity(id1, id2),
                    'duration': len(self.track_history[id1]['interaction_partners'][id2]) + 1
                }

                interaction['velocity_spikes'] = self._detect_velocity_spikes(id1, id2)
                interaction_type, confidence = self._classify_interaction(interaction)
                interaction['type'] = interaction_type
                self.track_history[id1]['interaction_partners'][id2].append(interaction)
                self.interaction_queue.append((id1, id2, interaction_type, confidence))

    def _classify_interaction(self, interaction):
        """Classify the type of interaction"""
        if (interaction['duration'] >= self.wrestle_min_frames and
                interaction['overlap'] > self.wrestle_overlap_threshold and
                any(v > self.velocity_spike_threshold for v in interaction['velocity_spikes'])):
            return "Wrestling", 0.9
        if (interaction['duration'] >= self.handshake_params['min_duration'] and
                interaction['distance'] < self.handshake_params['max_distance'] and
                interaction['overlap'] > self.handshake_params['min_overlap'] and
                interaction['rel_velocity'] < self.handshake_params['max_velocity'] and
                self._check_aspect_stability(interaction)):
            return "Handshake", min(0.9, (1 - interaction['distance'] / self.handshake_params['max_distance']))
        if interaction['rel_velocity'] > self.push_vel_threshold:
            return "Pushing", min(0.9, interaction['rel_velocity'] / 15)
        return "Interaction", 0.6

    def _check_aspect_stability(self, interaction):
        """Check if aspect ratios are stable during interaction"""
        track1_aspects = self.track_history[interaction['id1']]['aspect_ratio'][-5:]
        track2_aspects = self.track_history[interaction['id2']]['aspect_ratio'][-5:]
        std1 = np.std(track1_aspects) if len(track1_aspects) > 1 else 0
        std2 = np.std(track2_aspects) if len(track2_aspects) > 1 else 0
        return std1 < self.handshake_params['aspect_stability'] and \
               std2 < self.handshake_params['aspect_stability']

    def _detect_velocity_spikes(self, id1, id2):
        """Detect sudden velocity changes"""
        v1 = [np.linalg.norm(v) for v in self.track_history[id1]['velocity'][-5:]]
        v2 = [np.linalg.norm(v) for v in self.track_history[id2]['velocity'][-5:]]
        spikes = [
            max(v1[-1] - np.mean(v1[:-1]), 0) if len(v1) > 1 else 0,
            max(v2[-1] - np.mean(v2[:-1]), 0) if len(v2) > 1 else 0
        ]
        return spikes

    def _bbox_overlap(self, box1, box2):
        """Calculate overlap between two bounding boxes"""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / (area1 + area2 - intersection)

    def _box_center(self, box):
        """Calculate center coordinates of a bounding box"""
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    def _relative_velocity(self, id1, id2):
        """Calculate relative velocity between two tracks"""
        v1 = np.array(self.track_history[id1]['velocity']) if self.track_history[id1]['velocity'] else np.zeros(2)
        v2 = np.array(self.track_history[id2]['velocity']) if self.track_history[id2]['velocity'] else np.zeros(2)
        mean_v1 = np.mean(v1, axis=0) if len(v1) > 0 else np.zeros(2)
        mean_v2 = np.mean(v2, axis=0) if len(v2) > 0 else np.zeros(2)
        return np.linalg.norm(mean_v1 - mean_v2)

    def get_interactions(self):
        """Return and clear the current interaction queue"""
        interactions = self.interaction_queue
        self.interaction_queue = []
        return interactions

    def get_behavior_analysis(self):
        """Generate behavior analysis report"""
        behavior_report = {}
        for track_id, history in self.track_history.items():
            if not history['boxes']:
                continue

            behavior_report[track_id] = {}
            velocities = history['velocity']
            speeds = [np.linalg.norm(np.array(v)) for v in velocities] if velocities else []
            avg_speed = np.mean(speeds) if speeds else 0.0
            direction_changes = self._calculate_direction_changes(velocities)

            if avg_speed < 1.0:
                movement_pattern = "Stationary"
            elif 1.0 <= avg_speed < 5.0:
                movement_pattern = "Walking" if direction_changes < 1.0 else "Active"
            else:
                movement_pattern = "Running"

            if any(any(d['type'] == "Wrestling" for d in partners)
                   for partners in history['interaction_partners'].values()):
                movement_pattern = "Combat"

            behavior_report[track_id].update({
                'movement_pattern': movement_pattern,
                'avg_speed': avg_speed,
                'agitation': direction_changes,
                'interactions': {}
            })

            for partner, details in history['interaction_partners'].items():
                if details:
                    behavior_report[track_id]['interactions'][partner] = {
                        'type': details[-1]['type'],
                        'duration': len(details),
                        'avg_distance': np.mean([d['distance'] for d in details])
                    }

        return behavior_report

    def _calculate_direction_changes(self, velocities):
        """Calculate directional changes in movement"""
        if len(velocities) < 2:
            return 0.0
        directions = [np.arctan2(v[1], v[0]) for v in velocities]
        changes = [abs(directions[i] - directions[i-1]) for i in range(1, len(directions))]
        return np.mean(changes) if changes else 0.0

    def generate_interaction_plots(self, output_folder):
        """Generate interaction plots"""
        plot_folder = os.path.join(output_folder, 'interaction_plots')
        os.makedirs(plot_folder, exist_ok=True)

        for track_id in self.track_history:
            for partner, interactions in self.track_history[track_id]['interaction_partners'].items():
                if not interactions:
                    continue

                times = [inter['time'] for inter in interactions]
                distances = [inter['distance'] for inter in interactions]

                plt.figure()
                plt.plot(times, distances, marker='o')
                plt.title(f"Distance over Time for IDs {track_id} and {partner}")
                plt.xlabel("Frame")
                plt.ylabel("Distance (pixels)")
                plt.grid(True)
                plt.savefig(os.path.join(plot_folder, f"interaction_{track_id}_{partner}.png"))
                plt.close()