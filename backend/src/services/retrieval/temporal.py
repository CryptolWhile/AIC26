import heapq
from typing import List, Dict, Tuple
from collections import defaultdict
import bisect

class Frame:
    def __init__(self, image_id: str, timestamp: float, score: int, list_idx: int, frame_idx: int):
        self.video_name = self.get_video_name(image_id)
        self.image_id = image_id
        self.timestamp = timestamp
        self.score = score

        self.list_idx = list_idx  # which list this frame belongs to
        self.frame_idx = frame_idx  # index in the original list

    @staticmethod
    def get_video_name(image_id: str) -> str:
        return image_id[:8]
    
    def __str__(self) -> str:
        return f"Frame(image_id={self.image_id}, timestamp={self.timestamp}, score={self.score}, list_idx={self.list_idx}, frame_idx={self.frame_idx})"
    
    def __repr__(self) -> str:
        return f"Frame(image_id={self.image_id}, timestamp={self.timestamp:.2f}, score={self.score})"


class TemporalSolver:
    def __init__(self, lists: List[List[Dict]], top_k: int = 100):
        self.lists = lists
        self.k = len(lists)
        self.top_k = top_k

        self.video_frames = defaultdict(lambda: [[] for _ in range(self.k)])

        self._preprocess()

    def _preprocess(self):
        # Convert input to Frame objects
        for list_idx, lst in enumerate(self.lists):
            frames = []
            for frame_idx, f in enumerate(lst):
                frame = Frame(
                    f['image_id'],
                    f['timestamp'],
                    f['score'],
                    list_idx,
                    frame_idx
                )
                frames.append(frame)
                # Group by video
                self.video_frames[frame.video_name][list_idx].append(frame)

        # Sort frames by timestamp within each list for each video
        for video_name in self.video_frames:
            for list_idx in range(self.k):
                self.video_frames[video_name][list_idx].sort(key=lambda f: f.timestamp)

    @staticmethod
    def find_idx_by_timestamp(arr: List[Frame], timestamp: float) -> int:
        left = 0
        right = len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid].timestamp < timestamp:
                left = mid + 1
            elif arr[mid].timestamp > timestamp:
                right = mid - 1
            else:
                return mid

        return -1

    def _expand_frame(
            self,
            arr: List[Frame],
            timestamp: float,
            distance: float = 5.0
    ) -> List[Dict]:
        """Expand a frame to include all frames within distance seconds in the same list and video"""

        main_frame_idx = self.find_idx_by_timestamp(arr, timestamp)
        if main_frame_idx == -1:
            return []

        main_frame = arr[main_frame_idx]
        main_timestamp = main_frame.timestamp
        main_frame_data = {
            "image_id": main_frame.image_id,
            "timestamp": main_frame.timestamp,
            "score": main_frame.score,
            "is_main_frame": True
        }

        expanded_frames = []

        # Expand to the left
        left = main_frame_idx - 1
        while left >= 0:
            left_frame = arr[left]
            if abs(left_frame.timestamp - main_timestamp) <= distance:
                expanded_frames.append({
                    "image_id": left_frame.image_id,
                    "timestamp": left_frame.timestamp,
                    "score": left_frame.score,
                    "is_main_frame": False
                })
                left -= 1
            else:
                break

        expanded_frames.append(main_frame_data)

        # Expand to the right
        right = main_frame_idx + 1
        while right < len(arr):
            right_frame = arr[right]
            if abs(right_frame.timestamp - main_timestamp) <= distance:
                expanded_frames.append({
                    "image_id": right_frame.image_id,
                    "timestamp": right_frame.timestamp,
                    "score": right_frame.score,
                    "is_main_frame": False
                })
                right += 1
            else:
                break

        return expanded_frames



    def solve(self) -> List[Tuple[List[int], int]]:
        if self.k == 0:
            return []

        # Store (total_scores, frame_indices) - use min heap to keep top k largest
        top_tuples = []
        min_score_in_top = float('-inf')

        # Process each video independently
        for video_name, frames_by_list in self.video_frames.items():
            if any(len(frames) == 0 for frames in frames_by_list):
                continue

            # Precompute timestamps for binary search
            timestamps_by_list = []
            scores_by_list = []
            indices_by_list = []

            for frames in frames_by_list:
                timestamps_by_list.append([f.timestamp for f in frames])
                scores_by_list.append([f.score for f in frames])
                indices_by_list.append([f.frame_idx for f in frames])
            # Calculate maximum scores from each position for better pruning
            max_scores_from = []
            for frames in frames_by_list:
                if frames:
                    max_from = [0] * len(frames)
                    max_from[-1] = frames[-1].score
                    for i in range(len(frames) - 2, -1, -1):
                        max_from[i] = max(max_from[i + 1], frames[i].score)
                    max_scores_from.append(max_from)
                else:
                    max_scores_from.append([])

            # DFS to find all valid tuples for this video
            def find_tuples(list_idx: int, current_indices: List[int],
                            current_scores: int, last_timestamp: float):
                nonlocal min_score_in_top
                if list_idx == self.k:
                    tuple_key = (current_scores, tuple(current_indices))
                    if len(top_tuples) < self.top_k:
                        heapq.heappush(top_tuples, tuple_key)
                        if len(top_tuples) == self.top_k:
                            min_score_in_top = top_tuples[0][0]
                    elif current_scores > min_score_in_top:
                        heapq.heapreplace(top_tuples, tuple_key)
                        min_score_in_top = top_tuples[0][0]
                    return

                # Pruning
                if len(top_tuples) == self.top_k:
                    max_possible = current_scores
                    for i in range(list_idx, self.k):
                        if frames_by_list[i]:
                            idx = bisect.bisect_right(timestamps_by_list[i], last_timestamp)
                            if idx < len(frames_by_list[i]):
                                max_possible += max_scores_from[i][idx]
                            else:
                                return
                        else:
                            return

                    if max_possible <= min_score_in_top:
                        return

                # Binary search
                timestamps = timestamps_by_list[list_idx]
                start_idx = bisect.bisect_right(timestamps, last_timestamp)

                for idx in range(start_idx, len(frames_by_list[list_idx])):
                    frame_original_idx = indices_by_list[list_idx][idx]
                    frame_scores = scores_by_list[list_idx][idx]
                    frame_timestamp = timestamps[idx]

                    current_indices.append(frame_original_idx)
                    find_tuples(list_idx + 1, current_indices,
                                current_scores + frame_scores, frame_timestamp)
                    current_indices.pop()

            find_tuples(0, [], 0, float('-inf'))

        result = []
        while top_tuples:
            scores, indices = heapq.heappop(top_tuples)

            expanded_tuples = []
            for idx in range(len(indices)):
                main_frame = {
                    "image_id": self.lists[idx][indices[idx]]['image_id'],
                    "timestamp": self.lists[idx][indices[idx]]['timestamp'],
                    "score": self.lists[idx][indices[idx]]['score']
                }

                # Expand the frame
                video_name = Frame.get_video_name(main_frame['image_id'])
                expanded_frames = self._expand_frame(self.video_frames[video_name][idx], main_frame['timestamp'], 5)
                
                tuple_score = sum(frame['score'] for frame in expanded_frames)
                expanded_tuples.append({
                    "results": expanded_frames,
                    "score": tuple_score
                })

                    

            result.append(expanded_tuples)

        result.reverse()
        return result