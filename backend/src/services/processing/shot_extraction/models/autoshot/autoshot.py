import os
import torch
import numpy as np

from pathlib import Path
from typing import List, Dict, Any, Union

from src.services.processing.shot_extraction.models.autoshot.utils import get_frames, get_batches
from src.services.processing.shot_extraction.models.autoshot.modules.supernet_model import TransNetV2Supernet

def load_supernet_model(path, device):
    model = TransNetV2Supernet().eval()

    ckpt = torch.load(path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt['net'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if device == "cuda":
        model = model.cuda()
    return model

class Autoshot:
    
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "weights/ckpt_0_200_0.pth")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_supernet_model(model_dir, self.device)
    
    def predict(self, batch):
        batch = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]).float().to(self.device)
        one_hot_pred = self.model(batch)
        if isinstance(one_hot_pred, tuple):
            one_hot_pred = one_hot_pred[0]
        return torch.sigmoid(one_hot_pred[0])
    
    def predict_video(self, video_path: Union[str, Path]) -> List[Dict[str, Any]]:
        frames = get_frames(video_path)

        predictions = []
        for batch in get_batches(frames):
            one_hot_pred = self.predict(batch)
            one_hot_pred = one_hot_pred.detach().cpu().numpy()
            predictions.append(one_hot_pred[25:75])

        predictions = np.concatenate(predictions, axis=0)[:len(frames)]
        return predictions
    
    def predictions_to_scenes(self, predictions: np.ndarray, threshold: float = 0.293):
        scenes = []
        predictions = np.where(predictions > threshold, 1, 0)
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)