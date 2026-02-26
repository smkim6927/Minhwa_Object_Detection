from ultralytics import YOLO

from config import PredictConfig


class Yolo26Predictor:
    def __init__(self, cfg: PredictConfig):
        self.cfg = cfg
        self.model = YOLO(self.cfg.model_path)

    def predict(self, source: str):
        results = self.model.predict(
            source=source,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self.cfg.device,
            save=self.cfg.save_vis,
            project=self.cfg.project,
            name=self.cfg.name,
            verbose=self.cfg.verbose,
        )
        return results