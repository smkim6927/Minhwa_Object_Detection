from __future__ import annotations

from pathlib import Path
from typing import Optional

from ultralytics import YOLO
from config import TrainConfig
from wandb_utils import WandBManager


class Yolo26Trainer:
    def __init__(self, cfg: TrainConfig, wb: Optional[WandBManager] = None):
        self.cfg = cfg
        self.model = YOLO(self.cfg.model_name)
        self._save_dir: Optional[Path] = None
        self.wb = wb

        if self.wb is not None:
            self.wb.attach_ultralytics_callback(self.model)

    def train(self):
        results = self.model.train(
            data=self.cfg.data_yaml,
            epochs=self.cfg.epochs,
            imgsz=self.cfg.imgsz,  # ✅ 1080
            batch=self.cfg.batch,
            device=self.cfg.device,
            workers=self.cfg.workers,
            project=self.cfg.project,
            name=self.cfg.name,
            pretrained=self.cfg.pretrained,
            patience=self.cfg.patience,
            cache=self.cfg.cache,
            seed=self.cfg.seed,
            deterministic=self.cfg.deterministic,
            mosaic=self.cfg.mosaic,
            mixup=self.cfg.mixup,
            copy_paste=self.cfg.copy_paste,
            degrees=self.cfg.degrees,
            scale=self.cfg.scale,
            fliplr=self.cfg.fliplr,
            hsv_h=self.cfg.hsv_h,
            hsv_s=self.cfg.hsv_s,
            hsv_v=self.cfg.hsv_v,
        )

        self._save_dir = self._extract_save_dir(results) or self._guess_latest_save_dir()
        return results

    def get_best_weight_path(self) -> Path:
        if self._save_dir is not None:
            best = self._save_dir / "weights" / "best.pt"
            if best.exists():
                return best
            last = self._save_dir / "weights" / "last.pt"
            if last.exists():
                return last

        latest = self._guess_latest_save_dir()
        if latest is not None:
            best = latest / "weights" / "best.pt"
            if best.exists():
                return best
            last = latest / "weights" / "last.pt"
            if last.exists():
                return last

        raise FileNotFoundError(
            "No checkpoint found (best.pt/last.pt). "
            f"Checked save_dir={self._save_dir} and latest under project={self.cfg.project}."
        )

    def _extract_save_dir(self, results) -> Optional[Path]:
        for attr in ("save_dir", "dir"):
            try:
                v = getattr(results, attr, None)
                if v:
                    return Path(str(v))
            except Exception:
                pass
        return None

    def _guess_latest_save_dir(self) -> Optional[Path]:
        project_dir = Path(self.cfg.project)
        if not project_dir.exists():
            return None
        candidates = [p for p in project_dir.glob(f"{self.cfg.name}*") if p.is_dir()]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]