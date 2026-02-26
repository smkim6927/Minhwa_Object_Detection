from __future__ import annotations

from pathlib import Path
import shutil
from datetime import datetime

from config import TrainConfig, PredictConfig, JsonExportConfig
from trainer import Yolo26Trainer
from evaluator import Yolo26Evaluator
from predictor import Yolo26Predictor
from exporter import DetectionJsonExporter

from typing import Dict, Any, Optional

from ultralytics import YOLO
from wandb_utils import WandBManager


class MinHwaDetectPipeline:
    def __init__(self, train_cfg: TrainConfig, artifacts_dir: str):
        self.train_cfg = train_cfg
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # W&B run은 pipeline 단에서 1회 init 권장
        self.wb: Optional[WandBManager] = None
        if self.train_cfg.wandb_enable:
            self.wb = WandBManager(
                enable=True,
                project=self.train_cfg.wandb_project,
                entity=self.train_cfg.wandb_entity,
                run_name=self.train_cfg.wandb_name,
                job_type="pipeline",
                tags=self.train_cfg.wandb_tags,
                config=self.train_cfg.to_wandb_config_dict(),
            )

    def train(self) -> str:
        trainer = Yolo26Trainer(self.train_cfg, wb=self.wb)
        trainer.train()
        best = trainer.get_best_weight_path()
        # 모델 아티팩트도 남기고 싶으면 여기서 log_artifact_file 호출 가능
        if self.wb is not None:
            self.wb.log_artifact_file(str(best), name="model-best", type_="model")
        return str(best)

    def save_model_artifact(self, model_path: str, tag: str = "best", copy: bool = True) -> str:
        src = Path(model_path)
        dst = self.artifacts_dir / f"{src.stem}_{tag}{src.suffix}"
        if copy:
            dst.write_bytes(src.read_bytes())
        else:
            dst = src
        # W&B artifact 업로드
        if self.wb is not None:
            self.wb.log_artifact_file(str(dst), name=f"model-{tag}", type_="model")
        return str(dst)

    def evaluate(self, model_path: str) -> Dict[str, Any]:
        model = YOLO(model_path)
        # ultralytics val은 metrics dict를 반환
        res = model.val(data=self.train_cfg.data_yaml)
        # res.results_dict가 있을 수도, res가 dict일 수도 있어 방어적으로 처리
        metrics = {}
        if hasattr(res, "results_dict") and isinstance(res.results_dict, dict):
            metrics = res.results_dict
        elif isinstance(res, dict):
            metrics = res
        # pipeline prefix로 수동 로깅(안 깨짐)
        if self.wb is not None:
            self.wb.log_metrics(metrics, prefix="eval/")
        return {
            "results_dict": metrics
        }

    def predict_and_export_json(
        self,
        model_path: str,
        image_path: str,
        out_json_path: str,
        conf: float = 0.25,
        save_vis: bool = True,
    ) -> Dict[str, Any]:
        import json
        import numpy as np

        model = YOLO(model_path)
        results = model.predict(source=image_path, conf=conf, save=save_vis)
        r0 = results[0]

        detections = []
        if r0.boxes is not None and len(r0.boxes) > 0:
            boxes_xyxy = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            cls_ids = r0.boxes.cls.cpu().numpy().astype(int)
            names = model.names

            for i in range(len(cls_ids)):
                cid = int(cls_ids[i])
                detections.append(
                    {
                        "det_id": i + 1,
                        "class_id": cid,
                        "class_name": names.get(cid, str(cid)),
                        "confidence": float(confs[i]),
                        "bbox_xyxy": [float(x) for x in boxes_xyxy[i].tolist()],
                    }
                )

        payload = {
            "image_path": image_path,
            "model_path": model_path,
            "conf": conf,
            "counts": {"total": len(detections)},
            "detections": detections,
        }

        outp = Path(out_json_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # ---- W&B infer 로깅(안정적) ----
        if self.wb is not None:
            # (1) overlay 이미지 로깅
            try:
                plotted = r0.plot()  # BGR uint8
                plotted_rgb = plotted[..., ::-1]  # BGR -> RGB
                self.wb.log_infer_image_with_boxes(plotted_rgb, caption=f"conf={conf}")
            except Exception as e:
                print(f"[WANDB WARNING] inference overlay log failed: {e}")

            # (2) JSON artifact 로깅
            self.wb.log_artifact_file(str(outp), name="inference-json", type_="prediction")

        return payload

    def finish(self) -> None:
        if self.wb is not None:
            self.wb.finish()