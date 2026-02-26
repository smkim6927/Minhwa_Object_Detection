import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from config import JsonExportConfig


class DetectionJsonExporter:
    def __init__(self, cfg: JsonExportConfig | None = None):
        self.cfg = cfg or JsonExportConfig()

    def _result_to_payload(
        self,
        result,
        image_path: str,
        model_path: str,
        conf_threshold: float,
    ) -> Dict[str, Any]:
        r = result.cpu()  # GPU tensor -> CPU tensor
        image_h, image_w = r.orig_shape

        detections: List[Dict[str, Any]] = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.tolist()
            confs = r.boxes.conf.tolist()
            clss = r.boxes.cls.int().tolist()

            xywh = r.boxes.xywh.tolist() if self.cfg.include_xywh else None
            xywhn = r.boxes.xywhn.tolist() if self.cfg.include_normalized_bbox else None
            xyxyn = r.boxes.xyxyn.tolist() if self.cfg.include_normalized_bbox else None

            for i, (box_xyxy, score, cls_id) in enumerate(zip(xyxy, confs, clss), start=1):
                cls_name = r.names[int(cls_id)]  # data.yaml에 한글이면 여기서도 한글

                det = {
                    "det_id": i,
                    "class_id": int(cls_id),
                    "class_name": cls_name,
                    "confidence": float(score),
                    "bbox_xyxy": [float(v) for v in box_xyxy],
                }

                if self.cfg.include_xywh and xywh is not None:
                    det["bbox_xywh"] = [float(v) for v in xywh[i - 1]]

                if self.cfg.include_normalized_bbox:
                    if xywhn is not None:
                        det["bbox_xywhn"] = [float(v) for v in xywhn[i - 1]]
                    if xyxyn is not None:
                        det["bbox_xyxyn"] = [float(v) for v in xyxyn[i - 1]]

                detections.append(det)

        payload: Dict[str, Any] = {
            "schema_version": self.cfg.schema_version,
            "task": "detect",
            "model": {
                "framework": "ultralytics",
                "model_path": str(model_path),
            },
            "image": {
                "path": str(image_path),
                "width": int(image_w),
                "height": int(image_h),
            },
            "inference": {
                "conf_threshold": float(conf_threshold),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "detections": detections,
        }

        if self.cfg.include_counts:
            payload["counts"] = {"total": len(detections)}

        return payload

    def export_single_result(
        self,
        result,
        image_path: str,
        model_path: str,
        conf_threshold: float,
        out_json_path: str,
    ) -> Dict[str, Any]:
        payload = self._result_to_payload(
            result=result,
            image_path=image_path,
            model_path=model_path,
            conf_threshold=conf_threshold,
        )

        out_path = Path(out_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 핵심: 한글 깨짐 방지
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return payload