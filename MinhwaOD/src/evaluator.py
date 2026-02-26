from ultralytics import YOLO


class Yolo26Evaluator:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def evaluate(self, data_yaml: str | None = None) -> dict:
        metrics = self.model.val(data=data_yaml) if data_yaml else self.model.val()

        # Ultralytics metrics 객체에서 자주 쓰는 항목 추출
        out = {
            "map50_95": float(metrics.box.map),
            "map50": float(metrics.box.map50),
            "map75": float(metrics.box.map75),
        }

        # results_dict가 있으면 함께 추가(버전 호환성 고려)
        try:
            out["results_dict"] = metrics.results_dict
        except Exception:
            pass

        return out