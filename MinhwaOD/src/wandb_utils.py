from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, List

def _patch_wandb_ultralytics_callback() -> None:
    """
    wandb 0.25.0 + ultralytics 8.4.16 조합에서
    callback 내부가 특정 plot 심볼을 참조하는데 모듈에 없어서 NameError가 날 수 있음.
    => 모듈에 없으면 no-op을 주입해서 런타임 크래시를 방지.
    """
    import wandb.integration.ultralytics.callback as wandb_cb

    # RANK가 없을 때를 방어(환경/버전에 따라 누락되는 케이스 있음)
    if not hasattr(wandb_cb, "RANK"):
        setattr(wandb_cb, "RANK", -1)

    # 문제가 된 심볼: plot_detection_validation_results
    if not hasattr(wandb_cb, "plot_detection_validation_results"):
        def _noop(*args, **kwargs):
            return None
        setattr(wandb_cb, "plot_detection_validation_results", _noop)

    # 콜백 훅이 예외를 던져도 학습이 죽지 않게 래핑
    if hasattr(wandb_cb, "WandBUltralyticsCallback"):
        cls = wandb_cb.WandBUltralyticsCallback

        def _safe_wrap(method_name: str):
            if not hasattr(cls, method_name):
                return
            orig_attr = f"_orig_{method_name}"
            if hasattr(cls, orig_attr):
                return  # already wrapped
            setattr(cls, orig_attr, getattr(cls, method_name))
            orig = getattr(cls, orig_attr)

            def wrapped(self, *args, **kwargs):
                try:
                    return orig(self, *args, **kwargs)
                except Exception as e:
                    print(f"[WANDB WARNING] {method_name} failed (skip wandb plot/log): {e}")
                    return None

            setattr(cls, method_name, wrapped)

        # val/predict 쪽에서 plot이 자주 섞여서 터지므로 여기 위주로 방어
        for hook in ("on_val_end", "on_fit_epoch_end", "on_train_epoch_end", "on_predict_end", "on_model_save"):
            _safe_wrap(hook)


class WandBManager:
    """
    - train/eval/predict 모두에서 안전하게 wandb 로깅을 하기 위한 wrapper
    - callback 자동 로깅은 train에만 의존하고,
      eval/predict는 수동 log로 '확실히' 남긴다.
    """

    def __init__(
        self,
        enable: bool,
        project: str,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        job_type: str = "pipeline",
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.enable = enable
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.job_type = job_type
        self.tags = tags or []
        self._inited = False

        if self.enable:
            import wandb
            if wandb.run is None:
                wandb.init(
                    project=self.project,
                    entity=self.entity,
                    name=self.run_name,
                    job_type=self.job_type,
                    tags=self.tags,
                    config=config,
                )
            else:
                # 이미 run이 있으면 이어서 사용
                if config:
                    wandb.config.update(config, allow_val_change=True)
            self._inited = True

    def attach_ultralytics_callback(self, yolo_model) -> None:
        """
        train 단계: Ultralytics W&B callback을 붙여서 자동 로깅 최대 활용.
        단, plot 함수 누락 버그 방지를 위해 patch를 먼저 수행.
        """
        if not self.enable:
            return
        _patch_wandb_ultralytics_callback()
        from wandb.integration.ultralytics import add_wandb_callback
        add_wandb_callback(yolo_model)

    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> None:
        if not self.enable:
            return
        import wandb
        payload = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                payload[f"{prefix}{k}"] = v
        if payload:
            wandb.log(payload)

    def log_artifact_file(self, path: str, name: str, type_: str) -> None:
        if not self.enable:
            return
        import wandb
        art = wandb.Artifact(name=name, type=type_)
        art.add_file(path)
        wandb.log_artifact(art)

    def log_infer_image_with_boxes(self, plotted_rgb, caption: str = "prediction") -> None:
        """
        plotted_rgb: ultralytics result.plot()를 RGB로 변환한 numpy array (H,W,3)
        """
        if not self.enable:
            return
        import wandb
        wandb.log({"inference/overlay": wandb.Image(plotted_rgb, caption=caption)})

    def finish(self) -> None:
        if not self.enable:
            return
        import wandb
        wandb.finish()