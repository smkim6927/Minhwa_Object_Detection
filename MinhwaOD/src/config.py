from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Union, List, Any, Dict, Optional

DeviceType = Union[int, str]  # 예: 0, "0", "cpu"


@dataclass
class TrainConfig:
    # 모델/데이터
    model_name: str = "yolo26x.pt"
    data_yaml: str = ""

    # 학습 하이퍼파라미터
    epochs: int = 100
    imgsz: int = 1080  # ✅ 요청 반영
    batch: int = 4
    device: DeviceType = 0
    workers: int = 4

    # 러닝/저장
    project: str = "runs_minhwa"
    name: str = "yolo26x_minhwa_detect"
    pretrained: bool = True
    patience: int = 30
    cache: Union[bool, str] = False
    seed: int = 777
    deterministic: bool = True

    # -------------------------
    # Long-tail 대응: 데이터 레벨 image_weights (oversampling txt)
    # -------------------------
    image_weights_enable: bool = True
    image_weights_power: float = 1.0
    image_weights_max_rep: int = 3
    image_weights_out_train_txt: str = "train80_weighted.txt"
    image_weights_out_yaml: str = "data_ko_8020_weighted.yaml"

    # 희소 클래스 정의 기준: 빈도 하위 p% (예: 0.10이면 하위 10% 클래스)
    image_weights_rare_percentile: float = 0.10
    # 리포트에서 하위 빈도 클래스 몇 개 보여줄지
    image_weights_rare_topn: int = 20

    # (참고) focal loss gamma는 커스텀 loss 없이는 안정적으로 넣기 어려움
    focal_gamma: Optional[float] = None

    # W&B
    wandb_enable: bool = False
    wandb_project: str = "minhwa-od"
    wandb_entity: Union[str, None] = None
    wandb_job_type: str = "train"
    wandb_name: Union[str, None] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_finish: bool = False
    wandb_enable_ckpt: bool = False

    # augment
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    degrees: float = 5.0
    scale: float = 0.5
    fliplr: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

    def best_weight_path(self) -> Path:
        return Path(self.project) / self.name / "weights" / "best.pt"

    def to_wandb_config_dict(self) -> Dict[str, Any]:
        return asdict(self)
    


@dataclass
class PredictConfig:
    model_path: str
    conf: float = 0.25
    iou: float = 0.7
    device: DeviceType = 0
    save_vis: bool = False
    project: str = "runs_minhwa"
    name: str = "predict"
    verbose: bool = False


@dataclass
class JsonExportConfig:
    schema_version: str = "minhwa-detect-v1"
    include_xywh: bool = True
    include_normalized_bbox: bool = True
    include_counts: bool = True