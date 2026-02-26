import os
from pathlib import Path
import pprint

from config import TrainConfig
from dataset import DatasetManager
from pipeline import MinHwaDetectPipeline


def main():
    dataset_root = "/content/drive/MyDrive/SemioticRAG/MinhwaOD/preprocessing/KIDP_yolo_dt"
    artifacts_dir = str(Path(dataset_root) / "artifacts")

    test_image = (
        "/content/drive/MyDrive/SemioticRAG/MinhwaOD/preprocessing/KIDP_yolo_dt/"
        "data/images/train/0000049157P.JPG.jpg"
    )
    out_json = str(Path(artifacts_dir) / "sample1_detect.json")

    train_ratio = 0.8
    split_seed = 42

    model_name = "yolo26x.pt"
    epochs = 100
    imgsz = 1080
    batch = 4
    device = "0"
    workers = 4

    dm = DatasetManager(dataset_root)

    # 1) yaml 로드/한글 names 보정
    data_yaml = dm.find_data_yaml(prefer_ko=True)
    dm.rewrite_names_korean_visible(data_yaml)

    print("\n[Before split sanity check]")
    pprint.pprint(dm.sanity_check(data_yaml), sort_dicts=False)

    # 2) split
    split_summary = dm.split_train_test_by_train_txt(
        yaml_path=data_yaml,
        train_ratio=train_ratio,
        seed=split_seed,
        out_yaml_name="data_ko_8020.yaml",
        out_train_txt="train80.txt",
        out_test_txt="test20.txt",
    )

    print("\n[Split summary]")
    pprint.pprint(split_summary, sort_dicts=False)

    # 3) label 존재 검증 후 txt 정리(기존 로직 유지)
    def validate_and_clean_txt(txt_path):
        if not txt_path or not os.path.exists(txt_path):
            return
        print(f"[Data Validation] Checking {os.path.basename(txt_path)}...")
        with open(txt_path, "r") as f:
            lines = f.readlines()

        valid_lines = []
        removed_count = 0
        for line in lines:
            img_p = line.strip()
            if not img_p:
                continue
            lbl_p = img_p.replace("/images/", "/labels/") if "/images/" in img_p else img_p
            base, _ = os.path.splitext(lbl_p)
            lbl_p = base + ".txt"
            if os.path.exists(lbl_p):
                valid_lines.append(line)
            else:
                removed_count += 1

        if removed_count > 0:
            print(f"[Data Clean] Removed {removed_count} lines from {os.path.basename(txt_path)} (missing labels).")
            with open(txt_path, "w") as f:
                f.writelines(valid_lines)
        else:
            print(f"[Data Clean] {os.path.basename(txt_path)} is clean.")

    validate_and_clean_txt(split_summary.get("created_train_txt"))
    validate_and_clean_txt(split_summary.get("created_test_txt"))

    data_ko_8020 = Path(split_summary["yaml_path"])

    # 4) TrainConfig (✅ imgsz=1080 + weighted)
    train_cfg = TrainConfig(
        model_name=model_name,
        data_yaml=str(data_ko_8020),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(Path(dataset_root) / "runs_minhwa"),
        name="yolo26_minhwa_detect_8020_img1080_wt",

        image_weights_enable=True,
        image_weights_power=1.0,
        image_weights_max_rep=3,
        image_weights_out_train_txt="train80_weighted.txt",
        image_weights_out_yaml="data_ko_8020_weighted.yaml",
        image_weights_rare_percentile=0.10,  # 하위 10% 클래스를 희소로 정의
        image_weights_rare_topn=20,

        # W&B
        wandb_enable=True,
        wandb_project="minhwa-od",
        wandb_entity=None,
        wandb_job_type="train",
        wandb_name="colab_train_img1080_weighted",
        wandb_finish=False,
        wandb_enable_ckpt=False,
    )

    # 5) ✅ weighted train txt 생성 + yaml train 교체 + "희소 클래스 샘플링 개선 리포트" 출력
    if train_cfg.image_weights_enable:
        weighted_train_txt = str(Path(dataset_root) / train_cfg.image_weights_out_train_txt)

        report = dm.build_weighted_train_txt(
            train_txt_path=split_summary["train_txt"],
            out_txt_path=weighted_train_txt,
            image_weights_power=train_cfg.image_weights_power,
            image_weights_max_rep=train_cfg.image_weights_max_rep,
            rare_percentile=train_cfg.image_weights_rare_percentile,
            rare_topn=train_cfg.image_weights_rare_topn,
            seed=train_cfg.seed,
        )

        print("\n[Weighted Sampling Report] (핵심 요약)")
        # 핵심만 보기 좋게 출력
        print(f"- original images: {report['original']}")
        print(f"- expanded samples: {report['expanded']}  (avg_rep_all={report['avg_rep_all']:.2f}, max_rep={report['max_rep']})")
        print(f"- rare percentile: bottom {int(report['rare_percentile']*100)}% classes  (rare_class_count={report['rare_class_count']})")
        print(f"- rare-image ratio (orig): {report['rare_images_original_ratio']*100:.2f}%")
        print(f"- rare-image ratio (weighted): {report['rare_images_weighted_ratio']*100:.2f}%")
        print(f"- rare ratio multiplier: {report['rare_ratio_multiplier']:.2f}x")
        print(f"- avg rep (rare images): {report['avg_rep_rare_images']:.2f}")
        print(f"- avg rep (non-rare images): {report['avg_rep_nonrare_images']:.2f}")

        print("\n[Top rare rep samples] (희소 클래스 포함 이미지 중 rep 높은 샘플)")
        for s in report["top_rare_rep_samples"]:
            print(f"  - rep={s['rep']}  img={s['img']}")

        print("\n[Rarest classes topN] (이미지 단위 빈도 하위 클래스)")
        for item in report["rare_classes_lowfreq_topn"]:
            print(f"  - class_id={item['class_id']}  img_freq={item['img_freq']}")

        weighted_yaml = str(Path(dataset_root) / train_cfg.image_weights_out_yaml)
        new_yaml = dm.patch_yaml_train_path_text(
            yaml_path=str(data_ko_8020),
            new_train_txt_path=weighted_train_txt,
            out_yaml_path=weighted_yaml,
        )
        print(f"\n[YAML patched] {new_yaml}")

        train_cfg.data_yaml = new_yaml

    # 6) pipeline 실행
    pipeline = MinHwaDetectPipeline(train_cfg=train_cfg, artifacts_dir=artifacts_dir)

    try:
        print("\n[Train] start")
        best_model_path = pipeline.train()
        print(f"[Train] done -> best/last: {best_model_path}")

        saved_model_path = pipeline.save_model_artifact(best_model_path, tag="best", copy=True)
        print(f"[Model saved] {saved_model_path}")

        print("\n[Eval] start")
        metrics = pipeline.evaluate(saved_model_path)
        pprint.pprint(metrics, sort_dicts=False)

        print("\n[Predict] start")
        payload = pipeline.predict_and_export_json(
            model_path=saved_model_path,
            image_path=test_image,
            out_json_path=out_json,
            conf=0.25,
            save_vis=True,
        )

        print(f"[JSON saved] {out_json}")
        print(f"[Detections] {payload.get('counts', {}).get('total', len(payload['detections']))}")

        if payload["detections"]:
            print("\n[Preview]")
            for d in payload["detections"][:10]:
                print(d)

    finally:
        pipeline.finish()


if __name__ == "__main__":
    main()
