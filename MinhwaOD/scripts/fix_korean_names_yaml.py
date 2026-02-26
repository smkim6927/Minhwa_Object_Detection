from pathlib import Path
import yaml


def fix_yaml_unicode(yaml_path: str):
    p = Path(yaml_path)

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    print("[Loaded names]")
    print(data.get("names", {}))

    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"[Saved] {p} (Korean preserved)")


if __name__ == "__main__":
    fix_yaml_unicode(r"D:\datasets\minhwa_yolo_detect\data.yaml")  # 수정