from pathlib import Path
import yaml

def read_text_with_fallback(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # 마지막: 에러 무시로라도 읽기
    return path.read_text(encoding="utf-8", errors="replace")

def save_korean_visible_yaml(src_yaml: str, dst_yaml: str) -> None:
    src = Path(src_yaml)
    dst = Path(dst_yaml)
    dst.parent.mkdir(parents=True, exist_ok=True)

    raw = read_text_with_fallback(src)
    data = yaml.safe_load(raw)

    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] Saved Korean-visible YAML: {dst}")

if __name__ == "__main__":
    SRC = r"C:\Users\sumin\sumin\KIDP__yolo_dt\data.yaml"
    DST = r"C:\Users\sumin\sumin\KIDP__yolo_dt\data_ko.yaml"
    save_korean_visible_yaml(SRC, DST)