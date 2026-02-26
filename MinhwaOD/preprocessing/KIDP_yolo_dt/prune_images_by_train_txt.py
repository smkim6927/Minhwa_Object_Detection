from pathlib import Path
from typing import List, Set

# 이미지 확장자 후보 (원본 파일명 내부 확장자 판별용)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def extract_original_name_from_line(line: str) -> str | None:
    """
    train.txt 한 줄 예시:
      data/images/train/0000046116P.JPG.jpg
    -> 추출 결과:
      0000046116P.JPG

    빈 줄/주석은 None 반환
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    name = Path(line).name  # "0000046116P.JPG.jpg"

    # 마지막 .jpg 래퍼 제거 (원본 이름에 확장자가 이미 있을 때만)
    # 예: "0000046116P.JPG.jpg" -> "0000046116P.JPG"
    lower_name = name.lower()
    if lower_name.endswith(".jpg"):
        stem = name[:-4]  # 마지막 ".jpg" 제거
        inner_suffix = Path(stem).suffix.lower()  # ".JPG" -> ".jpg"
        if inner_suffix in IMAGE_EXTS:
            return stem

    # 위 패턴이 아니면 그대로 사용
    return name


def build_keep_list_from_train_txt(train_txt_path: str) -> List[str]:
    keep_names: List[str] = []

    with open(train_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            original_name = extract_original_name_from_line(line)
            if original_name is not None:
                keep_names.append(original_name)

    return keep_names


def normalize_folder_filename_for_compare(filename: str) -> str:
    """
    폴더 안 파일명이 아래 둘 중 어느 형태든 비교 가능하게 정규화:
    - 원본형: 0000046116P.JPG        -> 그대로 유지
    - 래퍼형: 0000046116P.JPG.jpg    -> 0000046116P.JPG 로 변환
    """
    lower_name = filename.lower()
    if lower_name.endswith(".jpg"):
        stem = filename[:-4]
        inner_suffix = Path(stem).suffix.lower()
        if inner_suffix in IMAGE_EXTS:
            return stem
    return filename


def remove_images_not_in_keep_list(
    train_txt_path: str,
    target_folder: str,
    dry_run: bool = True,
    recursive: bool = False,
) -> None:
    """
    train.txt 기준으로 유지할 이미지 목록을 만들고,
    target_folder 내에서 목록에 없는 이미지들을 삭제한다.

    Args:
        train_txt_path: train.txt 경로
        target_folder: 정리할 이미지 폴더 경로
        dry_run: True면 삭제하지 않고 대상만 출력
        recursive: True면 하위 폴더까지 탐색
    """
    target_dir = Path(target_folder)
    if not target_dir.exists() or not target_dir.is_dir():
        raise FileNotFoundError(f"target_folder not found or not a directory: {target_folder}")

    keep_list = build_keep_list_from_train_txt(train_txt_path)
    keep_set: Set[str] = set(keep_list)

    print(f"[INFO] train.txt keep count (raw): {len(keep_list)}")
    print(f"[INFO] train.txt keep count (unique): {len(keep_set)}")

    # 파일 탐색
    iterator = target_dir.rglob("*") if recursive else target_dir.glob("*")

    deleted_count = 0
    kept_count = 0
    skipped_non_image = 0

    for p in iterator:
        if not p.is_file():
            continue

        # 폴더 내 파일이 이미지인지 확인 (일반 이미지 확장자 + 래퍼형 *.jpg)
        suffix = p.suffix.lower()
        if suffix not in IMAGE_EXTS:
            skipped_non_image += 1
            continue

        compare_name = normalize_folder_filename_for_compare(p.name)

        if compare_name in keep_set:
            kept_count += 1
            continue

        # keep_set에 없으면 삭제 대상
        if dry_run:
            print(f"[DRY-RUN DELETE] {p}")
        else:
            p.unlink()
            print(f"[DELETED] {p}")
        deleted_count += 1

    print("\n=== Summary ===")
    print(f"Kept files           : {kept_count}")
    print(f"Delete candidates    : {deleted_count}")
    print(f"Skipped non-image    : {skipped_non_image}")
    print(f"Mode                 : {'DRY-RUN' if dry_run else 'DELETE'}")


if __name__ == "__main__":
    # ===== 여기만 수정해서 사용 =====
    TRAIN_TXT = r"/home/jovyan/sr_db/MinhwaOD/preprocessing/KIDP_yolo_dt/train.txt"
    TARGET_FOLDER = r"/home/jovyan/sr_db/MinhwaOD/preprocessing/KIDP_yolo_dt/data/images/train"
    DRY_RUN = True       # 먼저 True로 확인 후, 실제 삭제 시 False
    RECURSIVE = False    # 하위 폴더까지 탐색하려면 True
    # ============================

    remove_images_not_in_keep_list(
        train_txt_path=TRAIN_TXT,
        target_folder=TARGET_FOLDER,
        dry_run=DRY_RUN,
        recursive=RECURSIVE,
    )