import os
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


class DatasetManager:
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)

    # -------------------------
    # 프로젝트 기존 구현이 있으면 그걸 유지하세요
    # -------------------------
    def find_data_yaml(self, prefer_ko: bool = True) -> str:
        candidates = []
        if prefer_ko:
            candidates += ["data_ko.yaml", "data_ko.yml"]
        candidates += ["data.yaml", "data.yml"]
        for c in candidates:
            p = self.dataset_root / c
            if p.exists():
                return str(p)
        raise FileNotFoundError(f"data yaml not found under {self.dataset_root}")

    def rewrite_names_korean_visible(self, yaml_path: str) -> None:
        # 기존 코드가 있다면 유지(여기서는 no-op)
        return

    def sanity_check(self, yaml_path: str) -> Dict[str, Any]:
        return {"yaml_path": yaml_path}

    def split_train_test_by_train_txt(
        self,
        yaml_path: str,
        train_ratio: float,
        seed: int,
        out_yaml_name: str,
        out_train_txt: str,
        out_test_txt: str,
    ) -> Dict[str, Any]:
        """
        데이터셋의 이미지들을 읽어와 train_ratio 비율에 맞게 분할하고, 
        결과 txt 파일들과 새로운 yaml 파일을 생성합니다.
        """
        rng = random.Random(seed)
        orig_yaml = Path(yaml_path)
        all_images = []

        # 1. YAML 파일에서 기존 train 리스트 파악 시도
        if orig_yaml.exists():
            lines = orig_yaml.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if line.strip().startswith("train:"):
                    val = line.split("train:")[1].strip()
                    train_file = self.dataset_root / val
                    if train_file.is_file() and train_file.suffix == '.txt':
                        img_lines = train_file.read_text(encoding="utf-8").splitlines()
                        all_images = [p.strip() for p in img_lines if p.strip()]
                    break
        
        # 2. txt 파일에서 찾지 못했다면 이미지 디렉토리에서 직접 수집
        if not all_images:
            images_dir = self.dataset_root / "images"
            if images_dir.exists():
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                    all_images.extend([str(p.resolve()) for p in images_dir.rglob(ext)])
            else:
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                    all_images.extend([str(p.resolve()) for p in self.dataset_root.rglob(ext)])

        # 중복 제거 및 정렬 후 셔플 (재현성 확보)
        all_images = sorted(list(set(all_images)))
        rng.shuffle(all_images)

        # 3. 데이터 분할
        total_len = len(all_images)
        train_len = int(total_len * train_ratio)
        
        train_list = all_images[:train_len]
        test_list = all_images[train_len:]

        # 4. 분할된 리스트를 txt 파일로 저장
        out_train_path = self.dataset_root / out_train_txt
        out_test_path = self.dataset_root / out_test_txt
        
        out_train_path.parent.mkdir(parents=True, exist_ok=True)
        out_test_path.parent.mkdir(parents=True, exist_ok=True)
        
        out_train_path.write_text("\n".join(train_list) + "\n", encoding="utf-8")
        out_test_path.write_text("\n".join(test_list) + "\n", encoding="utf-8")

        # 5. 새로운 YAML 파일 생성
        new_yaml = self.dataset_root / out_yaml_name
        if orig_yaml.exists():
            lines = orig_yaml.read_text(encoding="utf-8").splitlines()
            out_lines = []
            has_train = False
            has_val = False
            
            for line in lines:
                if line.strip().startswith("train:"):
                    out_lines.append(f"train: {out_train_txt}")
                    has_train = True
                elif line.strip().startswith("val:"):
                    out_lines.append(f"val: {out_test_txt}")
                    has_val = True
                elif line.strip().startswith("test:"):
                    # test 키가 있다면 함께 업데이트
                    out_lines.append(f"test: {out_test_txt}")
                else:
                    out_lines.append(line)
                    
            if not has_train:
                out_lines.append(f"train: {out_train_txt}")
            if not has_val:
                out_lines.append(f"val: {out_test_txt}")
                
            new_yaml.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        else:
            # 원본 YAML이 없는 경우 기본 포맷으로 생성
            yaml_content = f"path: .\ntrain: {out_train_txt}\nval: {out_test_txt}\n"
            new_yaml.write_text(yaml_content, encoding="utf-8")

        return {
            "yaml_path": str(new_yaml),
            "train_txt": str(out_train_path),
            "test_txt": str(out_test_path),
            "total_images": total_len,
            "train_ratio": train_ratio,
            "train_count": len(train_list),
            "test_count": len(test_list)
        }

    # =========================================================
    # Weighted oversampling + 희소 클래스 샘플링 개선 리포트
    # =========================================================
    def build_weighted_train_txt(
        self,
        train_txt_path: str,
        out_txt_path: str,
        image_weights_power: float = 1.0,
        image_weights_max_rep: int = 3,
        rare_percentile: float = 0.10,
        rare_topn: int = 20,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        - train txt를 oversampling해서 out_txt 생성
        - 희소 클래스 포함 이미지가 실제로 얼마나 더 자주 뽑히는지(%) 리포트 제공
        """
        rng = random.Random(seed)

        train_txt = Path(train_txt_path)
        out_txt = Path(out_txt_path)
        out_txt.parent.mkdir(parents=True, exist_ok=True)

        lines = [l.strip() for l in train_txt.read_text(encoding="utf-8").splitlines() if l.strip()]

        # 1) img -> label path, img -> set(classes)
        img_to_lbl: Dict[str, str] = {}
        img_classset: Dict[str, set] = {}

        for img_p in lines:
            lbl_p = img_p.replace("/images/", "/labels/") if "/images/" in img_p else img_p
            base, _ = os.path.splitext(lbl_p)
            lbl_p = base + ".txt"
            img_to_lbl[img_p] = lbl_p

        # 2) class freq (이미지 단위: 해당 클래스를 포함한 이미지 수)
        class_freq: Dict[int, int] = {}

        for img_p, lbl_p in img_to_lbl.items():
            lp = Path(lbl_p)
            if not lp.exists():
                img_classset[img_p] = set()
                continue

            cset = set()
            for row in lp.read_text(encoding="utf-8").splitlines():
                row = row.strip()
                if not row:
                    continue
                parts = row.split()
                try:
                    cid = int(float(parts[0]))
                    cset.add(cid)
                except Exception:
                    continue

            img_classset[img_p] = cset
            for cid in cset:
                class_freq[cid] = class_freq.get(cid, 0) + 1

        if not class_freq:
            out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return {
                "out_txt": str(out_txt),
                "original": len(lines),
                "expanded": len(lines),
                "note": "no labels found -> keep original list",
            }

        # 3) 희소 클래스 집합 정의 (빈도 하위 p%)
        #    빈도 기준: 이미지 수
        freq_items = sorted(class_freq.items(), key=lambda x: x[1])  # (cid, freq) 오름차순
        k = max(1, int(math.ceil(len(freq_items) * float(rare_percentile))))
        rare_classes = set([cid for cid, _ in freq_items[:k]])

        # 4) 이미지 weight 계산: sum((1/freq[c])^power)
        weights: List[float] = []
        for img_p in lines:
            cset = img_classset.get(img_p, set())
            if not cset:
                weights.append(0.0)
                continue
            w = 0.0
            for cid in cset:
                f = class_freq.get(cid, 1)
                w += (1.0 / float(f)) ** float(image_weights_power)
            weights.append(w)

        # 중앙값 기준
        positive = sorted([w for w in weights if w > 0])
        median_w = positive[len(positive) // 2] if positive else 1.0
        if median_w <= 0:
            median_w = 1.0

        # 5) repetition 결정 + 확장 리스트 생성
        expanded: List[str] = []
        rep_map: Dict[str, int] = {}
        for img_p, w in zip(lines, weights):
            if w <= 0:
                rep = 1
            else:
                rep = int(math.ceil(w / median_w))
                rep = max(1, min(int(image_weights_max_rep), rep))
            rep_map[img_p] = rep
            expanded.extend([img_p] * rep)

        rng.shuffle(expanded)
        out_txt.write_text("\n".join(expanded) + "\n", encoding="utf-8")


        def is_rare_image(img_p: str) -> bool:
            cset = img_classset.get(img_p, set())
            return len(cset.intersection(rare_classes)) > 0

        orig_total = len(lines)
        exp_total = len(expanded)

        orig_rare_imgs = [p for p in lines if is_rare_image(p)]
        exp_rare_imgs = [p for p in expanded if is_rare_image(p)]

        orig_rare_ratio = (len(orig_rare_imgs) / orig_total) if orig_total else 0.0
        exp_rare_ratio = (len(exp_rare_imgs) / exp_total) if exp_total else 0.0

        # 희소 이미지 평균 rep
        rare_reps = [rep_map[p] for p in orig_rare_imgs] if orig_rare_imgs else []
        nonrare_imgs = [p for p in lines if not is_rare_image(p)]
        nonrare_reps = [rep_map[p] for p in nonrare_imgs] if nonrare_imgs else []

        avg_rep_all = exp_total / orig_total if orig_total else 1.0
        avg_rep_rare = sum(rare_reps) / len(rare_reps) if rare_reps else 1.0
        avg_rep_nonrare = sum(nonrare_reps) / len(nonrare_reps) if nonrare_reps else 1.0

        # 희소 이미지 중 rep 큰 상위 몇 개 샘플
        top_rep_imgs = sorted(orig_rare_imgs, key=lambda p: rep_map[p], reverse=True)[:10]
        top_rep_samples = [{"img": p, "rep": rep_map[p]} for p in top_rep_imgs]

        # 희소 클래스 topn 출력
        rare_classes_list = [{"class_id": cid, "img_freq": freq} for cid, freq in freq_items[:rare_topn]]

        report = {
            "out_txt": str(out_txt),
            "original": orig_total,
            "expanded": exp_total,
            "avg_rep_all": avg_rep_all,
            "median_weight": float(median_w),
            "image_weights_power": float(image_weights_power),
            "max_rep": int(image_weights_max_rep),

            "rare_percentile": float(rare_percentile),
            "rare_class_count": int(len(rare_classes)),
            "rare_images_original": int(len(orig_rare_imgs)),
            "rare_images_original_ratio": float(orig_rare_ratio),
            "rare_images_weighted_ratio": float(exp_rare_ratio),
            "rare_ratio_multiplier": float((exp_rare_ratio / orig_rare_ratio) if orig_rare_ratio > 0 else 0.0),

            "avg_rep_rare_images": float(avg_rep_rare),
            "avg_rep_nonrare_images": float(avg_rep_nonrare),

            "top_rare_rep_samples": top_rep_samples,
            "rare_classes_lowfreq_topn": rare_classes_list,
        }
        return report

    def patch_yaml_train_path_text(
        self,
        yaml_path: str,
        new_train_txt_path: str,
        out_yaml_path: str,
    ) -> str:
        yp = Path(yaml_path)
        outp = Path(out_yaml_path)
        outp.parent.mkdir(parents=True, exist_ok=True)

        lines = yp.read_text(encoding="utf-8").splitlines()

        # 보통 path: . + train: train80.txt 형태이므로
        # 가급적 파일명만 넣는 게 호환성이 좋음
        new_train_value = str(Path(new_train_txt_path).name)

        replaced = False
        out_lines = []
        for line in lines:
            if line.strip().startswith("train:"):
                prefix = line.split("train:")[0]
                out_lines.append(f"{prefix}train: {new_train_value}")
                replaced = True
            else:
                out_lines.append(line)

        if not replaced:
            out_lines.append(f"train: {new_train_value}")

        outp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        return str(outp)