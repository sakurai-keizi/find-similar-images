#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "Pillow>=10.0.0",
#   "rich>=13.0.0",
#   "ultralytics>=8.0.0",
#   "numpy>=1.24.0",
#   "open_clip_torch>=2.20.0",
# ]
# ///
"""
ディレクトリ内のJPEG画像を、写っている人物の服装・髪型が似ている順に並ぶよう
新しい名前で別フォルダにコピーするツール。

3種類の特徴量を組み合わせて類似度を判定する:
  - 胴体領域のHSV色ヒストグラム（服の色）
  - 頭部領域のHSV色ヒストグラム（髪色）
  - CLIP（ViT-B/32）画像埋め込み（服の種類・髪型などの意味的特徴）

YOLOv8-pose で胴体・頭部・人物bboxを検出する。胴体は両肩+両腰の4点、または
両肩のみで上半身を推定。頭部は両目+両耳から推定。CLIPは人物bbox全体に適用
し、服の種類（制服、スカート、水着など）や髪型のスタイルを意味的に捉える。

3つの特徴ベクトルを各々 L2 正規化した上で重み付きで連結し、貪欲法の最近傍
順序付け（NN-TSP）で1次元の順序を決定する。重みは --weight-torso /
--weight-hair / --weight-clip で調整可能。

新しいファイル名は `{順序番号:04d}_{元のファイル名}` の形式。
人物が検出できなかった画像は出力フォルダ内の `_no_person/` に元の名前のまま
コピーされる。元のファイルは変更されない（コピーのみ）。

初回実行時に YOLOv8-pose（~6MB）と CLIP ViT-B/32（~150MB）の重みが自動
ダウンロードされる。
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()

# COCO 17 キーポイントのインデックス
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_VIS_THRESHOLD = 0.3

# HSV ヒストグラムのビン数（H × S の2次元ヒスト = 128次元）
H_BINS = 16
S_BINS = 8

# 各特徴量のデフォルト重み（L2正規化後の連結ベクトルへの寄与）
DEFAULT_W_TORSO = 1.0
DEFAULT_W_HAIR = 0.5
DEFAULT_W_CLIP = 1.5

CLIP_EMBEDDING_DIM = 512  # ViT-B/32


def find_jpeg_files(directory: Path, exclude_prefix: str) -> list[Path]:
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        for f in directory.glob(ext):
            if not str(f).startswith(exclude_prefix):
                files.append(f)
    return sorted(set(files))


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def load_pose_model():
    from ultralytics import YOLO
    return YOLO("yolov8n-pose.pt")


def load_clip_model():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    return model, preprocess


def compute_torso_bbox_from_keypoints(
    xy: np.ndarray, vis: np.ndarray
) -> "tuple[float, float, float, float] | None":
    """胴体 bbox を返す。両肩+両腰の4点があればその矩形、両肩のみなら肩幅で推定。"""
    shoulders_visible = (
        vis[KP_LEFT_SHOULDER] >= KP_VIS_THRESHOLD
        and vis[KP_RIGHT_SHOULDER] >= KP_VIS_THRESHOLD
    )
    hips_visible = (
        vis[KP_LEFT_HIP] >= KP_VIS_THRESHOLD
        and vis[KP_RIGHT_HIP] >= KP_VIS_THRESHOLD
    )

    if shoulders_visible and hips_visible:
        xs = [xy[i, 0] for i in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP)]
        ys = [xy[i, 1] for i in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP)]
        return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

    if shoulders_visible:
        ls = xy[KP_LEFT_SHOULDER]
        rs = xy[KP_RIGHT_SHOULDER]
        shoulder_width = float(np.linalg.norm(ls - rs))
        if shoulder_width < 4:
            return None
        x1 = float(min(ls[0], rs[0]))
        x2 = float(max(ls[0], rs[0]))
        y_top = float(min(ls[1], rs[1]))
        y_bot = y_top + 1.5 * shoulder_width
        return x1, y_top, x2, y_bot

    return None


def compute_hair_bbox_from_keypoints(
    xy: np.ndarray, vis: np.ndarray
) -> "tuple[float, float, float, float] | None":
    """髪色サンプリング用の頭部 bbox を返す。

    両目が可視である必要がある。耳が両方見えていれば耳間の幅、無ければ目間距離 ×2.5
    を顔幅とみなし、目線の少し上から顔幅と同じ高さ分だけ上方向にサンプリング領域を取る。
    """
    if vis[KP_LEFT_EYE] < KP_VIS_THRESHOLD or vis[KP_RIGHT_EYE] < KP_VIS_THRESHOLD:
        return None

    le = xy[KP_LEFT_EYE]
    re = xy[KP_RIGHT_EYE]
    eye_mid = (le + re) / 2
    eye_dist = float(np.linalg.norm(le - re))
    if eye_dist < 4:
        return None

    if (
        vis[KP_LEFT_EAR] >= KP_VIS_THRESHOLD
        and vis[KP_RIGHT_EAR] >= KP_VIS_THRESHOLD
    ):
        l_ear = xy[KP_LEFT_EAR]
        r_ear = xy[KP_RIGHT_EAR]
        x1 = float(min(l_ear[0], r_ear[0]))
        x2 = float(max(l_ear[0], r_ear[0]))
    else:
        # 目間距離は顔幅のおよそ40%なので、目間距離×2.5 を顔幅とみなす
        face_w = eye_dist * 2.5
        x1 = float(eye_mid[0] - face_w / 2)
        x2 = float(eye_mid[0] + face_w / 2)

    width = x2 - x1
    if width < 4:
        return None
    # 目線の少し上から顔幅と同じ高さ分だけ上方向（髪が乗っている領域）
    y2 = float(eye_mid[1] - 0.1 * width)
    y1 = float(eye_mid[1] - 1.1 * width)
    if y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


def detect_bboxes(
    model, path: Path
) -> "tuple[tuple | None, tuple | None, tuple | None]":
    """画像から (torso_bbox, hair_bbox, person_bbox) を返す。各々 None の可能性あり。"""
    try:
        results = model(str(path), verbose=False)
    except Exception as e:
        console.print(f"[yellow]警告:[/yellow] {path.name} の姿勢推定に失敗 ({e})")
        return None, None, None
    if not results:
        return None, None, None
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, None, None
    if r.keypoints is None or r.keypoints.data is None or len(r.keypoints.data) == 0:
        return None, None, None

    confs = r.boxes.conf.cpu().numpy()
    best = int(confs.argmax())

    person_xyxy = r.boxes.xyxy[best].cpu().numpy()
    person_bbox = (
        float(person_xyxy[0]),
        float(person_xyxy[1]),
        float(person_xyxy[2]),
        float(person_xyxy[3]),
    )

    kpts = r.keypoints.data[best].cpu().numpy()
    xy = kpts[:, :2]
    vis = kpts[:, 2]

    torso_bbox = compute_torso_bbox_from_keypoints(xy, vis)
    hair_bbox = compute_hair_bbox_from_keypoints(xy, vis)
    return torso_bbox, hair_bbox, person_bbox


def compute_region_histogram(
    img_rgb: Image.Image, bbox: "tuple[float, float, float, float]"
) -> "np.ndarray | None":
    """RGB PIL 画像の指定 bbox の HSV 2次元ヒストグラム（H_BINS × S_BINS、L1正規化）を返す。"""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_rgb.width, x2)
    y2 = min(img_rgb.height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = img_rgb.crop((x1, y1, x2, y2)).convert("HSV")
    arr = np.asarray(crop, dtype=np.uint8)
    h_idx = arr[..., 0].astype(np.int32) * H_BINS // 256
    s_idx = arr[..., 1].astype(np.int32) * S_BINS // 256
    flat = (h_idx * S_BINS + s_idx).flatten()
    hist = np.bincount(flat, minlength=H_BINS * S_BINS).astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def compute_clip_embedding(
    model, preprocess, img_rgb: Image.Image, bbox: "tuple[float, float, float, float]"
) -> "np.ndarray | None":
    """人物 bbox にクロップした画像を CLIP に通して L2 正規化済み埋め込みを返す。"""
    import torch

    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_rgb.width, x2)
    y2 = min(img_rgb.height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = img_rgb.crop((x1, y1, x2, y2))
    try:
        with torch.no_grad():
            tensor = preprocess(crop).unsqueeze(0)
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().flatten().astype(np.float32)
    except Exception as e:
        console.print(f"[yellow]警告:[/yellow] CLIP 埋め込み計算に失敗 ({e})")
        return None


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def build_combined_feature(
    torso_hist: "np.ndarray | None",
    hair_hist: "np.ndarray | None",
    clip_emb: "np.ndarray | None",
    w_torso: float,
    w_hair: float,
    w_clip: float,
) -> np.ndarray:
    """3つの特徴を L2 正規化して重み付きで連結する。欠損部分はゼロベクトルで埋める。"""
    color_dim = H_BINS * S_BINS

    torso = l2_normalize(torso_hist) if torso_hist is not None else np.zeros(color_dim, dtype=np.float32)
    hair = l2_normalize(hair_hist) if hair_hist is not None else np.zeros(color_dim, dtype=np.float32)
    clip = clip_emb if clip_emb is not None else np.zeros(CLIP_EMBEDDING_DIM, dtype=np.float32)

    return np.concatenate(
        [
            (w_torso * torso).astype(np.float32),
            (w_hair * hair).astype(np.float32),
            (w_clip * clip).astype(np.float32),
        ]
    )


def greedy_nearest_neighbor_order(features: np.ndarray) -> list[int]:
    """貪欲法による最近傍順序付け。

    開始点は平均特徴から最も遠い点（外れ値）。隣同士が似ている1次元シーケンスを返す。
    計算量は O(n² × d) だが numpy ベクトル化されているため数千枚までは実用的。
    """
    n = features.shape[0]
    if n <= 1:
        return list(range(n))

    visited = np.zeros(n, dtype=bool)
    mean = features.mean(axis=0)
    dist_from_mean = np.linalg.norm(features - mean, axis=1)
    current = int(dist_from_mean.argmax())
    order = [current]
    visited[current] = True

    for _ in range(n - 1):
        d = np.linalg.norm(features - features[current], axis=1)
        d[visited] = np.inf
        nxt = int(d.argmin())
        order.append(nxt)
        visited[nxt] = True
        current = nxt
    return order


def main():
    parser = argparse.ArgumentParser(
        description="服装・髪型が似ている順に並ぶようJPEG画像を新しい名前で別フォルダにコピーする",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", help="JPEGファイルが含まれるフォルダパス")
    parser.add_argument(
        "--output-dir",
        default="ordered_by_outfit",
        metavar="NAME",
        help="出力ディレクトリ名（入力フォルダ内に作成、デフォルト: ordered_by_outfit）",
    )
    parser.add_argument(
        "--weight-torso",
        type=float,
        default=DEFAULT_W_TORSO,
        metavar="X",
        help=f"服色ヒストグラムの重み（デフォルト: {DEFAULT_W_TORSO}）",
    )
    parser.add_argument(
        "--weight-hair",
        type=float,
        default=DEFAULT_W_HAIR,
        metavar="X",
        help=f"髪色ヒストグラムの重み（デフォルト: {DEFAULT_W_HAIR}）",
    )
    parser.add_argument(
        "--weight-clip",
        type=float,
        default=DEFAULT_W_CLIP,
        metavar="X",
        help=f"CLIP意味的特徴の重み（デフォルト: {DEFAULT_W_CLIP}）",
    )
    args = parser.parse_args()

    target_dir = Path(args.directory).resolve()
    if not target_dir.is_dir():
        console.print(f"[red]エラー:[/red] '{target_dir}' はディレクトリではありません")
        sys.exit(1)

    output_dir = target_dir / args.output_dir
    t_start = time.monotonic()

    console.rule("[bold]服装・髪型で並べ替えてコピー")
    console.print(f"  対象フォルダ : [cyan]{target_dir}[/cyan]")
    console.print(f"  出力先       : [cyan]{output_dir}[/cyan]")
    console.print(
        f"  重み         : [cyan]服 {args.weight_torso}[/cyan]  "
        f"[cyan]髪 {args.weight_hair}[/cyan]  [cyan]CLIP {args.weight_clip}[/cyan]"
    )
    console.print()

    # ---- ファイル列挙 ----
    with console.status("[bold green]JPEGファイルを列挙中..."):
        files = find_jpeg_files(target_dir, str(output_dir))
    if not files:
        console.print("[yellow]JPEGファイルが見つかりませんでした。[/yellow]")
        sys.exit(0)
    console.print(f"[green]✓[/green] {len(files)} 枚のJPEGファイルを検出しました。\n")

    # ---- モデルロード ----
    with console.status("[bold green]YOLOv8-pose モデルを読み込み中..."):
        pose_model = load_pose_model()
    with console.status("[bold green]CLIP モデルを読み込み中（初回はDLあり、~150MB）..."):
        clip_model, clip_preprocess = load_clip_model()

    # ---- 特徴抽出 ----
    feature_records: list = []
    valid_files: list[Path] = []
    no_person: list[Path] = []
    hair_count = 0
    clip_count = 0

    with make_progress() as progress:
        task = progress.add_task(
            "[magenta]特徴抽出中（YOLO+CLIP）...[/magenta]", total=len(files)
        )
        for f in files:
            torso_bbox, hair_bbox, person_bbox = detect_bboxes(pose_model, f)
            if torso_bbox is None:
                no_person.append(f)
                progress.update(task, advance=1, description=f"[magenta]{f.name}[/magenta]")
                continue

            torso_hist = None
            hair_hist = None
            clip_emb = None
            try:
                with Image.open(f) as img:
                    img_rgb = img.convert("RGB")
                    torso_hist = compute_region_histogram(img_rgb, torso_bbox)
                    if hair_bbox is not None:
                        hair_hist = compute_region_histogram(img_rgb, hair_bbox)
                    if person_bbox is not None:
                        clip_emb = compute_clip_embedding(
                            clip_model, clip_preprocess, img_rgb, person_bbox
                        )
            except Exception as e:
                console.print(f"[yellow]警告:[/yellow] {f.name} 特徴抽出に失敗 ({e})")
                no_person.append(f)
                progress.update(task, advance=1, description=f"[magenta]{f.name}[/magenta]")
                continue

            if torso_hist is None:
                no_person.append(f)
                progress.update(task, advance=1, description=f"[magenta]{f.name}[/magenta]")
                continue

            if hair_hist is not None:
                hair_count += 1
            if clip_emb is not None:
                clip_count += 1
            feature_records.append((torso_hist, hair_hist, clip_emb))
            valid_files.append(f)
            progress.update(task, advance=1, description=f"[magenta]{f.name}[/magenta]")

    detect_summary = (
        f"\n[green]✓[/green] 特徴抽出完了  "
        f"[cyan]{len(valid_files)}/{len(files)}[/cyan] 枚で胴体検出"
        f"  [cyan]({hair_count} 枚で髪領域も)[/cyan]"
        f"  [cyan]({clip_count} 枚でCLIP)[/cyan]"
    )
    if no_person:
        detect_summary += f"  [yellow]（{len(no_person)} 枚は人物未検出）[/yellow]"
    console.print(detect_summary)

    if not valid_files:
        console.print("\n[yellow]人物が検出された画像が無いためコピーをスキップします。[/yellow]")
        sys.exit(0)

    # ---- 特徴量結合 ----
    combined = np.stack(
        [
            build_combined_feature(
                torso, hair, clip,
                args.weight_torso, args.weight_hair, args.weight_clip,
            )
            for torso, hair, clip in feature_records
        ]
    )

    # ---- 順序付け ----
    console.print()
    with console.status("[bold green]近傍順序を計算中..."):
        order = greedy_nearest_neighbor_order(combined)
    console.print(f"[green]✓[/green] 順序付け完了（{len(order)} 枚）\n")

    # ---- コピー ----
    output_dir.mkdir(parents=True, exist_ok=True)
    no_person_dir = output_dir / "_no_person"
    if no_person:
        no_person_dir.mkdir(exist_ok=True)

    width = max(4, len(str(len(order))))
    with make_progress() as progress:
        task = progress.add_task(
            "[cyan]コピー中...[/cyan]", total=len(order) + len(no_person)
        )
        for seq, idx in enumerate(order, start=1):
            src = valid_files[idx]
            dst = output_dir / f"{seq:0{width}d}_{src.name}"
            shutil.copy2(src, dst)
            progress.update(task, advance=1, description=f"[cyan]{dst.name}[/cyan]")
        for src in no_person:
            dst = no_person_dir / src.name
            shutil.copy2(src, dst)
            progress.update(task, advance=1, description=f"[cyan]{dst.name}[/cyan]")

    elapsed = time.monotonic() - t_start
    console.rule()
    summary = f"[bold green]完了:[/bold green] {len(order)} 枚を順序付けてコピー"
    if no_person:
        summary += f"、[yellow]{len(no_person)}[/yellow] 枚を _no_person/ にコピー"
    summary += f"  [dim]({elapsed:.1f}s)[/dim]"
    console.print(summary)
    console.print(
        f"[dim]問題なければ {output_dir} を確認してから本番リネームに進めてください。[/dim]"
    )


if __name__ == "__main__":
    main()
