#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "Pillow>=10.0.0",
#   "rich>=13.0.0",
#   "ultralytics>=8.0.0",
#   "numpy>=1.24.0",
# ]
# ///
"""
ディレクトリ内のJPEG画像を、写っている人物の服装が似ている順に並ぶよう
新しい名前で別フォルダにコピーするツール。

YOLOv8-pose で人物の胴体領域（両肩〜両腰の矩形）を検出し、その領域の
HSV色ヒストグラムを服装の特徴量として利用する。貪欲法の最近傍順序付け
（NN-TSP）で1次元の順序を決定し、隣り合う画像が似た服装になるよう配置する。

新しいファイル名は `{順序番号:04d}_{元のファイル名}` の形式。
人物が検出できなかった画像は出力フォルダ内の `_no_person/` に元の名前のまま
コピーされる。

元のファイルは変更されない（コピーのみ）。確認してから本番リネームに
進めるためのプレビュー用途を想定している。
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
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_VIS_THRESHOLD = 0.3

# HSV ヒストグラムのビン数（H × S の2次元ヒスト = 128次元）
H_BINS = 16
S_BINS = 8


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


def detect_torso_bbox(model, path: Path) -> "tuple[float, float, float, float] | None":
    """画像から人物の胴体 bbox (x1, y1, x2, y2) を返す。検出失敗時は None。"""
    try:
        results = model(str(path), verbose=False)
    except Exception as e:
        console.print(f"[yellow]警告:[/yellow] {path.name} の姿勢推定に失敗 ({e})")
        return None
    if not results:
        return None
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    if r.keypoints is None or r.keypoints.data is None or len(r.keypoints.data) == 0:
        return None

    confs = r.boxes.conf.cpu().numpy()
    best = int(confs.argmax())
    kpts = r.keypoints.data[best].cpu().numpy()
    xy = kpts[:, :2]
    vis = kpts[:, 2]

    anchor_vis = min(
        vis[KP_LEFT_SHOULDER], vis[KP_RIGHT_SHOULDER],
        vis[KP_LEFT_HIP], vis[KP_RIGHT_HIP],
    )
    if anchor_vis < KP_VIS_THRESHOLD:
        return None

    xs = [xy[i, 0] for i in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP)]
    ys = [xy[i, 1] for i in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP)]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def compute_torso_histogram(path: Path, bbox) -> "np.ndarray | None":
    """胴体 bbox の HSV 2次元ヒストグラム（H_BINS × S_BINS、L1正規化）を返す。"""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            x1, y1, x2, y2 = (int(round(v)) for v in bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                return None
            crop = img.crop((x1, y1, x2, y2)).convert("HSV")
            arr = np.asarray(crop, dtype=np.uint8)
            h_idx = arr[..., 0].astype(np.int32) * H_BINS // 256
            s_idx = arr[..., 1].astype(np.int32) * S_BINS // 256
            flat = (h_idx * S_BINS + s_idx).flatten()
            hist = np.bincount(flat, minlength=H_BINS * S_BINS).astype(np.float32)
            total = hist.sum()
            if total > 0:
                hist /= total
            return hist
    except Exception as e:
        console.print(f"[yellow]警告:[/yellow] {path.name} の切り出しに失敗 ({e})")
        return None


def greedy_nearest_neighbor_order(features: np.ndarray) -> list[int]:
    """貪欲法による最近傍順序付け。

    隣同士が似ているような1次元シーケンスを返す。開始点は平均特徴から最も遠い
    （外れ値的な）画像を選ぶことで、似た色の塊が中央付近にまとまりやすくなる。
    計算量は O(n² × d) だが numpy で行列演算するため数千枚までは実用的。
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
        description="服装が似ている順に並ぶようJPEG画像を新しい名前で別フォルダにコピーする",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", help="JPEGファイルが含まれるフォルダパス")
    parser.add_argument(
        "--output-dir",
        default="ordered_by_outfit",
        metavar="NAME",
        help="出力ディレクトリ名（入力フォルダ内に作成、デフォルト: ordered_by_outfit）",
    )
    args = parser.parse_args()

    target_dir = Path(args.directory).resolve()
    if not target_dir.is_dir():
        console.print(f"[red]エラー:[/red] '{target_dir}' はディレクトリではありません")
        sys.exit(1)

    output_dir = target_dir / args.output_dir
    t_start = time.monotonic()

    console.rule("[bold]服装で並べ替えてコピー")
    console.print(f"  対象フォルダ : [cyan]{target_dir}[/cyan]")
    console.print(f"  出力先       : [cyan]{output_dir}[/cyan]")
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
        model = load_pose_model()

    # ---- 服装特徴抽出 ----
    features: list[np.ndarray] = []
    valid_files: list[Path] = []
    no_person: list[Path] = []
    with make_progress() as progress:
        task = progress.add_task(
            "[magenta]服装特徴を抽出中...[/magenta]", total=len(files)
        )
        for f in files:
            bbox = detect_torso_bbox(model, f)
            hist = compute_torso_histogram(f, bbox) if bbox is not None else None
            progress.update(task, advance=1, description=f"[magenta]{f.name}[/magenta]")
            if hist is None:
                no_person.append(f)
            else:
                features.append(hist)
                valid_files.append(f)

    console.print(
        f"\n[green]✓[/green] 特徴抽出完了  "
        f"[cyan]{len(valid_files)}/{len(files)}[/cyan] 枚で胴体を検出"
        + (f"  [yellow]（{len(no_person)} 枚は人物未検出）[/yellow]" if no_person else "")
    )

    if not valid_files:
        console.print("\n[yellow]人物が検出された画像が無いためコピーをスキップします。[/yellow]")
        sys.exit(0)

    # ---- 順序付け ----
    console.print()
    with console.status("[bold green]近傍順序を計算中..."):
        feat_array = np.stack(features).astype(np.float32)
        order = greedy_nearest_neighbor_order(feat_array)
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
