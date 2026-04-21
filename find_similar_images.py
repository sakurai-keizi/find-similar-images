#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "Pillow>=10.0.0",
#   "imagehash>=4.3.0",
#   "rich>=13.0.0",
# ]
# ///
"""
指定ディレクトリのJPEGファイルを走査し、リサイズ・トリミングされた画像も含めて
類似画像をグループ化して別フォルダに移動する。

閾値の目安（--threshold）:
  0-5  : ほぼ同一ファイル（JPEG再圧縮程度の差）
  6-10 : リサイズ・画質変換された同じ画像（デフォルト）
 11-20 : 軽度のトリミングや色調補正を含む類似画像
 21-30 : 重度のトリミングや加工も含む（誤検知が増える）
"""

import argparse
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image
import imagehash
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
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
from rich.table import Table

console = Console()
DEFAULT_THRESHOLD = 10


def compute_hashes(path: Path) -> "tuple[imagehash.ImageHash, imagehash.ImageHash] | None":
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return imagehash.phash(img), imagehash.dhash(img)
    except Exception as e:
        console.print(f"[yellow]警告:[/yellow] {path.name} を読み込めませんでした ({e})")
        return None


def find_jpeg_files(directory: Path, exclude_prefix: str) -> list[Path]:
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        for f in directory.glob(ext):
            if not str(f).startswith(exclude_prefix):
                files.append(f)
    return sorted(set(files))


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def is_similar(h1, h2, threshold: int) -> bool:
    return (h1[0] - h2[0]) <= threshold or (h1[1] - h2[1]) <= threshold


def group_similar_images(files, hashes, threshold: int, progress, task_id):
    n = len(files)
    uf = UnionFind(n)
    total_pairs = n * (n - 1) // 2
    done = 0

    for i in range(n):
        for j in range(i + 1, n):
            if is_similar(hashes[i], hashes[j], threshold):
                uf.union(i, j)
            done += 1
        progress.update(task_id, completed=done, total=total_pairs)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    return {root: members for root, members in groups.items() if len(members) >= 2}


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


def print_groups_table(files, groups, dry_run: bool, output_dir: Path):
    dry_tag = "[bold yellow]DRY-RUN[/bold yellow] " if dry_run else ""
    for group_num, (_, members) in enumerate(sorted(groups.items()), start=1):
        group_dir = output_dir / f"group_{group_num:04d}"
        console.print(
            f"{dry_tag}[bold cyan]group_{group_num:04d}[/bold cyan]"
            f"  [white]{len(members)} 枚[/white]"
            f"  [dim]→ {group_dir.name}/[/dim]"
        )
        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            expand=False,
            padding=(0, 1),
        )
        table.add_column("ファイル名", style="white", no_wrap=True)
        table.add_column("サイズ", justify="right", style="dim")
        for idx in members:
            src = files[idx]
            try:
                size_kb = src.stat().st_size / 1024
                size_str = f"{size_kb:.1f} KB"
            except OSError:
                size_str = "-"
            table.add_row(src.name, size_str)
        console.print(table)
        console.print()


def move_groups(files, groups, output_dir: Path, dry_run: bool) -> int:
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    with make_progress() as progress:
        total = sum(len(m) for m in groups.values())
        task = progress.add_task(
            "[cyan]移動中...[/cyan]" if not dry_run else "[yellow]確認中...[/yellow]",
            total=total,
        )
        for _, members in sorted(groups.items()):
            group_num = sorted(groups.keys()).index(_) + 1
            group_dir = output_dir / f"group_{group_num:04d}"
            if not dry_run:
                group_dir.mkdir(exist_ok=True)
            for idx in members:
                src = files[idx]
                dst = group_dir / src.name
                if not dry_run and dst.exists():
                    stem, suffix = src.stem, src.suffix
                    counter = 1
                    while dst.exists():
                        dst = group_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                if not dry_run:
                    shutil.move(str(src), dst)
                progress.update(task, advance=1, description=f"[cyan]{src.name}[/cyan]")
                moved_count += 1

    return moved_count


def main():
    parser = argparse.ArgumentParser(
        description="類似JPEG画像をグループ化して別フォルダに移動する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("閾値")[1] if "閾値" in __doc__ else "",
    )
    parser.add_argument("directory", help="JPEGファイルが含まれるフォルダパス")
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        metavar="N",
        help=f"類似判定の閾値（ハミング距離 0〜64、デフォルト: {DEFAULT_THRESHOLD}）",
    )
    parser.add_argument(
        "--output-dir",
        default="similar_groups",
        metavar="NAME",
        help="出力ディレクトリ名（入力フォルダ内に作成、デフォルト: similar_groups）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="移動せずに検出結果だけ表示する",
    )
    args = parser.parse_args()

    target_dir = Path(args.directory).resolve()
    if not target_dir.is_dir():
        console.print(f"[red]エラー:[/red] '{target_dir}' はディレクトリではありません")
        sys.exit(1)

    output_dir = target_dir / args.output_dir
    t_start = time.monotonic()

    console.rule("[bold]類似画像検出")
    console.print(f"  対象フォルダ : [cyan]{target_dir}[/cyan]")
    console.print(f"  閾値         : [cyan]{args.threshold}[/cyan]")
    console.print(f"  出力先       : [cyan]{output_dir}[/cyan]")
    if args.dry_run:
        console.print("  モード       : [yellow]DRY-RUN（移動しない）[/yellow]")
    console.print()

    # ---- ファイル列挙 ----
    with console.status("[bold green]JPEGファイルを列挙中..."):
        files = find_jpeg_files(target_dir, str(output_dir))

    if not files:
        console.print("[yellow]JPEGファイルが見つかりませんでした。[/yellow]")
        sys.exit(0)

    console.print(f"[green]✓[/green] {len(files)} 枚のJPEGファイルを検出しました。\n")

    # ---- ハッシュ計算 ----
    hashes = []
    valid_files: list[Path] = []
    with make_progress() as progress:
        task = progress.add_task("[green]ハッシュ計算中...[/green]", total=len(files))
        for f in files:
            h = compute_hashes(f)
            progress.update(task, advance=1, description=f"[green]{f.name}[/green]")
            if h is not None:
                hashes.append(h)
                valid_files.append(f)

    skipped = len(files) - len(valid_files)
    console.print(
        f"\n[green]✓[/green] ハッシュ計算完了"
        + (f"  [yellow]（{skipped} 枚スキップ）[/yellow]" if skipped else "")
    )

    # ---- 類似検索 ----
    console.print()
    n = len(valid_files)
    total_pairs = n * (n - 1) // 2
    groups: dict[int, list[int]] = {}
    with make_progress() as progress:
        task = progress.add_task(
            f"[blue]類似検索中（閾値 {args.threshold}）...[/blue]",
            total=total_pairs,
        )
        groups = group_similar_images(valid_files, hashes, args.threshold, progress, task)

    if not groups:
        console.print("\n[yellow]類似画像のグループは見つかりませんでした。[/yellow]")
        console.print(f"ヒント: [cyan]--threshold {args.threshold + 10}[/cyan] のように閾値を上げると検出しやすくなります。")
        sys.exit(0)

    total_matched = sum(len(m) for m in groups.values())
    console.print(
        f"\n[green]✓[/green] [bold]{len(groups)} グループ[/bold]、"
        f"計 [bold]{total_matched} 枚[/bold] の類似画像を検出しました。\n"
    )

    # ---- グループ一覧表示 ----
    print_groups_table(valid_files, groups, args.dry_run, output_dir)
    console.print()

    # ---- 移動 ----
    if not args.dry_run:
        moved = move_groups(valid_files, groups, output_dir, dry_run=False)
    else:
        moved = total_matched

    elapsed = time.monotonic() - t_start
    console.rule()
    if args.dry_run:
        console.print(
            f"[yellow][DRY-RUN][/yellow] {moved} 枚が {len(groups)} グループに分類されます"
            f"  [dim]({elapsed:.1f}s)[/dim]"
        )
    else:
        console.print(
            f"[bold green]完了:[/bold green] {moved} 枚を {len(groups)} グループに移動しました"
            f"  [dim]({elapsed:.1f}s)[/dim]"
        )


if __name__ == "__main__":
    main()
