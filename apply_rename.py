#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "rich>=13.0.0",
# ]
# ///
"""
order_by_outfit.py で生成したコピーフォルダの命名を元のファイルに適用して
本番リネームするツール。

ワークフロー:
  1. uv run order_by_outfit.py /path/to/photos
       → /path/to/photos/ordered_by_outfit/ にプレビューがコピーされる
  2. プレビューを確認。不要なファイルはコピーフォルダ内で削除して調整
  3. uv run apply_rename.py /path/to/photos
       → ドライランで計画を表示
  4. uv run apply_rename.py /path/to/photos --apply
       → 実際にリネーム実行。.rename_log.tsv にログ保存

undo:
  uv run apply_rename.py /path/to/photos --undo --apply
  （直前の --apply で書かれたログを使って完全に元に戻す）

クリーンアップ:
  uv run apply_rename.py /path/to/photos --apply --cleanup
  （リネーム成功後にコピーフォルダ全体を削除）

コピーフォルダ内のファイル名は `{seq:04d}_c{cluster:02d}_{元名}` または
`{seq:04d}_{元名}` の形式。この接頭辞を解析して元ファイル `{元名}` を見つけ、
コピー側のファイル名に合わせて元のファイルをリネームする。
ログは入力フォルダ直下の `.rename_log.tsv` に保存される
（コピーフォルダを --cleanup で消してもログは残るので undo 可能）。
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

from rich.console import Console

console = Console()

LOG_FILENAME = ".rename_log.tsv"
PATTERN_WITH_CLUSTER = re.compile(r"^(\d+)_c(\d+)_(.+)$")
PATTERN_NO_CLUSTER = re.compile(r"^(\d+)_(.+)$")


def parse_copy_filename(name: str) -> "str | None":
    """コピー側のファイル名から元のファイル名部分を抜き出す。

    `0001_c01_foo.jpg` → `foo.jpg`
    `0050_foo.jpg`     → `foo.jpg`（クラスタIDなし、人物未検出パス）
    パターンに合わなければ None。
    """
    m = PATTERN_WITH_CLUSTER.match(name)
    if m:
        return m.group(3)
    m = PATTERN_NO_CLUSTER.match(name)
    if m:
        return m.group(2)
    return None


def collect_rename_plan(target_dir: Path, copy_dir: Path):
    """(plan, missing, skipped, duplicates) を返す。

    plan: [(元ファイルパス, 新ファイル名), ...]（接頭辞付きの新名）
    missing: コピー側に対応する元ファイルがなかった元名のリスト
    skipped: パターンに合わなかったコピー側ファイル名のリスト
    duplicates: 同じ元ファイルを複数回参照しているコピー側のリスト
    """
    plan = []
    missing = []
    skipped = []
    seen_originals: dict[Path, str] = {}
    duplicates = []

    files = sorted(copy_dir.iterdir(), key=lambda p: p.name)
    for f in files:
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        original_name = parse_copy_filename(f.name)
        if original_name is None:
            skipped.append(f.name)
            continue
        original_path = target_dir / original_name
        if not original_path.is_file():
            missing.append(original_name)
            continue
        if original_path in seen_originals:
            duplicates.append((original_name, seen_originals[original_path], f.name))
            continue
        seen_originals[original_path] = f.name
        plan.append((original_path, f.name))

    return plan, missing, skipped, duplicates


def write_log(log_path: Path, plan: list):
    """rename ログを TSV で書く。各行: 新ファイル名\\t元ファイル名"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        for src, new_name in plan:
            logf.write(f"{new_name}\t{src.name}\n")


def read_log(log_path: Path) -> list:
    pairs = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            pairs.append((parts[0], parts[1]))
    return pairs


def show_plan_preview(plan, max_show: int = 10):
    for src, new_name in plan[:max_show]:
        if src.name == new_name:
            console.print(f"  [dim]{src.name}  (既に同名、スキップ)[/dim]")
        else:
            console.print(f"  {src.name}  →  [green]{new_name}[/green]")
    if len(plan) > max_show:
        console.print(f"  [dim]... (+{len(plan) - max_show} 件)[/dim]")


def cmd_undo(target_dir: Path, log_path: Path, apply: bool) -> int:
    if not log_path.is_file():
        console.print(f"[red]エラー:[/red] ログが見つかりません: {log_path}")
        return 1
    pairs = read_log(log_path)
    console.rule("[bold]Undo（元に戻す）")
    console.print(f"  対象フォルダ : [cyan]{target_dir}[/cyan]")
    console.print(f"  ログ         : [cyan]{log_path}[/cyan]")
    console.print(f"  項目数       : [cyan]{len(pairs)}[/cyan]\n")

    for new_name, orig_name in pairs[:10]:
        console.print(f"  {new_name}  →  [yellow]{orig_name}[/yellow]")
    if len(pairs) > 10:
        console.print(f"  [dim]... (+{len(pairs) - 10} 件)[/dim]")
    console.print()

    if not apply:
        console.print("[yellow]DRY-RUN[/yellow] 実行するには [cyan]--apply[/cyan] を付けてください")
        return 0

    # 衝突チェック: 戻し先が既に他のファイルで占有されていないか
    for new_name, orig_name in pairs:
        src = target_dir / new_name
        dst = target_dir / orig_name
        if dst.exists() and dst.resolve() != src.resolve():
            console.print(
                f"[red]エラー:[/red] {orig_name} が既に存在します"
                f"（{new_name} の戻し先と衝突）。中止"
            )
            return 1

    succeeded = 0
    skipped_missing = 0
    for new_name, orig_name in pairs:
        src = target_dir / new_name
        dst = target_dir / orig_name
        if src.resolve() == dst.resolve():
            continue
        if not src.exists():
            console.print(f"[yellow]スキップ:[/yellow] {new_name} が見つかりません")
            skipped_missing += 1
            continue
        src.rename(dst)
        succeeded += 1

    log_path.unlink()
    console.rule()
    console.print(
        f"[bold green]✓ Undo 完了:[/bold green] {succeeded} 件を元に戻しました"
        + (f"  [yellow]（{skipped_missing} 件は不在でスキップ）[/yellow]" if skipped_missing else "")
    )
    console.print("[dim]ログファイルは削除しました[/dim]")
    return 0


def cmd_forward(target_dir: Path, copy_dir: Path, log_path: Path, apply: bool, cleanup: bool) -> int:
    if not copy_dir.is_dir():
        console.print(f"[red]エラー:[/red] コピーフォルダが見つかりません: {copy_dir}")
        console.print("先に [cyan]order_by_outfit.py[/cyan] を実行してください")
        return 1

    plan, missing, skipped, duplicates = collect_rename_plan(target_dir, copy_dir)

    console.rule("[bold]本番リネーム計画")
    console.print(f"  対象フォルダ : [cyan]{target_dir}[/cyan]")
    console.print(f"  コピー先     : [cyan]{copy_dir}[/cyan]")
    console.print(f"  リネーム対象 : [cyan]{len(plan)}[/cyan] 件")
    if missing:
        console.print(
            f"  [yellow]元ファイル不在: {len(missing)} 件[/yellow]"
            f"  [dim]例: {', '.join(missing[:3])}[/dim]"
        )
    if skipped:
        console.print(
            f"  [yellow]パターン不一致でスキップ: {len(skipped)} 件[/yellow]"
            f"  [dim]例: {', '.join(skipped[:3])}[/dim]"
        )
    if duplicates:
        console.print(f"  [red]重複参照: {len(duplicates)} 件[/red]  [dim](同じ元ファイルを複数のコピーが参照)[/dim]")
        for orig, first, second in duplicates[:3]:
            console.print(f"    [dim]{orig}: {first} と {second}[/dim]")
        console.print("[red]重複があると衝突するため中止します。コピーフォルダから不要分を削除してください[/red]")
        return 1
    console.print()

    show_plan_preview(plan)
    console.print()

    if not plan:
        console.print("[yellow]リネーム対象がありません[/yellow]")
        return 0

    if not apply:
        console.print("[yellow]DRY-RUN[/yellow] 実行するには [cyan]--apply[/cyan] を付けてください")
        return 0

    # 衝突チェック: 自身でないファイルがリネーム先に既存していないか
    for src, new_name in plan:
        dst = target_dir / new_name
        if dst.exists() and dst.resolve() != src.resolve():
            console.print(
                f"[red]エラー:[/red] {new_name} が既に存在します"
                f"（{src.name} のリネーム先と衝突）。中止"
            )
            return 1

    if log_path.exists():
        console.print(f"[yellow]既存のログ {log_path.name} を上書きします[/yellow]")

    # ログを先に書く（途中で失敗しても undo できるように）
    write_log(log_path, plan)

    succeeded = 0
    for src, new_name in plan:
        dst = target_dir / new_name
        if src.resolve() == dst.resolve():
            continue
        src.rename(dst)
        succeeded += 1

    console.rule()
    console.print(f"[bold green]✓ 完了:[/bold green] {succeeded} 件をリネームしました")
    console.print(f"  ログ : [cyan]{log_path}[/cyan]")
    console.print(
        f"  undo : [dim]uv run apply_rename.py {target_dir} --undo --apply[/dim]"
    )

    if cleanup:
        shutil.rmtree(copy_dir)
        console.print(f"[dim]コピーフォルダ {copy_dir} を削除しました[/dim]")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="コピーフォルダの命名を元のファイルに適用して本番リネームする",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", help="JPEGファイルがあるフォルダ")
    parser.add_argument(
        "--copy-dir",
        default="ordered_by_outfit",
        metavar="NAME",
        help="プレビューのコピー先フォルダ名（デフォルト: ordered_by_outfit）",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="実際にリネームする（指定しないと dry-run）",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="リネーム成功後にコピーフォルダを削除する",
    )
    parser.add_argument(
        "--undo",
        action="store_true",
        help="前回の --apply で書かれたログを使ってリネームを元に戻す",
    )
    args = parser.parse_args()

    target_dir = Path(args.directory).resolve()
    if not target_dir.is_dir():
        console.print(f"[red]エラー:[/red] {target_dir} はディレクトリではありません")
        sys.exit(1)

    log_path = target_dir / LOG_FILENAME

    if args.undo:
        sys.exit(cmd_undo(target_dir, log_path, apply=args.apply))

    copy_dir = target_dir / args.copy_dir
    sys.exit(cmd_forward(target_dir, copy_dir, log_path, apply=args.apply, cleanup=args.cleanup))


if __name__ == "__main__":
    main()
