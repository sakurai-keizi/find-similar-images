#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
order_by_outfit.py で並び順のプレビューコピーを作成し、続けて apply_rename.py
で本番リネーム + コピーフォルダ削除まで一気に実行するパイプライン。

レビュー手順を完全にスキップして一気通貫でリネームまで進める。途中失敗時は
その時点で中断する。本番リネームに進んだあとも、入力フォルダ直下の
`.rename_log.tsv` が残っているので
  uv run apply_rename.py <dir> --undo --apply
で完全に元に戻せる。

使い方:
  uv run pipeline.py <directory> [order_by_outfit のオプション...]

例:
  uv run pipeline.py ~/photos
  uv run pipeline.py ~/photos --clusters 12 --mask
  uv run pipeline.py ~/photos --output-dir my_preview --weight-clip 2.0

挙動:
  1. order_by_outfit.py <directory> [追加オプション] を実行
  2. 成功したら apply_rename.py <directory> --copy-dir <output-dir>
     --apply --cleanup を実行（コピーフォルダは削除される）
  3. どちらかが失敗したらその exit code で終了

--output-dir を渡された場合は apply_rename.py の --copy-dir にも自動で
反映するので、両ステップでフォルダ名を合わせる必要はない。
"""

import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent.resolve()
ORDER_SCRIPT = THIS_DIR / "order_by_outfit.py"
RENAME_SCRIPT = THIS_DIR / "apply_rename.py"
DEFAULT_OUTPUT_DIR = "ordered_by_outfit"


def extract_output_dir(args: list) -> str:
    """forward_args から --output-dir の値を抜き出す。"""
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--output-dir" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--output-dir="):
            return a.split("=", 1)[1]
        i += 1
    return DEFAULT_OUTPUT_DIR


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    target = sys.argv[1]
    forward_args = sys.argv[2:]
    output_dir_name = extract_output_dir(forward_args)

    print("=== Step 1/2: order_by_outfit.py ===", flush=True)
    r1 = subprocess.run(["uv", "run", str(ORDER_SCRIPT), target] + forward_args)
    if r1.returncode != 0:
        print(
            f"\norder_by_outfit.py が失敗しました（exit {r1.returncode}）。中止します。",
            file=sys.stderr,
        )
        sys.exit(r1.returncode)

    print("\n=== Step 2/2: apply_rename.py --apply --cleanup ===", flush=True)
    r2 = subprocess.run(
        [
            "uv", "run", str(RENAME_SCRIPT), target,
            "--copy-dir", output_dir_name,
            "--apply", "--cleanup",
        ]
    )
    if r2.returncode != 0:
        print(
            f"\napply_rename.py が失敗しました（exit {r2.returncode}）。",
            file=sys.stderr,
        )
        print(
            f"コピーフォルダ {output_dir_name}/ が残っているはずなので、"
            f"原因を解消してから apply_rename.py を手動で再実行できます。",
            file=sys.stderr,
        )
        sys.exit(r2.returncode)

    print("\n=== 全ステップ完了 ===", flush=True)


if __name__ == "__main__":
    main()
