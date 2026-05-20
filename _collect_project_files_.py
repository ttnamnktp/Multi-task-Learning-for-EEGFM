#!/usr/bin/env python3
"""
collect_project_files.py
Quét toàn bộ file .py, .sh, .yaml trong thư mục dự án
và tổng hợp nội dung thành một file Markdown.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

# Các đuôi file cần thu thập
TARGET_EXTENSIONS = {".py", ".sh", ".yaml", ".yml"}

# Mapping đuôi file → tên ngôn ngữ cho code fence
LANG_MAP = {
    ".py":   "python",
    ".sh":   "bash",
    ".yaml": "yaml",
    ".yml":  "yaml",
}


def collect_files(root: Path, exclude_dirs: set[str], exclude_files: set[str]) -> list[Path]:
    result = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in exclude_dirs and not d.startswith("outputs") and not d.endswith("bak")
        )

        for fname in sorted(filenames):
            if fname in exclude_files:
                continue

            fpath = Path(dirpath) / fname

            if fpath.suffix.lower() in TARGET_EXTENSIONS:
                result.append(fpath)

    return result


def read_file_safe(fpath: Path) -> str:
    """Đọc nội dung file, trả về thông báo lỗi nếu không đọc được."""
    try:
        return fpath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return fpath.read_text(encoding="latin-1")
        except Exception as e:
            return f"[Không thể đọc file: {e}]"
    except Exception as e:
        return f"[Lỗi: {e}]"


def build_markdown(root: Path, files: list[Path]) -> str:
    """Tạo nội dung Markdown từ danh sách file."""
    lines = []

    # ── Tiêu đề ──────────────────────────────────────────────────────────
    lines.append(f"# Project Source Map — `{root.name}`\n")
    lines.append(
        f"> Được tạo lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        f"> Thư mục gốc: `{root.resolve()}`  \n"
        f"> Tổng số file: **{len(files)}**\n"
    )

    # ── Mục lục ───────────────────────────────────────────────────────────
    lines.append("\n---\n\n## Mục lục\n")
    for fpath in files:
        rel = fpath.relative_to(root)
        # Tạo anchor kiểu GitHub Markdown
        anchor = str(rel).replace("/", "").replace(".", "").replace("_", "-").replace(" ", "-").lower()
        lines.append(f"- [`{rel}`](#{anchor})")

    # ── Nội dung từng file ────────────────────────────────────────────────
    lines.append("\n---\n")
    for fpath in files:
        rel      = fpath.relative_to(root)
        lang     = LANG_MAP.get(fpath.suffix.lower(), "")
        content  = read_file_safe(fpath)
        size_kb  = fpath.stat().st_size / 1024

        lines.append(f"\n## `{rel}`\n")
        lines.append(f"**Đường dẫn đầy đủ:** `{fpath.resolve()}`  ")
        lines.append(f"**Kích thước:** `{size_kb:.1f} KB`  ")
        lines.append(f"**Loại:** `{fpath.suffix}`\n")
        lines.append(f"```{lang}")
        lines.append(content.rstrip())
        lines.append("```\n")
        lines.append("---\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Tổng hợp source code dự án thành một file Markdown."
    )
    parser.add_argument(
        "project_dir",
        nargs="?",
        default="eegfm",
        help="Thư mục gốc của dự án (mặc định: eegfm)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Tên file output .md (mặc định: <tên_dự_án>_source_map.md)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[
            ".git", "__pycache__", ".venv", "venv", "env",
            "node_modules", ".mypy_cache", ".pytest_cache",
            "dist", "build", ".tox",
            "logs", "outputs", "bash_logs",
        ],
        help="Danh sách thư mục cần bỏ qua (mặc định: .git, __pycache__, …)",
    )
    args = parser.parse_args()

    root = Path(args.project_dir).resolve()
    if not root.exists():
        print(f"[LỖI] Thư mục không tồn tại: {root}")
        return 1
    if not root.is_dir():
        print(f"[LỖI] Đây không phải thư mục: {root}")
        return 1

    output_path = Path(args.output) if args.output else Path(f"{root.name}_source_map.md")

    print(f"📂 Đang quét: {root}")
    print(f"🚫 Bỏ qua:    {', '.join(args.exclude)}")

    exclude_files = {Path(__file__).name}
    files = collect_files(root, set(args.exclude), exclude_files)

    if not files:
        print("⚠️  Không tìm thấy file nào phù hợp (.py / .sh / .yaml / .yml).")
        return 0

    print(f"✅ Tìm thấy {len(files)} file. Đang tạo Markdown …")

    md_content = build_markdown(root, files)
    output_path.write_text(md_content, encoding="utf-8")

    print(f"📄 Đã lưu: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())