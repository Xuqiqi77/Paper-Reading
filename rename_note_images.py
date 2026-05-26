#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rename pasted markdown images like:
  image.png, image-1.png, image-2.png
to:
  ADP.png, ADP-1.png, ADP-2.png

And update references in:
  ADP.md

Usage:
  python rename_note_images.py --name ADP
  python rename_note_images.py --dir ./papernotes/26_5_agent --name AgentGym-RL
  python rename_note_images.py --name ADP --dry-run
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


def is_default_image_name(path: Path, old_prefix: str = "image") -> bool:
    """
    Match:
      image.png
      image-1.png
      image-2.jpg
    But not:
      image-abc.png
      my-image-1.png
    """
    if path.suffix.lower() not in IMAGE_EXTS:
        return False

    pattern = rf"^{re.escape(old_prefix)}(?:-(\d+))?$"
    return re.match(pattern, path.stem) is not None


def image_sort_key(path: Path, old_prefix: str = "image") -> tuple[int, str]:
    """
    Sort image.png before image-1.png before image-2.png.
    """
    if path.stem == old_prefix:
        return (0, path.name)

    match = re.match(rf"^{re.escape(old_prefix)}-(\d+)$", path.stem)
    if match:
        return (int(match.group(1)), path.name)

    return (10**9, path.name)


def target_image_name(src: Path, new_prefix: str, old_prefix: str = "image") -> str:
    """
    image.png    -> ADP.png
    image-1.png  -> ADP-1.png
    image-2.jpg  -> ADP-2.jpg
    """
    if src.stem == old_prefix:
        return f"{new_prefix}{src.suffix}"

    match = re.match(rf"^{re.escape(old_prefix)}-(\d+)$", src.stem)
    if not match:
        raise ValueError(f"Not a default image name: {src.name}")

    return f"{new_prefix}-{match.group(1)}{src.suffix}"


def collect_default_images(directory: Path, old_prefix: str = "image") -> list[Path]:
    images = [
        p for p in directory.iterdir()
        if p.is_file() and is_default_image_name(p, old_prefix=old_prefix)
    ]
    return sorted(images, key=lambda p: image_sort_key(p, old_prefix=old_prefix))


def build_rename_plan(
    directory: Path,
    new_prefix: str,
    old_prefix: str = "image",
) -> list[tuple[Path, Path]]:
    images = collect_default_images(directory, old_prefix=old_prefix)

    plan: list[tuple[Path, Path]] = []
    for src in images:
        dst = directory / target_image_name(src, new_prefix, old_prefix=old_prefix)
        plan.append((src, dst))

    return plan


def check_conflicts(plan: Iterable[tuple[Path, Path]]) -> None:
    """
    Avoid overwriting existing files unless it is the same source file.
    """
    plan = list(plan)
    sources = {src.resolve() for src, _ in plan}

    for src, dst in plan:
        if dst.exists() and dst.resolve() not in sources:
            raise FileExistsError(
                f"Target already exists: {dst}\n"
                f"Refuse to overwrite it. Rename/delete it first."
            )


def rename_images(
    directory: Path,
    new_prefix: str,
    old_prefix: str = "image",
    dry_run: bool = False,
) -> list[tuple[Path, Path]]:
    """
    Rename images safely.

    To avoid conflicts during rename, first rename to temporary names,
    then rename temporary names to final names.
    """
    plan = build_rename_plan(directory, new_prefix, old_prefix=old_prefix)
    check_conflicts(plan)

    if not plan:
        print(f"No {old_prefix}*.image files found in: {directory}")
        return []

    print("Rename plan:")
    for src, dst in plan:
        print(f"  {src.name} -> {dst.name}")

    if dry_run:
        print("\nDry run: no files renamed.")
        return plan

    temp_plan: list[tuple[Path, Path, Path]] = []
    for i, (src, dst) in enumerate(plan):
        tmp = directory / f".__tmp_rename_note_images_{i}{src.suffix}"
        temp_plan.append((src, tmp, dst))

    for src, tmp, _ in temp_plan:
        src.rename(tmp)

    for _, tmp, dst in temp_plan:
        tmp.rename(dst)

    print("\nImages renamed.")
    return plan


def replace_markdown_references(
    md_path: Path,
    old_prefix: str,
    new_prefix: str,
    dry_run: bool = False,
) -> int:
    """
    Replace markdown image/file references:
      image.png       -> ADP.png
      image-1.png     -> ADP-1.png
      ./image-1.png   -> ./ADP-1.png
      images/image.png -> images/ADP.png

    It only changes file basenames matching image/image-N with image extensions.
    """
    if not md_path.exists():
        print(f"Markdown file not found, skip: {md_path}")
        return 0

    text = md_path.read_text(encoding="utf-8")

    ext_pattern = "|".join(re.escape(ext.lstrip(".")) for ext in IMAGE_EXTS)

    # Match path prefix + basename + extension.
    # Examples matched:
    # image.png
    # image-1.png
    # ./image.png
    # assets/image-2.webp
    pattern = re.compile(
        rf"(?P<prefix>(?:\.?/)?(?:[\w\-. ]+/)*)"
        rf"{re.escape(old_prefix)}"
        rf"(?P<num>-\d+)?"
        rf"(?P<ext>\.(?:{ext_pattern}))",
        flags=re.IGNORECASE,
    )

    def repl(match: re.Match[str]) -> str:
        prefix = match.group("prefix") or ""
        num = match.group("num") or ""
        ext = match.group("ext")
        return f"{prefix}{new_prefix}{num}{ext}"

    new_text, count = pattern.subn(repl, text)

    if count == 0:
        print(f"No markdown references replaced in: {md_path.name}")
        return 0

    print(f"Markdown replacements in {md_path.name}: {count}")

    if dry_run:
        print("Dry run: markdown not modified.")
        return count

    md_path.write_text(new_text, encoding="utf-8")
    return count


def run(
    directory: Path,
    name: str,
    md_file: str | None,
    old_prefix: str = "image",
    rename_only: bool = False,
    md_only: bool = False,
    dry_run: bool = False,
) -> None:
    directory = directory.resolve()

    if not directory.exists() or not directory.is_dir():
        raise NotADirectoryError(f"Invalid directory: {directory}")

    if md_file is None:
        md_path = directory / f"{name}.md"
    else:
        md_path = directory / md_file

    if not md_only:
        rename_images(
            directory=directory,
            new_prefix=name,
            old_prefix=old_prefix,
            dry_run=dry_run,
        )

    if not rename_only:
        replace_markdown_references(
            md_path=md_path,
            old_prefix=old_prefix,
            new_prefix=name,
            dry_run=dry_run,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename pasted markdown images and update markdown references."
    )

    parser.add_argument(
        "--dir",
        default=".",
        help="Target directory. Default: current directory.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Expected basename, e.g. ADP or AgentGym-RL.",
    )
    parser.add_argument(
        "--md",
        default=None,
        help="Markdown file name. Default: <name>.md",
    )
    parser.add_argument(
        "--old-prefix",
        default="image",
        help="Old pasted image prefix. Default: image",
    )
    parser.add_argument(
        "--rename-only",
        action="store_true",
        help="Only rename image files, do not modify markdown.",
    )
    parser.add_argument(
        "--md-only",
        action="store_true",
        help="Only modify markdown, do not rename image files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files.",
    )

    args = parser.parse_args()

    if args.rename_only and args.md_only:
        parser.error("--rename-only and --md-only cannot be used together.")

    run(
        directory=Path(args.dir),
        name=args.name,
        md_file=args.md,
        old_prefix=args.old_prefix,
        rename_only=args.rename_only,
        md_only=args.md_only,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()