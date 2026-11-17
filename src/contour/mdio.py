"""Markdown + YAML frontmatter I/O and wikilink helpers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml

WIKI_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
_FM_RE = re.compile(r"\A---[ \t]*\n(?:(.*?)\n)?---[ \t]*(?:\n|\Z)", re.DOTALL)


class FrontmatterError(ValueError):
    """Raised when a Markdown file has malformed YAML frontmatter."""


def strip_wiki(val: Any) -> str:
    """`[[id|alias]]` / `[[id#heading]]` / `id` -> `id`."""
    s = str(val).strip()
    m = WIKI_RE.search(s)
    if m:
        s = m.group(1)
    s = s.split("|", 1)[0].split("#", 1)[0]
    return s.strip()


def strip_wiki_list(vals: Any) -> list[str]:
    if not vals:
        return []
    if isinstance(vals, str):
        return [strip_wiki(vals)]
    return [strip_wiki(v) for v in vals if v]


def as_wikilink(node_id: str) -> str:
    return f"[[{strip_wiki(node_id)}]]"


def parse_markdown(text: str) -> tuple[dict[str, Any], str]:
    """Split a Markdown document into (frontmatter, body). Tolerates CRLF and empty frontmatter."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1) or "") or {}
    except yaml.YAMLError as exc:
        raise FrontmatterError(str(exc)) from exc
    if not isinstance(fm, dict):
        raise FrontmatterError("frontmatter must be a YAML mapping")
    return fm, text[m.end():]


def parse_markdown_file(path: Path) -> tuple[dict[str, Any], str]:
    try:
        return parse_markdown(path.read_text(encoding="utf-8"))
    except FrontmatterError as exc:
        raise FrontmatterError(f"{path}: {exc}") from exc


def render_markdown(frontmatter: dict[str, Any], body: str) -> str:
    fm_str = yaml.safe_dump(
        frontmatter, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).strip()
    content = f"---\n{fm_str}\n---\n\n{body.lstrip(chr(10))}"
    if not content.endswith("\n"):
        content += "\n"
    return content


def write_markdown_file(path: Path, frontmatter: dict[str, Any], body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(frontmatter, body), encoding="utf-8")
