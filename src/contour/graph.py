"""In-memory view of the Contour graph (Expeditions, Cairns, FieldLogs)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ContourConfig, load_config
from .mdio import FrontmatterError, parse_markdown_file, strip_wiki_list

DEFAULT_VALIDITY = "ok"
UNRESOLVED_PREFIX = "run:"

NODE_DIRS = (
    ("expeditions", "expedition"),
    ("cairns", "cairn"),
    ("fieldlogs", "fieldlog"),
)
EDGE_KEYS = ("supports", "contradicts", "qualifies")


@dataclass
class ContourNode:
    id: str
    type: str  # "expedition" | "cairn" | "fieldlog"
    title: str
    path: Path
    frontmatter: dict[str, Any] = field(default_factory=dict)
    body: str = ""
    sections: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.frontmatter.get(key, default)

    def refs(self, key: str) -> list[str]:
        return strip_wiki_list(self.frontmatter.get(key))

    @property
    def status(self) -> str:
        return str(self.frontmatter.get("status", "unknown"))

    # --- fieldlog accessors -------------------------------------------------
    @property
    def validity(self) -> str:
        return str(self.frontmatter.get("validity", DEFAULT_VALIDITY))

    @property
    def is_usable_evidence(self) -> bool:
        return self.type == "fieldlog" and self.status == "final" and self.validity == "ok"

    @property
    def interventions(self) -> frozenset[str]:
        return frozenset(str(x) for x in (self.frontmatter.get("interventions") or []))

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self.frontmatter.get("metrics") or {})

    @property
    def telemetry(self) -> dict[str, float]:
        return dict(self.frontmatter.get("telemetry") or {})

    @property
    def gpu_hours(self) -> float | None:
        v = (self.frontmatter.get("compute") or {}).get("gpu_hours")
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def evals(self) -> list[str]:
        return [str(x) for x in (self.frontmatter.get("dataset_eval") or [])]

    @property
    def parents(self) -> list[str]:
        return [p for p in self.refs("parent_runs") if not p.startswith(UNRESOLVED_PREFIX)]

    @property
    def tldr(self) -> str:
        if "tldr" in self.frontmatter:
            return str(self.frontmatter["tldr"]).strip()
        for heading, text in self.sections.items():
            if any(k in heading.lower() for k in ("takeaway", "tldr", "proposition", "north star")):
                first = next((ln.strip("- *").strip() for ln in text.splitlines() if ln.strip()), "")
                if first:
                    return first
        first = next(
            (ln.strip("- *").strip() for ln in self.body.splitlines() if ln.strip() and not ln.startswith("#")),
            "",
        )
        return first[:140]


def split_sections(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    heading, buf = "Overview", []
    for line in body.splitlines():
        if line.startswith("## "):
            if buf:
                sections[heading] = "\n".join(buf).strip()
            heading, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if buf:
        sections[heading] = "\n".join(buf).strip()
    return sections


class ContourGraph:
    def __init__(self, root: Path, config: ContourConfig | None = None) -> None:
        self.root = root
        self.config = config or load_config(root)
        self.expeditions: dict[str, ContourNode] = {}
        self.cairns: dict[str, ContourNode] = {}
        self.fieldlogs: dict[str, ContourNode] = {}
        self.all_nodes: dict[str, ContourNode] = {}
        self.load_errors: list[str] = []
        self.duplicate_ids: list[str] = []
        self._load()

    def _load(self) -> None:
        stores = {
            "expedition": self.expeditions,
            "cairn": self.cairns,
            "fieldlog": self.fieldlogs,
        }
        for folder, ntype in NODE_DIRS:
            d = self.root / folder
            if not d.exists():
                continue
            for path in sorted(d.glob("*.md")):
                try:
                    fm, body = parse_markdown_file(path)
                except FrontmatterError as exc:
                    self.load_errors.append(str(exc))
                    continue
                nid = str(fm.get("id", path.stem))
                node = ContourNode(
                    id=nid,
                    type=str(fm.get("type", ntype)),
                    title=str(fm.get("title", nid)),
                    path=path,
                    frontmatter=fm,
                    body=body.strip(),
                    sections=split_sections(body),
                )
                if nid in self.all_nodes:
                    self.duplicate_ids.append(nid)
                stores[ntype][nid] = node
                self.all_nodes[nid] = node

    def cairn_expeditions(self, cairn: ContourNode) -> list[str]:
        expds = set(cairn.refs("expeditions"))
        for e in self.expeditions.values():
            if cairn.id in e.refs("cairns"):
                expds.add(e.id)
        return sorted(expds)

    def expedition_cairns(self, expd_id: str) -> list[ContourNode]:
        ids: set[str] = set()
        if expd_id in self.expeditions:
            ids.update(self.expeditions[expd_id].refs("cairns"))
        for c in self.cairns.values():
            if expd_id in c.refs("expeditions"):
                ids.add(c.id)
        return [self.cairns[i] for i in sorted(ids) if i in self.cairns]

    def cairn_edges(self, cairn: ContourNode) -> dict[str, list[str]]:
        return {k: cairn.refs(k) for k in EDGE_KEYS}

    def usable_fieldlogs(self) -> list[ContourNode]:
        return [f for f in self.fieldlogs.values() if f.is_usable_evidence]
