"""In-memory view of the Contour research state (Goals, Claims, Runs)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ContourConfig, load_config
from .mdio import FrontmatterError, parse_markdown_file, strip_wiki, strip_wiki_list

NODE_DIRS = (
    ("goals", "goal"),
    ("claims", "claim"),
    ("runs", "run"),
)
EDGE_KEYS = ("supports", "contradicts", "qualifies")

GOAL_STATUSES = ("active", "achieved", "paused", "abandoned")
CLAIM_STATUSES = ("active", "supported", "falsified", "superseded")
VERIFICATION_STATES = ("draft", "agent_audited", "human_verified")
RUN_STATUSES = ("running", "final", "failed")
RUN_VALIDITIES = ("ok", "bugged", "superseded")
RUN_TIERS = ("probe", "proxy", "full")

DEFAULT_VALIDITY = "ok"
FACET_KEYS = ("task", "arch_family", "arch_scale", "dataset_train", "tier")


@dataclass
class ContourNode:
    id: str
    type: str  # "goal" | "claim" | "run"
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

    # --- claim accessors ----------------------------------------------------
    @property
    def verification(self) -> str:
        return str(self.frontmatter.get("verification", "draft"))

    # --- run accessors ------------------------------------------------------
    @property
    def validity(self) -> str:
        return str(self.frontmatter.get("validity", DEFAULT_VALIDITY))

    @property
    def is_usable_evidence(self) -> bool:
        return self.type == "run" and self.status == "final" and self.validity == "ok"

    @property
    def interventions(self) -> list[str]:
        return [str(x) for x in (self.frontmatter.get("interventions") or [])]

    @property
    def config(self) -> dict[str, Any]:
        return dict(self.frontmatter.get("config") or {})

    @property
    def metrics(self) -> dict[str, float]:
        raw = self.frontmatter.get("metrics") or {}
        return {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

    @property
    def compute(self) -> dict[str, Any]:
        return dict(self.frontmatter.get("compute") or {})

    @property
    def code(self) -> dict[str, Any]:
        return dict(self.frontmatter.get("code") or {})

    @property
    def gpu_hours(self) -> float | None:
        v = self.compute.get("gpu_hours")
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    @property
    def eval_scopes(self) -> list[str]:
        """Derived eval scopes from metric keys (excluding sys)."""
        seen: list[str] = []
        for k in self.metrics:
            if "/" in k:
                scope = k.split("/", 1)[0]
                if scope != "sys" and scope not in seen:
                    seen.append(scope)
        return seen

    @property
    def eval_datasets(self) -> list[str]:
        """Derived eval dataset scopes (excluding sys, train, val)."""
        return [s for s in self.eval_scopes if s not in ("train", "val")]

    @property
    def parents(self) -> list[str]:
        return self.refs("parent_runs")

    @property
    def effective_baseline(self) -> str | None:
        base = self.frontmatter.get("baseline")
        if base:
            return strip_wiki(base)
        return self.parents[0] if self.parents else None

    @property
    def tldr(self) -> str:
        if "tldr" in self.frontmatter:
            return str(self.frontmatter["tldr"]).strip()
        for heading, text in self.sections.items():
            if any(k in heading.lower() for k in ("takeaway", "tldr", "objective", "hypothesis")):
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
        self.goals: dict[str, ContourNode] = {}
        self.claims: dict[str, ContourNode] = {}
        self.runs: dict[str, ContourNode] = {}
        self.all_nodes: dict[str, ContourNode] = {}
        self.load_errors: list[str] = []
        self.duplicate_ids: list[str] = []
        self._load()

    def _load(self) -> None:
        stores = {
            "goal": self.goals,
            "claim": self.claims,
            "run": self.runs,
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

    def claim_goals(self, claim: ContourNode) -> list[str]:
        return sorted(set(claim.refs("goals")))

    def goal_claims(self, goal_id: str) -> list[ContourNode]:
        """Membership is declared only on the Claim (`goals:`). Derived dynamically."""
        gid = strip_wiki(goal_id)
        return sorted(
            [c for c in self.claims.values() if gid in set(c.refs("goals"))],
            key=lambda c: c.id,
        )

    def claim_edges(self, claim: ContourNode) -> dict[str, list[str]]:
        return {k: claim.refs(k) for k in EDGE_KEYS}

    def usable_runs(self) -> list[ContourNode]:
        return [r for r in self.runs.values() if r.is_usable_evidence]
