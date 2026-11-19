"""In-memory view of the Contour research state (Goals, Claims, Runs, Analyses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import ContourConfig, load_config
from .mdio import FrontmatterError, parse_markdown_file, strip_wiki, strip_wiki_list

NODE_DIRS = (
    ("goals", "goal"),
    ("claims", "claim"),
    ("runs", "run"),
    ("analyses", "analysis"),
)
EDGE_KEYS = ("supports", "contradicts", "qualifies")

GOAL_STATUSES = ("active", "achieved", "paused", "abandoned")
CLAIM_STATUSES = ("active", "supported", "falsified", "archived")
VERIFICATION_STATES = ("draft", "agent_audited", "human_verified")
RUN_STATUSES = ("running", "final", "failed")
ANALYSIS_KINDS = ("analysis", "eval")


def normalize_artifacts(raw: Any) -> list[dict[str, Any]]:
    """Normalize frontmatter `artifacts` into a list of `{path, kind?, note?}` dicts."""
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        for name, val in raw.items():
            if isinstance(val, dict):
                entry = dict(val)
                entry.setdefault("kind", str(name))
                if "uri" in entry and "path" not in entry:
                    entry["path"] = entry.pop("uri")
                if "path" in entry:
                    entry["path"] = str(entry["path"])
                    out.append(entry)
            elif val is not None:
                out.append({"path": str(val), "kind": str(name)})
        return out
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                out.append({"path": item})
            elif isinstance(item, dict):
                entry = dict(item)
                if "uri" in entry and "path" not in entry:
                    entry["path"] = entry.pop("uri")
                if "path" in entry:
                    entry["path"] = str(entry["path"])
                    out.append(entry)
    return out


def is_remote_uri(path_str: str) -> bool:
    s = path_str.strip().lower()
    return "://" in s and s.split("://", 1)[0].replace("+", "").replace("-", "").replace(".", "").isalnum()


@dataclass
class ContourNode:
    id: str
    type: str  # inferred from folder: "goal" | "claim" | "run" | "analysis"
    title: str
    path: Path
    frontmatter: dict[str, Any] = field(default_factory=dict)
    body: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    resolved_config: dict[str, Any] = field(default_factory=dict)
    config_error: str | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.frontmatter.get(key, default)

    def refs(self, key: str) -> list[str]:
        return strip_wiki_list(self.frontmatter.get(key))

    @property
    def status(self) -> str:
        return str(self.frontmatter.get("status", "unknown"))

    # --- goal accessors -----------------------------------------------------
    @property
    def open_questions(self) -> list[str]:
        """Parse bullets from `## Open Questions` in the Markdown body."""
        for heading, text in self.sections.items():
            if heading.strip().lower() == "open questions":
                out: list[str] = []
                for ln in text.splitlines():
                    stripped = ln.strip()
                    if not stripped:
                        continue
                    item = stripped.lstrip("-* ").strip()
                    if item:
                        out.append(item)
                return out
        return []

    # --- claim accessors ----------------------------------------------------
    @property
    def verification(self) -> str:
        return str(self.frontmatter.get("verification", "draft"))

    # --- evidence (run & analysis) accessors --------------------------------
    @property
    def bugged(self) -> bool:
        val = self.frontmatter.get("bugged", False)
        return bool(val) if isinstance(val, bool) else False

    @property
    def is_usable_evidence(self) -> bool:
        return self.type in ("run", "analysis") and self.status == "final" and not self.bugged

    @property
    def config(self) -> dict[str, Any]:
        if self.resolved_config:
            return dict(self.resolved_config)
        return dict(self.frontmatter.get("config") or {}) if self.type == "run" else {}

    @property
    def config_path(self) -> str | None:
        val = self.frontmatter.get("config_path")
        return str(val) if val else None

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
    def artifacts(self) -> list[dict[str, Any]]:
        return normalize_artifacts(self.frontmatter.get("artifacts"))

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

    # --- run-specific accessors ---------------------------------------------
    @property
    def parent(self) -> str | None:
        raw = self.frontmatter.get("parent")
        if raw:
            return strip_wiki(raw)
        # Graceful fallback if a legacy file still uses `parent_runs`
        legacy = self.refs("parent_runs")
        return legacy[0] if legacy else None

    @property
    def effective_baseline(self) -> str | None:
        return self.parent

    # --- analysis-specific accessors ----------------------------------------
    @property
    def kind(self) -> str:
        return str(self.frontmatter.get("kind", "analysis"))

    @property
    def subjects(self) -> list[str]:
        return self.refs("subjects")

    @property
    def script(self) -> str | None:
        val = self.frontmatter.get("script")
        return str(val) if val else None

    @property
    def script_sha256(self) -> str | None:
        val = self.frontmatter.get("script_sha256")
        return str(val) if val else None

    @property
    def tldr(self) -> str:
        if "tldr" in self.frontmatter:
            return str(self.frontmatter["tldr"]).strip()
        for heading, text in self.sections.items():
            if any(k in heading.lower() for k in ("takeaway", "tldr", "objective", "hypothesis", "finding")):
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


def _load_run_config(root: Path, fm: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Resolve `config_path` sidecar YAML and/or inline `config:` mapping for a Run."""
    inline = fm.get("config")
    cfg: dict[str, Any] = dict(inline) if isinstance(inline, dict) else {}
    cfg_path_raw = fm.get("config_path")
    if not cfg_path_raw:
        return cfg, None

    p = Path(str(cfg_path_raw))
    resolved = p if p.is_absolute() else (root / p)
    if not resolved.exists():
        return cfg, f"Missing config_path {cfg_path_raw!r}"
    try:
        loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        return cfg, f"Malformed config_path {cfg_path_raw!r}: {exc}"
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        return cfg, f"config_path {cfg_path_raw!r} must contain a YAML mapping"
    merged = dict(loaded)
    merged.update(cfg)
    return merged, None


class ContourGraph:
    def __init__(self, root: Path, config: ContourConfig | None = None) -> None:
        self.root = root
        self.config = config or load_config(root)
        self.goals: dict[str, ContourNode] = {}
        self.claims: dict[str, ContourNode] = {}
        self.runs: dict[str, ContourNode] = {}
        self.analyses: dict[str, ContourNode] = {}
        self.all_nodes: dict[str, ContourNode] = {}
        self.load_errors: list[str] = []
        self.duplicate_ids: list[str] = []
        self._subject_to_analyses: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        stores = {
            "goal": self.goals,
            "claim": self.claims,
            "run": self.runs,
            "analysis": self.analyses,
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
                # Path = identity: ID is filename stem, type is inferred from directory
                nid = path.stem
                resolved_cfg, cfg_err = (
                    _load_run_config(self.root, fm) if ntype == "run" else ({}, None)
                )
                node = ContourNode(
                    id=nid,
                    type=ntype,
                    title=str(fm.get("title", nid)),
                    path=path,
                    frontmatter=fm,
                    body=body.strip(),
                    sections=split_sections(body),
                    resolved_config=resolved_cfg,
                    config_error=cfg_err,
                )
                if nid in self.all_nodes:
                    self.duplicate_ids.append(nid)
                stores[ntype][nid] = node
                self.all_nodes[nid] = node

        for ana in sorted(self.analyses.values(), key=lambda a: (str(a.get("created_at") or ""), a.id)):
            for sid in ana.subjects:
                self._subject_to_analyses.setdefault(sid, []).append(ana.id)

    def get_evidence(self, node_id: str) -> ContourNode | None:
        nid = strip_wiki(node_id)
        return self.runs.get(nid) or self.analyses.get(nid)

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

    def usable_analyses(self) -> list[ContourNode]:
        out: list[ContourNode] = []
        for a in self.analyses.values():
            if not a.is_usable_evidence:
                continue
            if all((sid in self.runs and self.runs[sid].is_usable_evidence) for sid in a.subjects):
                out.append(a)
        return out

    def run_analyses(self, run_id: str, *, kind: str | None = None) -> list[ContourNode]:
        """Return analyses referencing `run_id` in `subjects`, optionally filtered by `kind`."""
        rid = strip_wiki(run_id)
        nodes = [self.analyses[aid] for aid in self._subject_to_analyses.get(rid, []) if aid in self.analyses]
        if kind is not None:
            nodes = [n for n in nodes if n.kind == kind]
        return nodes

    def attached_eval_metrics(self, run_id: str) -> dict[str, tuple[float, ContourNode]]:
        """Return latest usable single-subject `kind: eval` metric values attached to `run_id`."""
        rid = strip_wiki(run_id)
        result: dict[str, tuple[float, ContourNode]] = {}
        for ev in self.run_analyses(rid, kind="eval"):
            if not ev.is_usable_evidence:
                continue
            if len(ev.subjects) != 1 or ev.subjects[0] != rid:
                continue
            for mkey, val in ev.metrics.items():
                result[mkey] = (val, ev)
        return result

    def effective_run_metrics(self, run: ContourNode) -> dict[str, float]:
        """Return run's inline metrics merged with attached usable `kind: eval` metrics (inline wins)."""
        merged = {k: v for k, (v, _) in self.attached_eval_metrics(run.id).items()}
        merged.update(run.metrics)
        return merged
