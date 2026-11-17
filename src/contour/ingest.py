"""Ingest validated Field Bundles into `bundles/` (archive) and `fieldlogs/` (cards).

Field ownership
---------------
* Machine-owned frontmatter keys (MACHINE_KEYS) are rewritten from the bundle on every ingest,
  including `vs_parent` (auto-computed diff against the first resolved parent run).
* Everything else in the frontmatter (`title`, `validity`, `validity_note`, `tldr`, `tags`, ...)
  and the entire Markdown body are human-owned and are never overwritten.
* Once a run is terminal (`final` / `failed`), its evidence is immutable: re-ingesting a
  *different* bundle for the same `run_id` is rejected unless `force=True`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .bundle import TERMINAL_STATUSES, canonical_json, fieldlog_id_for, validate_bundle
from .compare import compute_vs_parent
from .config import load_config
from .haze import load_haze
from .mdio import as_wikilink, parse_markdown_file, strip_wiki, write_markdown_file

FIELDLOG_DIR = "fieldlogs"
BUNDLE_DIR = "bundles"
CAIRN_DIR = "cairns"

VALIDITY_VALUES = ("ok", "bugged", "noisy", "superseded")
DEFAULT_VALIDITY = "ok"

# Order matters: this is the order keys appear in the card.
MACHINE_KEYS = (
    "id", "type", "run_id", "status", "tier", "created_at", "finished_at",
    "task", "arch_family", "arch_scale", "seed", "dataset_train", "dataset_eval",
    "interventions", "intervention_note", "parent_runs", "motivated_by",
    "compute", "git_commit", "dirty_tree", "launch_cmd",
    "metrics", "telemetry", "vs_parent", "artifacts", "bundle",
)
# Human-owned keys that get a default on card creation and are placed near the top.
HUMAN_SEEDED_KEYS = ("title", "validity")

UNRESOLVED_PREFIX = "run:"


@dataclass
class IngestResult:
    action: str  # created | updated | unchanged | rejected
    log_id: str | None
    path: Path | None
    message: str = ""
    flagged: list[str] = field(default_factory=list)


class FieldlogIndex:
    """run_id -> fieldlog id lookup over the current fieldlogs/ directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.by_run_id: dict[str, str] = {}
        self.by_log_id: dict[str, dict[str, Any]] = {}
        d = root / FIELDLOG_DIR
        if d.exists():
            for p in sorted(d.glob("*.md")):
                fm, _ = parse_markdown_file(p)
                lid = str(fm.get("id", p.stem))
                self.add(lid, fm)

    def add(self, log_id: str, fm: dict[str, Any]) -> None:
        self.by_log_id[log_id] = fm
        if fm.get("run_id"):
            self.by_run_id[str(fm["run_id"])] = log_id

    def resolve(self, ref: str) -> str | None:
        ref = strip_wiki(ref)
        if ref.startswith(UNRESOLVED_PREFIX):
            ref = ref[len(UNRESOLVED_PREFIX):]
        if ref in self.by_log_id:
            return ref
        return self.by_run_id.get(ref)


def _parent_refs(parents: list[str], index: FieldlogIndex) -> list[str]:
    out = []
    for p in parents:
        lid = index.resolve(p)
        out.append(as_wikilink(lid) if lid else f"{UNRESOLVED_PREFIX}{strip_wiki(p)}")
    return out


def _machine_frontmatter(log_id: str, b: dict[str, Any], index: FieldlogIndex) -> dict[str, Any]:
    f, code, launch = b["facets"], b["code"], b["launch"]
    return {
        "id": log_id,
        "type": "fieldlog",
        "run_id": b["run_id"],
        "status": b["status"],
        "tier": f["tier"],
        "created_at": b["created_at"],
        "finished_at": b["finished_at"],
        "task": f["task"],
        "arch_family": f["arch_family"],
        "arch_scale": f["arch_scale"],
        "seed": f["seed"],
        "dataset_train": b["data"]["train"],
        "dataset_eval": b["data"]["eval"],
        "interventions": b["interventions"],
        "intervention_note": b["intervention_note"],
        "parent_runs": _parent_refs(b["lineage"]["parents"], index),
        "motivated_by": [as_wikilink(c) for c in b["lineage"]["motivated_by"]],
        "compute": b["compute"],
        "git_commit": code.get("commit"),
        "dirty_tree": code.get("dirty"),
        "launch_cmd": launch.get("cmd"),
        "metrics": {k: round(v, 6) for k, v in b["metrics"].items()},
        "telemetry": {k: round(v, 6) for k, v in b["telemetry"].items()},
        "vs_parent": None,
        "artifacts": b["artifacts"],
        "bundle": f"{BUNDLE_DIR}/{log_id}.json",
    }


def _attach_vs_parent(root: Path, fm: dict[str, Any], index: FieldlogIndex) -> None:
    parents = fm.get("parent_runs") or []
    first_resolved = None
    for p in parents:
        s = str(p)
        if not s.startswith(UNRESOLVED_PREFIX):
            pid = strip_wiki(s)
            if pid in index.by_log_id:
                first_resolved = index.by_log_id[pid]
                break
    if first_resolved is None:
        fm["vs_parent"] = None
        return
    try:
        cfg = load_config(root)
    except Exception:
        cfg = None
    haze = load_haze(root)
    fm["vs_parent"] = compute_vs_parent(root, fm, first_resolved, haze=haze, config=cfg)


def merge_frontmatter(machine: dict[str, Any], old: dict[str, Any], seeds: dict[str, Any]) -> dict[str, Any]:
    """Machine keys from `machine`; human keys preserved from `old` (or seeded on creation)."""
    human = {k: v for k, v in old.items() if k not in MACHINE_KEYS and k != "route"}
    for k, v in seeds.items():
        human.setdefault(k, v)
    out: dict[str, Any] = {"id": machine["id"], "type": machine["type"]}
    for k in HUMAN_SEEDED_KEYS:
        if k in human:
            out[k] = human.pop(k)
    for k in MACHINE_KEYS:
        if k not in out:
            out[k] = machine.get(k)
    out.update(human)
    return out


def _new_body(b: dict[str, Any], parent_refs: list[str]) -> str:
    ref = parent_refs[0] if parent_refs else "baseline"
    notes = f"- Run notes: {b['notes']}\n" if b.get("notes") else "- Training dynamics / loss curve notes:\n"
    return (
        "## Key Takeaway (TL;DR)\n"
        f"- TODO: Summarize key outcome vs {ref}.\n\n"
        "## Observations & Anomalies\n"
        f"{notes}"
    )


def flag_cairns_stale(root: Path, cairn_refs: list[str], reason: str) -> list[str]:
    flagged: list[str] = []
    for ref in cairn_refs:
        cid = strip_wiki(ref)
        if not cid.startswith("crn_"):
            continue
        cpath = root / CAIRN_DIR / f"{cid}.md"
        if not cpath.exists():
            continue
        fm, body = parse_markdown_file(cpath)
        if not fm.get("needs_synthesis", False):
            fm["needs_synthesis"] = True
            write_markdown_file(cpath, fm, body)
            flagged.append(f"{cid} ({reason})")
    return flagged


def ingest_bundle(
    root: Path, bundle: dict[str, Any], index: FieldlogIndex | None = None, force: bool = False
) -> IngestResult:
    """Ingest one *validated* bundle (see bundle.validate_bundle)."""
    index = index or FieldlogIndex(root)
    run_id = bundle["run_id"]
    existing_id = index.by_run_id.get(run_id)
    log_id = existing_id or fieldlog_id_for(run_id, bundle["created_at"])
    card = root / FIELDLOG_DIR / f"{log_id}.md"
    archive = root / BUNDLE_DIR / f"{log_id}.json"

    old_fm: dict[str, Any] = {}
    body = ""
    if card.exists():
        old_fm, body = parse_markdown_file(card)
        if old_fm.get("run_id") not in (None, run_id):
            return IngestResult(
                "rejected", log_id, card,
                f"id collision: {log_id} already belongs to run_id {old_fm.get('run_id')!r}",
            )

    old_status = old_fm.get("status")
    old_bundle = json.loads(archive.read_text(encoding="utf-8")) if archive.exists() else None
    if old_bundle == bundle and card.exists() and not force:
        return IngestResult("unchanged", log_id, card)
    if old_status in TERMINAL_STATUSES and not force:
        return IngestResult(
            "rejected", log_id, card,
            f"{log_id} is {old_status}; its evidence is immutable. "
            "Re-run with --force only if the original bundle was wrong.",
        )

    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(canonical_json(bundle), encoding="utf-8")

    machine = _machine_frontmatter(log_id, bundle, index)
    fm = merge_frontmatter(
        machine, old_fm, {"title": bundle["title"], "validity": DEFAULT_VALIDITY}
    )
    _attach_vs_parent(root, fm, index)
    if not body.strip():
        body = _new_body(bundle, machine["parent_runs"])
    write_markdown_file(card, fm, body)
    index.add(log_id, fm)

    flagged: list[str] = []
    if bundle["status"] == "final" and old_status != "final":
        flagged = flag_cairns_stale(root, machine["motivated_by"], f"new evidence {log_id}")
    return IngestResult("updated" if old_fm else "created", log_id, card, flagged=flagged)


def relink_parents(root: Path, index: FieldlogIndex) -> list[str]:
    """Resolve `run:<id>` parent placeholders whose parents have since been ingested,
    and compute/refresh `vs_parent` diffs across fieldlogs.

    Parent links and `vs_parent` diffs are derived lineage (not raw run evidence),
    so updating them is allowed on terminal cards.
    """
    relinked: list[str] = []
    for log_id, fm in list(index.by_log_id.items()):
        parents = fm.get("parent_runs") or []
        changed_links = False
        new_parents = list(parents)
        if any(str(p).startswith(UNRESOLVED_PREFIX) for p in parents):
            new_parents = []
            for p in parents:
                lid = index.resolve(str(p)) if str(p).startswith(UNRESOLVED_PREFIX) else None
                new_parents.append(as_wikilink(lid) if lid else p)
            changed_links = new_parents != parents

        card = root / FIELDLOG_DIR / f"{log_id}.md"
        if not card.exists():
            continue
        cfm, body = parse_markdown_file(card)
        old_vs = cfm.get("vs_parent")
        had_route = "route" in cfm
        cfm.pop("route", None)
        if changed_links:
            cfm["parent_runs"] = new_parents
        _attach_vs_parent(root, cfm, index)
        if changed_links or had_route or cfm.get("vs_parent") != old_vs:
            write_markdown_file(card, cfm, body)
            index.add(log_id, cfm)
            if changed_links:
                relinked.append(log_id)
    return relinked


def iter_bundle_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("*.json")))
        elif p.exists():
            out.append(p)
    return out


def load_bundle_file(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        from .bundle import BundleError

        raise BundleError([f"not valid JSON: {exc}"], str(path)) from exc
    return validate_bundle(raw, source=str(path))
