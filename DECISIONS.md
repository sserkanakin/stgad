# Design Decisions

Log of non-obvious design choices and their rationale. Append entries as work progresses.
Format: `## YYYY-MM-DD — <short title>` followed by the decision and reason.

---

## External code pins

| Component | Repo | Commit hash |
|-----------|------|-------------|
| STID      | github.com/GestaltCogTeam/STID | TBD (record after Phase 2 clone) |

---

## 2026-04-22 — Project root is repo root

**Decision**: scaffold directly in the repo root (`research-tud/`) rather than a `thesis_diffusion/` subdirectory.

**Reason**: avoids an extra nesting level with no benefit; project context files moved to `docs/`.

---

## Template entry

**Decision**: <what was chosen>.

**Alternatives considered**: <what else was considered>.

**Reason**: <why this choice>.

**Spec reference**: §<N> of PROJECT_SPEC.md.
