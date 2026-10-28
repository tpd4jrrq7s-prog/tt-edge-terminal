"""Extract typed TargetVectors from TrainingExamples.

Every target is derived strictly from the target match's own recorded
outcome (`TrainingExample.player_a_won` / `.best_of`) — never from
anything inside the paired `FeatureSnapshot`, which was built to
exclude the target match entirely.

Future extension (not implemented here): set-level and point-level
prediction targets, which would require point/set progression data
beyond what `TrainingExample` currently carries.
"""

from __future__ import annotations

from datasets.models import TrainingExample
from ml.errors import MissingTargetValueError
from ml.models import TargetKind, TargetVector

TARGET_SCHEMA_VERSION = "1.0.0"


def extract_targets(examples: list[TrainingExample], kind: TargetKind) -> TargetVector:
    """Build a TargetVector for `kind`, positionally aligned with `examples` (and any FeatureMatrix built from them)."""
    if kind is TargetKind.PLAYER_A_WON:
        values = [float(e.player_a_won) for e in examples]
    elif kind is TargetKind.PLAYER_B_WON:
        values = [float(1 - e.player_a_won) for e in examples]
    elif kind is TargetKind.BEST_OF:
        missing = [e.id for e in examples if e.best_of is None]
        if missing:
            raise MissingTargetValueError(
                f"best_of is unknown for {len(missing)} example(s), e.g. {missing[:3]!r}; "
                "cannot build a best_of target vector without silently fabricating values"
            )
        values = [float(e.best_of) for e in examples]  # type: ignore[arg-type]
    else:  # pragma: no cover - defensive; TargetKind is a closed enum
        raise ValueError(f"Unsupported target kind: {kind!r}")

    return TargetVector(
        kind=kind, schema_version=TARGET_SCHEMA_VERSION, values=values, source_ids=[e.id for e in examples]
    )
