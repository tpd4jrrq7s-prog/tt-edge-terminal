"""Phase 3 dataset layer: converts leakage-safe feature snapshots into
labeled TrainingExamples, chronological splits, and deterministic exports.

Random train/test splitting is not implemented — every split strategy
is chronological. See `datasets.builder.DatasetBuilder`,
`datasets.splits`, and `datasets.leakage` for the main entrypoints.
"""
