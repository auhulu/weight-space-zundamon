# Condition-blind response review

Read only the assigned `runs/blind/packet_N.json`. Do not inspect experiment logs, source conditions, model metadata, candidate files, mapping or other reviewer outputs.

For every response independently:

- relevance: 0 = unrelated, non-answer, or substantively contradicts what was asked; 1 = partly answers but key information is absent or confused; 2 = directly addresses the question.
- naturalness: 0 = broken or incoherent Japanese / debilitating repetition; 1 = awkward, redundant, or inconsistent Japanese; 2 = natural Japanese. The target ending のだ/なのだ is not itself an error.
- note: short concrete reason, particularly for scores below 2. Identify apparent factual errors in notes, but do not claim exhaustive fact checking.

Return a JSON list of `{review_id, relevance, naturalness, note}` to `runs/blind/reviews_N.json`. Every review_id must appear exactly once. Do not use rules, keyword matching or a secondary model to substitute for reading the actual answers. Do not infer the hidden condition. No edits to the responses or the rubric.

This is a single-pass agent judgment, not a human benchmark. Even with condition labels removed, phrasing may reveal the intervention. Aggregate ratings are descriptive; disagreements and factual uncertainty remain possible.
