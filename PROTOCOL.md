# Preregistered local experiment

Frozen before candidate generation and final evaluation (2026-11-05, Asia/Tokyo).

## Question
Can a direct weight edit make Qwen3.5-0.8B use Japanese endings のだ/なのだ without a style instruction, while retaining useful answers? This tests style elicitation, not acquisition of knowledge or a complete character persona.

## Conditions and data
`config.json` fixes model revision, ordinary system message, style instruction, sampling settings and seeds. `data/` fixes train24, validation24, test48 and instruction-following side16. No test or side data is supplied to the editor. Primary generations use thinking disabled, max256 tokens, temperature0.7/top_p0.9/top_k20. Validation uses seed42; final uses 101/202/303. Evaluation questions never become editing feedback.

Compare base; base with style prompt; edited with only ordinary prompt. Side tests use the same system messages as their respective condition; explicit task instructions are in user messages. A style system message may conflict with user formatting requests: report this limitation rather than claiming a neutral prompt baseline on those cases.

## Editing and selection
Use paired last-prompt-token residual activations and MLP down-projection inputs from train only. A separate editor proposes at most six reasoned rank-one weight edits; no gradients, optimizer, parameter sweep, LoRA, runtime steering or output rewriting. Candidate construction uses FP32 and casts the resulting weights to FP16. Candidate code and specs are retained before validation. Editing activation hooks are removed before generation.

Select among candidates whose counts of empty answers and repetitive answers do not exceed base validation counts. Maximize sentence-final target ending fraction, breaking ties by actual changed FP16 element count. Style-only outputs are included in the degenerate-answer count together with empty answers. If no eligible candidate exists, explicitly report failure and evaluate the candidate with highest target fraction as a diagnostic, without presenting it as passing selection. Nonempty one-word punctuation-free answers count as one sentence. Quotations and fenced code are excluded from target-ending counts; exclusions do not exempt answers from quality review.

## Final assessment
Report sentence-final target fraction, within-answer appearance, degenerate/repetitive flags, token counts and cap hits. Independent quality review uses condition-blind randomized response IDs with question text and rubric: relevance 0=unrelated/non-answer, 1=partially answers, 2=directly answers; naturalness 0=broken, 1=awkward, 2=natural. Intended のだ endings alone do not count as unnatural. Automated agent judgments are not human ratings or factuality guarantees; styling may reveal the condition despite removed labels. Use paired bootstrap over question IDs (retain all seed responses per question) for uncertainty of edited-minus-base target fraction. Side tests are descriptive and excluded from selection.

## Validation and limits
Check finite tensors, actual changed keys/elements, unchanged tensor hashes, exact FP16 restoration between candidates, exact saved/reloaded tensor hashes and deterministic greedy output equality on a train question. No claim of exact cross-hardware stochastic reproducibility. Single model, small hand-authored Japanese dataset, one edit-design run. Report failed edits, time and restrictions. No public posting or remote push.
