# Independent edit design

The editor inspected the locally installed Transformers Qwen3.5 implementation and only `runs/train_stats.pt` from the experimental data. It did not inspect train generations, validation data/results, side evaluations, test questions, or final generations. Four hypotheses were implemented before inspecting the activation statistics. Layer choices below were then determined mechanically by those hypotheses, without evaluation feedback.

## Architecture and measurements

`Qwen3_5MLP` computes `down_proj(act(gate_proj(x)) * up_proj(x))`. Its weight has shape hidden size by intermediate size. Thus a down-projection input hook measures the correct post-gating key vector. The contrast direction is the mean paired style-minus-normal decoder-block output at the last prompt token. The normal mean post-gating vector is the key. All statistics come from the specified 24 TRAIN prompt pairs.

Let h be the normal mean key, d the mean residual contrast, D the root mean squared contrast-vector norm, and R the root mean squared normal residual-vector norm. Layer ranking uses `(norm(d)/D) * (norm(d)/R)`, with coherence bounded at one for floating-point robustness. This favors contrasts which are consistent across prompts and large relative to the residual stream. The first quarter of layers is excluded; instruction and position effects are particularly plausible there.

## Fixed hypotheses and selected layers

| Candidate | Selected zero-based layers | Key | Train ranking score | Contrast coherence |
| --- | --- | --- | --- | --- |
| middle_mean | 12 | Mean key | 0.439981 | 0.966379 |
| late_mean | 19 | Mean key | 0.166791 | 0.782397 |
| distributed_mean | 12, 11, 13 | Mean key, one-third budget each | 0.439981, 0.407718, 0.362205 | 0.966379, 0.974698, 0.936503 |
| diagonal_key | 12 | Diagonal second-moment preconditioned key | 0.439981 | 0.966379 |

For a mean key, the update is `delta_W = scale * outer(d, h) / dot(h, h)`. For the diagonal hypothesis, replace the right-hand h by `v = h / clamp(mean(h_sample**2), min=0.1*mean_over_coordinates(mean(h_sample**2)))`, and use `dot(h, v)` in the denominator. This retains the exact mean-key target while reducing the contribution of coordinates with large second moments. A small numerical floor is used only to prevent division by zero.

The scale is `min(1, 0.10*R/norm(d)) * share`, where share is one for single-layer candidates and one-third for the distributed candidate. The common 10% cap is a fixed conservative design choice, not a tuned strength. The distributed hypothesis divides this normalized budget instead of applying a full contrast at all three layers. Every candidate starts from the same base weights. There are no gradients, optimizers, losses, fitting iterations, or strength sweeps.

## Interpretation and limitations

The paired residual difference is not a pure style direction: it also contains the content, token positions, and contextual effects of the added instruction. High coherence does not establish causal specificity. The update is exact at the normal *mean* prompt key in real arithmetic; it is not an exact shift at every individual key. During generation its effect is multiplied by each token's projection onto the selected key, and intervening nonlinear layers may alter the direction substantially. Last-prompt-token measurements do not establish that the same intervention works at sentence endings. The fixed cap may make an intervention too weak; failure would not justify adapting it after inspecting held-out outputs under this protocol.

Only MLP down-projection weights are edited. Updates are formed in float32 and stored in the model's existing dtype. Metadata counts actual changed stored elements after rounding and reports realized relative Frobenius norms. The model dtype can erase some small requested updates. The design does not claim preservation of unrelated capabilities; that is an evaluation question.

## Verification

`python -m unittest test_edit_design -v` passes two algebraic tests covering the intended mean-key action of all four candidates, unchanged unrelated parameters, JSON-serializable specifications/metadata, and exact counting of changed float16 weight elements. No model output scores are involved in those tests.
