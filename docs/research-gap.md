# Research gap (2026-11-18)

Literature check for "Privacy-Preserving Edge Video Analytics with Semantic Obfuscation", and the gap we propose
to claim. Proposal, to settle with Valerian.

## Closest prior work

| Work | What it does | What it does not do |
|---|---|---|
| **MBAM**, Khalili et al., IROS 2026 ([repo](https://github.com/ssysarch/MBAM), [pdf](https://ssysarch.ee.ucla.edu/files/26_MultiframeMasking_IROS.pdf)) | Temporal box fusion over a short window + hybrid heavy/light detector scheduling + frame decorrelation. Motivation is ours: one clear frame exposes identity. **ChokePoint** (Face-S1, Face-S2) + CARLA, Jetson Orin Nano, 31 ms/frame, recall 0.97 on Face-S2. | Privacy = detection recall only. No face-recognition attacker, no identity-level metric. Fixed schedule, not risk-driven. Code not released. |
| Multi-view OR anonymization, Chen et al., arXiv 2602.02850 | ByteTrack (fwd+bwd) + multi-view association recover low-score detections, pseudo-label self-training. | Detection recall only, no recognition attacker, desktop GPU, fixed multi-camera rooms. |
| LA3D, arXiv 2410.18717 | Blur strength adapted to person size/distance, anomaly-detection utility. | Strength set per frame, not by accumulated risk. |
| FaceLinkGen, arXiv 2602.02914 | Distillation attack breaks 7 privacy-preserving / de-ID methods (Recall@1 48-90%). | Still images only, no aggregation over frames or tracks, blur/pixelation not tested. |
| anonymized-not-anonymous ([repo](https://github.com/safwanx/anonymized-not-anonymous)) | After face blur, body re-ID stays ~85% on NTU RGB+D (unreviewed). | Leak through body, not through imperfect face masks. |
| Pedestrian-analysis privacy framework, WACV 2023 (likely the "PAPA" reading) | Replaces pedestrians with wireframes; privacy-utility on attributes. | Offline, no video exposure metric, no edge. |
| Revelio (deblurring attack) | Diffusion restoration of Gaussian-blurred faces, 95.9% re-ID on CelebA-HQ. | Single image. |
| Conformal risk control (e.g. arXiv 2304.06052) | Thresholds with guarantees on expected miss rate. | Never applied to anonymization or identity exposure. |

**Consequence for us:** "tracking/temporal recovery of missed faces on the edge" is no longer new on its own:
MBAM does it on ChokePoint, on an edge board, with the same motivation. Our contribution has to be what MBAM
and the others do not measure or control: **identity**, not boxes.

## The gap

Every edge anonymization system we found reports privacy as detection recall, a per-frame quantity. Nobody
measures what an attacker who holds the whole anonymized video can do, and nobody uses that risk to decide where
an edge device spends its compute.

Three parts, each checkable with our harness:

1. **Set-level attacker (evaluation).** An attacker that aggregates ArcFace embeddings over a whole track of
   partially covered or blurred crops (mean/quality-weighted template), against our current per-frame judge.
   Hypothesis: frames that are individually below threshold add up, so identity exposure is higher than our
   "one frame" judge says, and recall ranks systems differently from identity exposure. Not published as far as we
   found (FaceLinkGen and Revelio are single-image).
2. **Track-level exposure budget (method).** Each track keeps an on-device estimate of accumulated leak risk,
   computed without the judge (uncovered fraction of the predicted face box, face size, detector score, frames since
   last confident detection). When a track's budget runs low, the device escalates: larger mask, coasting, pose
   fallback, or an extra full-resolution detector pass. Compute goes where identity risk is, unlike MBAM's fixed
   heavy/light schedule. Cheap on the CPU, fits the 6-7 ms left in the 40 ms budget.
3. **Guarantee (calibration).** Calibrate the detector threshold and the budget with conformal risk control on val,
   so that P(identity exposed) <= alpha on exchangeable data. Report where it breaks on the stress tier.

Working title: **"Recall Is Not Privacy: Track-Level Exposure Budgets for Identity-Safe Video Anonymization on an
Edge NPU"** (or keep "One Frame Is Enough" as the main title).

## Why it is worth it

- One sentence of novelty: the first edge anonymization pipeline evaluated and controlled at the level of identities
  under a set-level recognition attacker.
- Direct comparison with the strongest recent baseline on the same dataset: our re-implementation of MBAM's
  temporal fusion on ChokePoint, scored with our identity metric. If MBAM's 0.97 recall still exposes people, that
  is the paper's headline.
- Builds on what exists: ArcFace judge, coverage matcher, cached board detections, fast_blur, NPU pipeline.

## What to run first

1. Set-level attacker in `src/judge/` and re-score the cached detections (`experiments/d1`, `experiments/board`).
   Gate: does it expose more identities than the per-frame judge on any system? If not, part 1 weakens and the
   budget becomes the main contribution.
2. Stress tier (Phase 6) so detectors actually miss.
3. Re-implementation of MBAM fusion + ByteTrack baseline on cached detections (Phase 7).
4. Exposure budget, then conformal calibration.

Semantic obfuscation beyond the face (silhouette/skeleton, WACV 2023 style) stays optional, measured with
people-counting utility like MBAM, if there is time after 1-4.
