# One Frame Is Enough — benchmark & pipeline repo

Research project (AI master's, Edge Computing Applications course), intended to become a workshop paper.
Full title: "One Frame Is Enough: Miss-Aware Semantic Obfuscation with Adaptive Synthetic Identity Replacement on Edge Devices".

## Core idea
If a person's face is exposed in even ONE frame, their identity is compromised. Frame-level detection
recall hides this: 95%+ frame recall can still leak many identities over long tracks. We measure
**identity-level exposure** and reduce it with miss-aware recovery (tracking + pose fallback) and an
adaptive anonymization policy, running on an NXP i.MX 8M Plus NPU.

## Team / ownership
- Valerian: computer vision + privacy — detection/tracking, missed-face recovery, anonymization, privacy evaluation (this benchmark).
- Moro: edge AI + integration — NXP deployment, pipeline, adaptive logic, performance testing.

## Research questions
- RQ1: How does frame-level recall translate into identity-level exposure as videos get longer and conditions degrade?
- RQ2: How much does temporal + pose-based recovery of missed faces reduce exposure, and at what masking cost?
- RQ3: Does confidence-based choice between synthetic replacement and conservative masking beat any fixed method on privacy–utility?
- RQ4: How do INT8, detection interval N and CPU/NPU split shift the trade-off; which config is real-time on i.MX 8M Plus?

## Benchmark (current phase: baselines first, answers RQ1)

### Datasets
| Tier | Dataset | Use |
|---|---|---|
| 1 main | ChokePoint (Zenodo record 815657) + PLUSFiaQ (FaceQSORT, arXiv 2501.11741; also ships box annotations for 6 ChokePoint sequences) | Multi-face surveillance; enroll identities from OTHER sequences of the same person |
| 2 hard | Music Video dataset (Zhang et al., 8 videos, ~3.8k tracklets, ~118k face boxes) | Shot changes, fast motion; enroll from held-out frames |
| 3 stress | Tier 1 + synthetic degradations (low light, motion blur, downscale, occlusion) at several severities | Exposure vs severity curve |
| optional | FANVID (low-res faces, HR mugshot gallery) | Check license first |

Do NOT use YouTube Faces as ground truth (single face, detector-generated boxes).
Start with 4–6 ChokePoint sequences + PLUSFiaQ; add tiers once the harness works.

### Systems under test (per-frame, same blur, same mask scale 1.3)
| System | Role | Notes |
|---|---|---|
| deface (CenterFace) | Common real-world tool | `pip install deface`; per-frame |
| EgoBlur face (Gen1) | Strong/heavy per-frame ceiling | Apache 2.0, Faster R-CNN ~104M params, desktop GPU only |
| YuNet FP32 + INT8 (OpenCV Zoo) | Our edge detector | MIT; detects faces ~10–300 px → downscale close-ups so large faces aren't false misses |
| SCRFD-2.5G-KPS | Second light detector | InsightFace, non-commercial only |
| YuNet + ByteTrack | Tracking baseline (B2) | Fairest comparison for our recovery |
| YOLOX + RTMPose (rtmlib) | Pose fallback baseline | Face box from nose/eye/ear keypoints |
| DeepPrivacy2 | Synthetic replacement reference | Desktop only, utility/quality comparison |

### Leak judge (attacker) — separate from all detectors
- ArcFace R50, InsightFace `buffalo_l`. Optional second judge: AdaFace (low-quality faces).
- Threshold = similarity at 1% FAR on impostor pairs from the enrollment gallery.

### Scoring rules
1. A GT face is protected if the anonymization mask covers >= 80% of its GT box (coverage, NOT IoU).
2. For any GT face not fully covered: crop it from the ANONYMIZED frame (blur applied as-is), embed with the judge,
   compare to that identity's gallery. Above threshold = leaked frame.
3. Identity leaked = at least one leaked frame.

### Metrics (per system, per dataset)
- Frame recall; % identities leaked; max consecutive exposed run (frames); re-ID attack success
- Fraction of frame area masked (utility cost)
- FPS, p50/p95 latency (desktop now, board later — compare edge to edge)

## Fair-comparison rules
- Same detector across our pipeline and baselines where the comparison is about recovery/policy.
- Same videos, same degradations, same judge, same mask scale for every system.
- Re-implementations of published ideas are labelled "our re-implementation of X", never as their system.

## Conventions
- Unified GT format (MOT-style CSV): `frame, track_id, identity, x, y, w, h` (pixels, top-left origin).
- System outputs use the same CSV minus identity, plus `score`.
- Python 3.10+, `opencv-python`, `onnxruntime`, `insightface`, `rtmlib`, `deface`, `numpy`, `pandas`.
- Every experiment run writes a config + results JSON so tables/plots are reproducible.

## Suggested layout
```
data/            # raw downloads (gitignored)
gt/              # converted GT CSVs
src/loaders/     # chokepoint.py, plusfiaq.py, musicvideo.py -> unified CSV
src/systems/     # deface_runner.py, yunet.py, scrfd.py, egoblur.py, bytetrack_wrap.py
src/judge/       # arcface.py, enrollment.py, threshold.py (1% FAR)
src/metrics/     # coverage matcher, identity exposure, max run
src/degrade/     # low light, blur, downscale, occlusion
experiments/     # configs + results
```

## Next tasks
Phases, gates and the ChokePoint train/val/test split: docs/plan.md.

## Decisions
- Real-time target: 25 FPS on the i.MX 8M Plus, i.e. a 40 ms budget per frame for the WHOLE pipeline
  (decode + detect + track/recovery + blur), not the detector alone.
- Edge detector: YuNet (fine-tuned), not YOLOv8n-face 320; YOLO stays as the "big detector" reference.
  Main question: can a small fine-tuned YuNet + pipeline match a big YOLO? Reasons: docs/decisions/why-not-yolo320.md.

## Open decisions (settle with Moro)
- RQ3 scope if synthetic replacement is too heavy for the board (blur vs face bank vs masking).

## Licensing / ethics
- InsightFace models (SCRFD, ArcFace) are non-commercial research only — fine for the paper, keep YuNet on the board.
- Own recordings need written consent from everyone filmed; state it in the paper.