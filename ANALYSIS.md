# Digitalisierung — Technical Analysis (Baseline Audit)

> Baseline audit captured before the `feature/wire-pipeline-and-harden` work.
> Score at time of writing: **4 / 10**. Target ceiling after remediation: **~8.5 / 10**.

## 1. Purpose, packages & modules

A **fully local, offline** system to digitise four German bureaucratic document types —
`Meldebescheinigung`, `Steuerbescheid`, `Gehaltsausweis`, `Personalausweis` — into structured,
per-field data with confidence scores, provenance bounding boxes, an audit trail, and a
human-correction learning loop. FastAPI backend + Next.js dashboard. Surya OCR + (intended)
VLM/LLM, ChromaDB for corrections + image-learning. Designed for a 16 GB Apple-Silicon Mac.

Key packages: `fastapi`, `pydantic` v2, `surya-ocr`, `transformers`, `pytesseract`, `pillow<11`,
`numpy>=2.1`, `llama-cpp-python` (darwin-only), `instructor`, `chromadb`, `sentence-transformers`,
`psutil`, `pypdf`. **Imported but undeclared:** `cv2`/opencv, `scipy` (NumPy fallbacks exist).

## 2. The process — claimed vs. actual (the keystone finding)

The docstrings, the `ExtractionPhase` enum, and the audit log describe a 7-stage pipeline.
The executed path is much shorter.

| Stage | Claimed | Actual |
|---|---|---|
| 1. Quality Gate | Refuse inadmissible images | Runs, but **very lax**; hard-refuses only below 0.20 legibility |
| 2. Dual-Pass VLM | Two independent VLM passes, cross-validated | **One** Surya OCR call; both maps from same output (tautological) |
| 3. Few-shot corrections | Retrieve corrections from ChromaDB | Retrieves image-learning hints only; **corrections never retrieved** |
| 4. LLM structuring | Schema-forced via Instructor + llama.cpp | **Never invoked** (`llm_model="dual-pass-only"`) |
| 5. Symbolic validation | Regex + checksum + business rules | **Never invoked** in `process()` |
| 6. Audit logging | Record completed phases | Logs phases 2–5 as "completed" though they didn't run |
| 7. Return | Structured, schema-valid fields | Generic OCR lines (`line_0001`, colon-splits) |

## 3. Genuine engineering (real strengths)

- `surya_extractor.py`: real Surya loading, pytesseract fallback, **MPS attention bugfix**.
- Quality-gate CV heuristics are real (with NumPy fallbacks).
- IBAN mod-97 and PLZ validators are correct.
- Real, XSS-clean ~1,380-line Next.js dashboard.
- Real security-header/CSP middleware; path-traversal handled; temp cleanup in `finally`.

## 4. Accuracy & false positives

- **No semantic field mapping** — strict Pydantic schemas referenced only by dead LLM layer + tests.
  Output is OCR line-splits, not schema fields. Surya layout blocks computed then discarded.
- **Dual-pass cross-validation is theater** with Surya (identical key sets → inflated consistency).
- **Quality gate barely gates**: Michelson contrast saturates to ~1.0; Laplacian variance treats
  noise as sharpness; admission ≈ `sharpness ≥ 0.25`. Orientation detected but **never applied**.
- **Confidence is not calibrated to correctness** (`OCR_conf × legibility`).
- **Validators unwired and partly wrong**: Steuer-ID uses a prime-weighted scheme, not ISO 7064
  Mod 11,10; `regex_patterns` use `\\s` (literal backslash+s, not whitespace) so multi-word values
  fail; street validator `\w` accepts digits (`"12345"` passes).
- **Tests prove plumbing, not accuracy**: 329 test fns, mock-dominated; none run real Surya/llama.cpp;
  none load the 4 real `test_documents/` images; only benchmark is 8 English SROIE receipts.

## 5. Improvement & hardening (the remediation backlog)

Correctness: wire the documented pipeline; map OCR→schema using layout blocks; close the correction
loop; fix the validators; apply orientation; tune the quality gate.
Security: add auth + bind `127.0.0.1`; magic-byte upload validation + fail-closed size; DOCX
zip-bomb cap; tighten CORS; stop mounting `/uploads` publicly; untrack `.env`; pin frontend deps;
verify model-download checksums.
Reliability: wire audit-log rotation; persist the extraction store.
Evidence: build a German-labelled eval set with CER/WER and wire `test_documents/`.

## 6. Score: 4 / 10

| Dimension | Score |
|---|---|
| Component engineering | 6.5 |
| Functional completeness vs. spec | 3 |
| Accuracy & validation | 2 |
| Honesty of docs/naming vs. reality | 3 |
| Security posture | 4 |
| Testing quality | 3.5 |
| **Overall** | **≈ 4** |

## 7. Potential

Most of the hard scaffolding already exists, so the gap is fixable. Wiring layout-aware OCR→schema
mapping + the existing validator/correction code into the live path, fixing the bug-level defects,
a focused security pass, and a real German eval set with CER/WER could realistically reach
**7.5–8.5 / 10** — a genuinely strong local, privacy-first extractor. The decisive lift is making the
documented pipeline actually execute end-to-end and **proving accuracy on real dirty German documents**.

---
*See `REMEDIATION.md` for the change log of fixes applied on `feature/wire-pipeline-and-harden`.*
