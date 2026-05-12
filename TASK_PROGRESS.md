# Upload Flow Fix - Task Progress

- [x] Fix Issue 2: File picker does nothing (onchange handler unreliable via innerHTML)
- [x] Fix Issue 1: Network error on XHR from file:// origin (serve dashboard from backend)
- [x] Update start scripts to open dashboard from backend URL
- [x] Verify both fixes work end-to-end

## Additional Fixes Applied

### Frontend (dashboard.html)
- XSS protection via HTML escaping (`esc()` helper) on all user-controlled render paths
- Fixed `window.history` shadowing by renaming to `extractionHistory`
- Added client-side file type validation and duplicate detection
- Fixed drag-leave flicker on dropzone
- Fixed blob URL memory leaks
- Fixed upload progress phase label stuck on "Quality Gate"
- Fixed HTML attribute injection in correction dialog

### Backend
- Fixed file size limit inconsistency (200 MB → 50 MB)
- Fixed upload filename collision (microsecond-precision timestamp)
- Fixed ProvenanceTracker mixed coordinate normalization bug
- Fixed ProvenanceTracker PIL highlight overlay not compositing
- Fixed steuerklasse regex accepting invalid Roman numerals (e.g. "VII")
- Fixed validate_steuer_id test data (invalid double-duplicate ID)
- Fixed QualityGate OCR performance on full-resolution images
- Fixed CorrectionCapture hardcoded `doc_type = "Unbekannt"`
- Fixed CorrectionCapture token over-estimation (~4x too conservative)
- Fixed EmbeddingModel `is_loaded` returning False in stub mode
- Fixed `.env` discovery when server starts from `backend/` directory
- Added missing `use_mocks`, `audit_log_path`, `chroma_persist_dir` to ExtractionPipeline
- Added missing `dual_pass`, `validator`, `prompt_builder`, `ram` properties to ExtractionPipeline
- Added `get_history()` and `rotate_logs()` to AuditLogger
- Swapped VLMManager to the Surya OCR adapter and kept file-not-found checks before inference

### Test Fixes
- Fixed all 7 validator test failures
- Fixed all 4 vision test failures
- Fixed all 20 pipeline test failures
- Fixed all API tests (31/31 pass)
- All LLM tests pass (45/45)
- **Total: 265 passed** across test_api, test_validators, test_vision, test_pipeline, test_llm
