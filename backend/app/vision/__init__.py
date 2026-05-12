"""Vision engine package for Digitalisierung ABE."""

from app.vision.vlm_loader import MockVLMManager, VLMManager
from app.vision.quality_gate import QualityAssessment, QualityGate, RefusalResult
from app.vision.dual_pass import DualPassExtractor, DualPassResult
from app.vision.provenance import ProvenanceTracker
from app.vision.surya_extractor import (
    SuryaDocumentExtractor,
    SuryaExtraction,
    SuryaLayoutBlock,
    SuryaTextLine,
)

__all__ = [
    "VLMManager",
    "MockVLMManager",
    "QualityGate",
    "QualityAssessment",
    "RefusalResult",
    "DualPassExtractor",
    "DualPassResult",
    "ProvenanceTracker",
    "SuryaDocumentExtractor",
    "SuryaExtraction",
    "SuryaLayoutBlock",
    "SuryaTextLine",
]
