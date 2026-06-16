"""
Synthetic German-document generation.

Produces ``{image, fields}`` pairs for the four supported document types with
checksum-valid faked data, used to fine-tune a domain-specialised extraction
model. Training is on synthetic data; evaluation is on the *real* held-out
``test_documents/`` (never trained on).
"""
