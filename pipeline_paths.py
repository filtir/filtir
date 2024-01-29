"""A module that contains the paths to the various files and folders used in the pipeline."""
from pathlib import Path

PIPELINE_PATHS = {
    # google queries
    "google_custom_search_engine_id_path": "google_custom_search_engine_id.txt",
    # raw inputs to pipeline
    "source_document_dir": "data/source_documents",
    # claim extraction
    "extracted_claims_dir": "data/extracted_claims",
    "extracted_claims_with_anchor_fixes_dir": "data/extracted_claims_with_anchor_fixes",
    "extracted_claims_with_classifications_dir": "data/extracted_with_classifications_claims",
    "objective_claims_dir": "data/objective_claims",
    # evidence gathering
    "cohere_wikipedia_evidence": "data/evidence_gathering/cohere_wikipedia",
    "google_search_results_evidence": "data/evidence_gathering/google_search_results",
    "faiss_db_embeddings_for_evidence": "data/faiss_db_embeddings_for_evidence",
    "web_evidence_chunks": "data/evidence_gathering/web_evidence_chunks",
    # claim evaluation
    "evaluated_claims_dir": "data/claim_evaluation/claim_verdicts",
    # reformatted document
    "fact_checked_document_dir": "data/fact_checked_documents",
}

PIPELINE_PATHS = {key: Path(value) for key, value in PIPELINE_PATHS.items()}
