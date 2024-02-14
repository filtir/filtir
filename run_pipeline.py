from step1_api_claim_extractor import ClaimExtractor
from step2_api_fix_passage_anchors import FixAnchors
from step3_api_identify_objective_claims import ClassifyClaims
from step41_api_fetch_cohere_wikipedia_evidence import CohereEvidence
from step42_api_fetch_google_search_evidence import GoogleEvidence
from step5_api_embed_search_results import EmbedResults
from step6_api_claims_to_evidence import ClaimToEvidence
from step7_api_check_claims_against_evidence import CheckClaimAgainstEvidence
from step8_api_format_fact_checked_document import FormatDocument

import argparse
from pipeline_paths import PIPELINE_PATHS
import json
from pathlib import Path
import os
import copy
from dotenv import load_dotenv

load_dotenv()


def get_fact_checked(text_input, model="gpt-3.5-turbo", mode="slow"):
    text_input = text_input.strip()

    results = {}

    # STEP1
    print("Step1: Extracting claims")
    step1 = ClaimExtractor(model=model)
    step1_json = step1.extract_claims(text_input)
    results["step1_claims"] = copy.deepcopy(step1_json)

    # STEP2
    print("Step2: Anchoring claims")
    try:
        step2 = FixAnchors(model=model)
        step2_json = step2.fix_passage_anchors(step1_json, text_input)
    except:
        if model != "gpt-4":
            print("Step2 failed with gpt-3.5, trying with gpt-4!")
            step2 = FixAnchors(model="gpt-4")
            step2_json = step2.fix_passage_anchors(step1_json, text_input)
    results["step2_anchored_claims"] = copy.deepcopy(step2_json)

    # STEP3
    print("Step3: Classifying claims")
    step3 = ClassifyClaims(model=model)
    step3_json = step3.classify_claims(step2_json)
    step3_filter = step3.filter_to_objective_claims(step3_json)
    results["step3_classify_claims"] = copy.deepcopy(step3_json)
    results["step3_objective_claims"] = copy.deepcopy(step3_filter)

    if len(step3_filter) == 0:
        return {"fact_checked_md": "No objective claims found!"}

    # STEP4.1
    print("Step4.1: Gathering evidence")
    step4_cohere = CohereEvidence()
    step4_json_cohere = (
        step4_cohere.fetch_cohere_semantic_search_results_to_gather_evidence(
            step3_filter
        )
    )
    results["step41_cohere_evidence"] = copy.deepcopy(step4_json_cohere)

    # STEP4.2
    print("Step4.2: Gathering evidence")
    step4_json_google = None
    if mode == "slow":
        step4_json_google = ""
        try:
            step4_google = GoogleEvidence(model=model)
            step4_json_google = step4_google.fetch_search_results_to_gather_evidence(
                step3_filter
            )
        except Exception as e:
            print(f"Google search failed: {e}")
            pass
        results["step42_google_evidence"] = copy.deepcopy(step4_json_google)

    embedding_model = "text-embedding-ada-002"
    text_embedding_chunk_size = 500

    srcs = [step4_json_cohere]
    if step4_json_google:
        srcs.append(step4_json_google)

    # STEP 5
    print("Step5: Embedding evidence")
    step5 = EmbedResults(
        embedding_model=embedding_model,
        text_embedding_chunk_size=text_embedding_chunk_size,
    )
    faiss_db = step5.embed_for_uuid(srcs)

    # STEP 6
    print("Step6: Linking claims to evidence")
    step6 = ClaimToEvidence()
    step6_json = step6.link_claims_to_evidence(step3_filter, faiss_db)
    results["step6_claim_to_evidence"] = copy.deepcopy(step6_json)

    # STEP 7
    print("Step7: Checking claims against evidence")
    step7 = CheckClaimAgainstEvidence(model=model)
    step7_json = step7.check_claims_against_evidence(step6_json)
    results["step7_evaluated_claims"] = copy.deepcopy(step7_json)

    # STEP 8
    print("Step8: Formatting")
    if mode == "slow":
        step8 = FormatDocument(model=model, footnote_style="verbose")
        step8_md = step8.reformat_document_to_include_claims(
            text_input, step7_json, footnote_style="verbose"
        )
        step8_md_terse = step8.reformat_document_to_include_claims(
            text_input, step7_json, footnote_style="terse"
        )

        results["fact_checked_md"] = copy.deepcopy(step8_md)
        results["fact_checked_terse"] = copy.deepcopy(step8_md_terse)
    return results


def main(args):
    with open(args.file, "r") as f:
        text = f.read()
    out = get_fact_checked(text, mode="slow", model=args.model)
    print(out["fact_checked_md"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a file.")
    parser.add_argument("--file", type=str, help="File to process", required=True)
    parser.add_argument("--model", type=str, help="Model to use", required=True)
    args = parser.parse_args()
    main(args)
