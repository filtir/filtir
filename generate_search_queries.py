import argparse
from pathlib import Path
import numpy as np
from pipeline_paths import PIPELINE_PATHS
import json
from zsvision.zs_utils import BlockTimer
from typing import Dict, List
from llm_api_utils import (
    call_openai_with_exponetial_backoff,
    estimate_cost_of_text_generation_api_call,
    init_openai_with_api_key,
)


def generate_search_queries(args, src_path: Path, dest_path: Path):
    """
    Generate a search query that can be used to verify a claim.
    """
    init_openai_with_api_key(api_key_path=args.api_key_path)
    with open(src_path, "r") as f:
        claims_and_sources = json.load(f)

    # exclude subjective claims
    original_num_claims = len(claims_and_sources)
    claims_and_sources = [
        claim_and_source
        for claim_and_source in claims_and_sources
        if claim_and_source["label"] == "objective"
    ]
    num_claims = len(claims_and_sources)
    print(
        f"Filtered from {original_num_claims} claims to {num_claims} objective claims"
    )

    # we limit the number of claims per api call (otherwise GPT-4 can choke)
    num_batches = int(np.ceil(num_claims / args.max_claims_per_api_call))
    claims_and_sources_batches = [
        batch.tolist() for batch in np.array_split(claims_and_sources, num_batches)
    ]
    queries = []

    all_claims_str = "\n".join([claim["claim"] for claim in claims_and_sources])

    for idx, claims_and_sources_batch in enumerate(claims_and_sources_batches):
        print(
            f"Processing batch {idx+1} of {len(claims_and_sources_batches)} (containing {len(claims_and_sources_batch)} claims)"
        )

        claim_str = "\n".join([claim["claim"] for claim in claims_and_sources_batch])
        num_batch_claims = len(claims_and_sources_batch)

        # we provide the full list of claims as context (to help resolve ambiguity), but only ask for queries for the current batch
        prompt = f"""\
You are working as part of a team and your individual task is to help check a subset of the following claims:\n
{all_claims_str}

Your individual task is as follows. \
For each of the {num_batch_claims} claims made below, provide a suitable Google search query that would enable a human to verify the claim. \
Note that Google can perform calculations and conversions, so you can use it to check numerical claims. \
If you think no Google query will be useful, then write "no suitable query". \
Each proposed Google search query should be on a separate line (do not prefix your queries with bullet points or numbers). \
There should be {num_batch_claims} queries in total.\n \

{claim_str}
"""
        persona = "You are a careful research assistant who helps with fact-checking and editing informative articles."
        system_message = {"role": "system", "content": persona}
        user_message = {"role": "user", "content": prompt}
        messages = [system_message, user_message]

        with BlockTimer(f"Using OpenAI API to extract claims with {args.model}"):
            response = call_openai_with_exponetial_backoff(
                model=args.model,
                temperature=args.temperature,
                messages=messages,
            )

        cost = estimate_cost_of_text_generation_api_call(
            model=args.model, response=response, verbose=True
        )

        proposed_queries = response.choices[0].message.content
        batch_queries = proposed_queries.split("\n")
        assert (
            len(batch_queries) == num_batch_claims
        ), f"Expected {num_batch_claims} queries, but got {len(queries)}"
        print(f"Generated {len(batch_queries)} queries (cost: {cost:.4f} USD)")
        queries.extend(batch_queries)

    querysets = []
    for claim_and_source, query in zip(claims_and_sources, queries):
        queryset = {**claim_and_source, "search_query": query}
        querysets.append(queryset)

    dest_path.parent.mkdir(exist_ok=True, parents=True)
    with open(dest_path, "w") as f:
        json.dump(querysets, f, indent=4, sort_keys=True)


def main():
    args = parse_args()

    src_paths = list(
        PIPELINE_PATHS["extracted_claims_with_classifications_dir"].glob("**/*.json")
    )
    print(
        f"Found {len(src_paths)} claim files in {PIPELINE_PATHS['extracted_claims_with_classifications_dir']}"
    )
    dest_dir = PIPELINE_PATHS["search_queries_for_evidence"]

    for src_path in src_paths:
        dest_path = dest_dir / src_path.relative_to(
            PIPELINE_PATHS["extracted_claims_with_classifications_dir"]
        )
        if not dest_path.exists() or args.refresh:
            generate_search_queries(
                args=args,
                src_path=src_path,
                dest_path=dest_path,
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument(
        "--model", default="gpt-3.5-turbo", choices=["gpt-4", "gpt-3.5-turbo"]
    )
    parser.add_argument("--dest_dir", default="data/search_queries", type=Path)
    parser.add_argument("--api_key_path", default="OPENAI_API_KEY.txt")
    parser.add_argument("--max_claims_per_api_call", type=int, default=10)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
