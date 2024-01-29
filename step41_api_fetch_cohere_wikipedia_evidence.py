import argparse
import json
import multiprocessing as mp
from datetime import datetime
import time
from zsvision.zs_multiproc import starmap_with_kwargs
import weaviate
from pathlib import Path
from pipeline_paths import PIPELINE_PATHS
import os


class CohereEvidence:
    def __init__(self, processes=8, filter_str="", refresh=False):
        self.processes = processes
        self.filter_str = filter_str
        self.refresh = refresh

    def semantic_search(self, query, client, results_lang=""):
        """
        Query the vectors database and return the top results.


        Parameters
        ----------
            query: str
                The search query

            results_lang: str (optional)
                Retrieve results only in the specified language.
                The demo dataset has those languages:
                en, de, fr, es, it, ja, ar, zh, ko, hi

        """

        nearText = {"concepts": [query]}
        properties = ["text", "title", "url", "views", "lang", "_additional {distance}"]

        # To filter by language
        if results_lang != "":
            where_filter = {
                "path": ["lang"],
                "operator": "Equal",
                "valueString": results_lang,
            }
            response = (
                client.query.get("Articles", properties)
                .with_where(where_filter)
                .with_near_text(nearText)
                .with_limit(5)
                .do()
            )

        # Search all languages
        else:
            response = (
                client.query.get("Articles", properties)
                .with_near_text(nearText)
                .with_limit(5)
                .do()
            )

        result = response["data"]["Get"]["Articles"]

        return result

    def fetch_cohere_semantic_search_results_to_gather_evidence(
        self,
        queryset: dict,
    ):
        """
        Generate a search query that can be used to verify a claim.
        """
        # 10M wiki embeddings (1M in English)
        weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")

        cohere_api_key = os.environ.get("COHERE_API_KEY")

        client = weaviate.Client(
            url="https://cohere-demo.weaviate.network/",
            auth_client_secret=weaviate.auth.AuthApiKey(
                api_key=weaviate_api_key
            ),  # Replace w/ your Weaviate instance API key
            additional_headers={
                "X-Cohere-Api-Key": cohere_api_key  # Replace with your inference API key
            },
        )

        while not client.is_ready():
            print(f"Waiting for client to be ready")
            time.sleep(1)

        for item in queryset:
            results = self.semantic_search(
                item["claim"], client=client, results_lang="en"
            )
            # rename "url" to "link" to be consistent with google results
            reformatted_results = []
            for result in results:
                result["link"] = result.pop("url")
                reformatted_results.append(result)
            item["search_results"] = reformatted_results

        # update the queryset with new information
        date_str = datetime.now().strftime("%Y-%m-%d")
        results = {
            "documents": queryset,
            "dates": {"results_fetched_from_wikipedia_1M_with_cohere-22-12": date_str},
        }
        print(f"Returning Cohere Wikipedia paragraph for {len(queryset)} queries")
        return results


def main():
    args = parse_args()
    cohere_evidence = CohereEvidence(
        cohere_api_key_path=args.cohere_api_key_path,
        processes=args.processes,
        filter_str=args.filter_str,
        refresh=args.refresh,
    )

    src_dir = PIPELINE_PATHS["objective_claims_dir"]
    src_paths = list(src_dir.glob("**/*.json"))

    if args.filter_str:
        num_paths = len(src_paths)
        src_paths = [
            src_path for src_path in src_paths if args.filter_str in src_path.name
        ]
        print(f"Filtering for {args.filter_str} (from {num_paths} to {len(src_paths)})")
    else:
        print(f"Found {len(src_paths)} files in {src_dir}")

    dest_dir = PIPELINE_PATHS["cohere_wikipedia_evidence"]

    kwarg_list = []

    dest_paths = []
    for idx, objective_claim_path in enumerate(src_paths):
        dest_path = dest_dir / objective_claim_path.relative_to(src_dir)
        # check if we already have evidence for this set of claims
        if not dest_path.exists() or args.refresh:
            with open(objective_claim_path, "r") as f:
                queryset = json.load(f)
                kwarg_list.append(
                    {
                        "queryset": queryset,
                    }
                )
            dest_paths.append(dest_path)

    # single process
    if args.processes == 1:
        cost = 0
        results = []
        for kwargs in kwarg_list:
            result = (
                cohere_evidence.fetch_cohere_semantic_search_results_to_gather_evidence(
                    **kwargs
                )
            )
            results.append(result)
    else:  # multiprocess
        func = cohere_evidence.fetch_cohere_semantic_search_results_to_gather_evidence
        # TODO: check if this is returning results in the same order as the inputs
        with mp.Pool(processes=args.processes) as pool:
            results = starmap_with_kwargs(pool=pool, func=func, kwargs_iter=kwarg_list)

    for result, dest_path in zip(results, dest_paths):
        dest_path.parent.mkdir(exist_ok=True, parents=True)
        with open(dest_path, "w") as f:
            f.write(json.dumps(result, indent=4, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohere_api_key_path", default="COHERE_API_KEY.txt")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--filter_str", default="")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
