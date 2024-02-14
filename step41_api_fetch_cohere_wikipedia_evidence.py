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
