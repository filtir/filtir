import os
from functools import lru_cache
from pipeline_paths import PIPELINE_PATHS
from langchain_community.utilities import GoogleSearchAPIWrapper

@lru_cache(maxsize=2)
def get_search_wrapper():
    os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_CLOUD_API_KEY")

    os.environ["GOOGLE_CSE_ID"] = os.environ.get("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
    return GoogleSearchAPIWrapper()


def get_google_search_results(query_str: str, num_results: int):
    google_search_tool = get_search_wrapper()
    search_results = google_search_tool.results(
        query=query_str, num_results=num_results
    )
    return search_results
