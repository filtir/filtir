import requests
from bs4 import BeautifulSoup
from zsvision.zs_utils import BlockTimer
import json
import json5
import argparse
from pathlib import Path
import multiprocessing as mp
from zsvision.zs_multiproc import starmap_with_kwargs
from datetime import datetime
import urllib.robotparser
import urllib.parse
from urllib.parse import urlunparse
from utils import get_google_search_results
from pipeline_paths import PIPELINE_PATHS

import time
from random import randint
from fake_useragent import UserAgent
from newspaper import Article, Config


class GoogleEvidence:
    def __init__(
        self,
        model="gpt-3.5-turbo",
        limit=0,
        refresh=False,
        num_search_results_to_keep=3,
        filter_str="",
        processes=8,
    ):
        self.model = model
        self.limit = limit
        self.refresh = refresh
        self.num_search_results_to_keep = num_search_results_to_keep
        self.filter_str = filter_str
        self.processes = processes

    def can_index(self, url, user_agent_name):
        rp = urllib.robotparser.RobotFileParser()
        robots_url = f"{url.scheme}://{url.netloc}/robots.txt"

        headers = {
            "User-Agent": user_agent_name,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        try:
            req = urllib.request.Request(robots_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                rp.parse(response.read().decode("utf-8").splitlines())

            ok_to_index = rp.can_fetch(user_agent_name, url.geturl())
        except urllib.error.URLError:
            # If there is no robots.txt or there is an error accessing it, assume it's okay to index
            ok_to_index = True
        except Exception as e:
            # TODO: fix this hacky solution
            print(f"An unexpected error occurred in step42: {e}")
            # going the safe route
            ok_to_index = False
        return ok_to_index

    def fetch_search_results_to_gather_evidence(
        self,
        queryset: dict,
    ):
        user_agent = UserAgent()
        config = Config()
        config.fetch_images = False

        user_agent_name = "FiltirBot/1.0 (+https://filtir.com/filtirbot-info)"

        headers = {
            "User-Agent": user_agent_name,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        # we assume some sites won't permit indexing, so we'll skip these
        num_results = self.num_search_results_to_keep + 5
        results = {}

        print(f"Found {len(queryset)} claims to fetch search results for")

        for queryset_idx, item in enumerate(queryset):
            with BlockTimer(
                f"Fetching search results from Google {queryset_idx + 1}/{len(queryset)}"
            ):
                search_results = get_google_search_results(
                    query_str=item["claim"], num_results=num_results
                )

            if search_results == [{"Result": "No good Google Search Result was found"}]:
                item["search_results"] = []
                continue

            parsed_results = []
            for search_result in search_results:
                if not self.can_index(
                    urllib.parse.urlparse(search_result["link"]),
                    user_agent_name=user_agent_name,
                ):
                    print(
                        f"Skipping {search_result['link']} because it doesn't permit indexing"
                    )
                    continue
                try:
                    config.browser_user_agent = user_agent.random
                    article = Article(
                        search_result["link"], language="en", config=config
                    )
                    article.download()
                    article.parse()
                    text = article.text
                except Exception as e:
                    print(f"Error parsing article: {e}, trying with requests.get...")
                    try:
                        response = requests.get(
                            search_result["link"], timeout=15, headers=headers
                        )
                        html = response.text
                        soup = BeautifulSoup(html, features="html.parser")
                        text = soup.get_text()
                    except Exception as exception:
                        print(f"Error parsing article: {exception}, skipping")
                        continue

                search_result["text"] = text
                parsed_results.append(search_result)
                if len(parsed_results) == self.num_search_results_to_keep:
                    break
            item["search_results"] = parsed_results

        # update the queryset with new information
        date_str = datetime.now().strftime("%Y-%m-%d")
        results = {"documents": queryset, "dates": {"search_results_fetched": date_str}}

        print(f"Returning web pages for search results for {len(queryset)} queries")
        return results


def main():
    args = parse_args()

    google_evidence = GoogleEvidence(
        model=args.model,
        limit=args.limit,
        refresh=args.refresh,
        num_search_results_to_keep=args.num_search_results_to_keep,
        filter_str=args.filter_str,
        processes=args.processes,
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

    print(f"Found {len(src_paths)} claim files in {src_dir}")
    dest_dir = PIPELINE_PATHS["google_search_results_evidence"]

    if args.limit:
        print(f"Limited to {args.limit} search querysets")
        src_paths = src_paths[: args.limit]

    kwarg_list = []
    dest_paths = []
    for idx, objective_claims_path in enumerate(src_paths):
        rel_path = objective_claims_path.relative_to(src_dir)
        dest_path = dest_dir / rel_path

        if dest_path.exists() and not args.refresh:
            print(
                f"For {objective_claims_path}, found results at {dest_path}, skipping"
            )
            continue

        with open(objective_claims_path, "r") as f:
            queryset = json.load(f)
            kwarg_list.append(
                {
                    # "search_results_dest_path": dest_path,
                    "queryset": queryset,
                }
            )
        dest_paths.append(dest_path)

    # single process
    if args.processes == 1:
        results = []
        for kwargs in kwarg_list:
            result = google_evidence.fetch_search_results_to_gather_evidence(**kwargs)
            results.append(result)
    else:  # multiprocess
        func = google_evidence.fetch_search_results_to_gather_evidence
        with mp.Pool(processes=args.processes) as pool:
            results = starmap_with_kwargs(pool=pool, func=func, kwargs_iter=kwarg_list)

    for result, dest_path in zip(results, dest_paths):
        dest_path.parent.mkdir(exist_ok=True, parents=True)
        with open(dest_path, "w") as f:
            f.write(json.dumps(result, indent=4, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="gpt-3.5-turbo", choices=["gpt-4", "gpt-3.5-turbo"]
    )
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--num_search_results_to_keep", type=int, default=3)
    parser.add_argument("--filter_str", default="")
    parser.add_argument("--processes", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
