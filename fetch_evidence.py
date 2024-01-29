import requests
from bs4 import BeautifulSoup
import json
import json5
import argparse
from pathlib import Path
import multiprocessing as mp
from zsvision.zs_multiproc import starmap_with_kwargs
from pipeline_paths import PIPELINE_PATHS
from datetime import datetime
import urllib.robotparser
import urllib.parse
from utils import get_google_search_results

import time
from random import randint
from fake_useragent import UserAgent
from newspaper import Article, Config


def can_scrape(url, user_agent="*"):
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{url.scheme}://{url.netloc}/robots.txt")
    # be conservative - if we can't find robots.txt, don't scrapes
    try:
        rp.read()
        ok_to_scrape = rp.can_fetch(user_agent, url.geturl())
    except urllib.error.URLError:
        ok_to_scrape = False
    return ok_to_scrape


def fetch_search_results_to_gather_evidence(
    args,
    idx: int,
    total: int,
    search_results_dest_path: Path,
    queryset: dict,
):
    user_agent = UserAgent()
    config = Config()
    config.fetch_images = False
    print(f"Query {idx}/{total}")

    search_results_dest_path.parent.mkdir(exist_ok=True, parents=True)

    # check if we already have search_results for this title
    if search_results_dest_path.exists() and not args.refresh:
        print(f"Found existing search results at {search_results_dest_path}, skipping")
        return 0

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    # we assume some sites won't permit scraping, so we'll skip these
    num_results = args.num_search_results_to_keep + 5
    results = {}

    for item in queryset:
        if item["search_query"] == "no suitable query":
            item["search_results"] = []
            continue

        search_results = get_google_search_results(
            query_str=item["search_query"], num_results=num_results
        )

        if search_results == [{"Result": "No good Google Search Result was found"}]:
            item["search_results"] = []
            continue

        parsed_results = []
        for search_result in search_results:
            if not can_scrape(
                urllib.parse.urlparse(search_result["link"]), user_agent="MyScraper"
            ):
                print(
                    f"Skipping {search_result['link']} because it doesn't permit scraping"
                )
                continue
            try:
                config.browser_user_agent = user_agent.random
                article = Article(search_result["link"], language="en", config=config)
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
                    print(f"Error parsing article: {exception}")
                    raise exception

            search_result["text"] = text
            parsed_results.append(search_result)
            if len(parsed_results) == args.num_search_results_to_keep:
                break
        item["search_results"] = parsed_results

    # update the queryset with new information
    date_str = datetime.now().strftime("%Y-%m-%d")
    results = {"documents": queryset, "dates": {"search_results_fetched": date_str}}

    print(
        f"Writing web pages for search results for {len(queryset)} queries to {search_results_dest_path}"
    )
    with open(search_results_dest_path, "w") as f:
        f.write(json.dumps(results, indent=4, sort_keys=True))


def main():
    args = parse_args()
    search_query_paths = list(
        PIPELINE_PATHS["search_queries_for_evidence"].glob("**/*.json")
    )

    if args.limit:
        print(f"Limited to {args.limit} search querysets")
        search_query_paths = search_query_paths[: args.limit]

    kwarg_list = []
    for idx, search_query_path in enumerate(search_query_paths):
        rel_path = search_query_path.relative_to(
            PIPELINE_PATHS["search_queries_for_evidence"]
        )
        dest_path = PIPELINE_PATHS["google_search_results_evidence"] / rel_path

        if dest_path.exists() and not args.refresh:
            print(f"For {search_query_path}, found results at {dest_path}, skipping")
            continue

        with open(search_query_path, "r") as f:
            queryset = json.load(f)
            kwarg_list.append(
                {
                    "idx": idx,
                    "total": len(search_query_paths),
                    "search_results_dest_path": dest_path,
                    "args": args,
                    "queryset": queryset,
                }
            )

    # provide the total number of queries to each process
    for kwargs in kwarg_list:
        kwargs["total"] = len(kwarg_list)

    # single process
    if args.processes == 1:
        cost = 0
        for kwargs in kwarg_list:
            fetch_search_results_to_gather_evidence(**kwargs)
    else:  # multiprocess
        func = fetch_search_results_to_gather_evidence
        with mp.Pool(processes=args.processes) as pool:
            starmap_with_kwargs(pool=pool, func=func, kwargs_iter=kwarg_list)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="gpt-3.5-turbo", choices=["gpt-4", "gpt-3.5-turbo"]
    )
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--num_search_results_to_keep", type=int, default=3)
    parser.add_argument("--processes", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
