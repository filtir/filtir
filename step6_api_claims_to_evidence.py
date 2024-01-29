import faiss
import shutil
from beartype import beartype
import numpy as np
import json
import argparse
from zsvision.zs_utils import BlockTimer
import tiktoken
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import multiprocessing as mp
from zsvision.zs_multiproc import starmap_with_kwargs
from llm_api_utils import init_openai_with_api_key, PRICE_PER_1K_TOKENS
import multiprocessing as mp
from zsvision.zs_multiproc import starmap_with_kwargs
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore
from pipeline_paths import PIPELINE_PATHS


class ClaimToEvidence:
    def __init__(
        self,
        embedding_model="ada",
        limit=0,
        refresh=False,
        processes=1,
        num_chunks_per_worker=50,
        filter_str="",
        text_embedding_chunk_size=500,
        k_nearest_neighbours=3,
    ):
        self.embedding_model = embedding_model
        self.limit = limit
        self.refresh = refresh
        self.processes = processes
        self.num_chunks_per_worker = num_chunks_per_worker
        self.filter_str = filter_str
        self.text_embedding_chunk_size = text_embedding_chunk_size
        self.k_nearest_neighbours = k_nearest_neighbours

    @beartype
    def link_claims_to_evidence(
        self,
        metas,
        faiss_db,
    ):
        embedding_function = OpenAIEmbeddings()

        # build a query from the claim and source fragment
        queries = [
            f"Evidence for {x['claim']} (Based on {x['verbatim_quote']})" for x in metas
        ]
        encoding = tiktoken.encoding_for_model(self.embedding_model)

        num_tokens = len(encoding.encode(" ".join(queries)))
        print(
            f"Step6: Estimated cost: {num_tokens * PRICE_PER_1K_TOKENS[self.embedding_model]['embed'] / 1000:.2f} USD"
        )
        k_nearest_neighbours = min(
            len(faiss_db.index_to_docstore_id), self.k_nearest_neighbours
        )

        for text_query, meta in zip(queries, metas):
            docs_and_scores = faiss_db.similarity_search_with_relevance_scores(
                text_query, k=k_nearest_neighbours
            )

            # allow evidence to be serialised
            evidences = []
            for document, score in docs_and_scores:
                evidence = {
                    "chunk_tag": document.metadata["chunk_tag"],
                    "link": document.metadata["link"],
                    "query": document.metadata["query"],
                    "date_accessed": document.metadata["date_accessed"],
                    "text": document.page_content,
                    "similarity_score": float(score),
                }
                evidences.append(evidence)

            meta["evidences"] = evidences
            meta["embedded_query_used_to_find_evidence"] = text_query

        print(f"Returning {len(metas)} queries with supporting evidence")
        return metas


def main():
    args = parse_args()

    claim_to_evidence = ClaimToEvidence(
        embedding_model=args.embedding_model,
        limit=args.limit,
        refresh=args.refresh,
        processes=args.processes,
        api_key_path=args.api_key_path,
        num_chunks_per_worker=args.num_chunks_per_worker,
        filter_str=args.filter_str,
        text_embedding_chunk_size=args.text_embedding_chunk_size,
        k_nearest_neighbours=args.k_nearest_neighbours,
    )

    init_openai_with_api_key(api_key_path=args.api_key_path)
    src_dir = PIPELINE_PATHS["objective_claims_dir"]
    src_paths = list(src_dir.glob("**/*.json"))
    print(f"Found {len(src_paths)} claim files in {src_dir}")

    if args.filter_str:
        num_paths = len(src_paths)
        src_paths = [
            src_path for src_path in src_paths if args.filter_str in src_path.name
        ]
        print(f"Filtering for {args.filter_str} (from {num_paths} to {len(src_paths)})")

    dest_dir = PIPELINE_PATHS["web_evidence_chunks"]
    faiss_persist_dir = (
        PIPELINE_PATHS["faiss_db_embeddings_for_evidence"]
        / f"{args.embedding_model}_chunk_size_{args.text_embedding_chunk_size}"
    )

    kwarg_list = []
    dest_paths = []
    for idx, src_path in enumerate(src_paths):
        dest_path = dest_dir / src_path.relative_to(src_dir)
        with open(src_path) as f:
            metas = json.load(f)
        kwarg_list.append(
            {
                "metas": metas,
                "faiss_persist_dir": faiss_persist_dir,
            }
        )
        dest_paths.append(dest_path)

    # single process
    if args.processes == 1:
        results = []
        for kwargs in kwarg_list:
            result = claim_to_evidence.link_claims_to_evidence(**kwargs)
            results.append(result)
    else:  # multiprocess
        func = claim_to_evidence.link_claims_to_evidence
        with mp.Pool(processes=args.processes) as pool:
            results = starmap_with_kwargs(pool=pool, func=func, kwargs_iter=kwarg_list)

    for result, dest_path in zip(results, dest_paths):
        dest_path.parent.mkdir(exist_ok=True, parents=True)
        with open(dest_path, "w") as f:
            f.write(json.dumps(result, indent=4, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding_model",
        default="ada",
        choices=["ada", "babbage"],
        help="choose a model to compute chunk embeddings",
    )
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--api_key_path", default="OPENAI_API_KEY.txt")
    parser.add_argument("--num_chunks_per_worker", type=int, default=50)
    parser.add_argument("--filter_str", default="")
    parser.add_argument("--text_embedding_chunk_size", default=500, type=int)
    parser.add_argument("--k_nearest_neighbours", default=3, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
