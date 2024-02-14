import faiss
import shutil
from beartype import beartype
import numpy as np
import json
import argparse
from zsvision.zs_utils import BlockTimer
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import multiprocessing as mp
from zsvision.zs_multiproc import starmap_with_kwargs
from llm_api_utils import init_openai_with_api_key, PRICE_PER_1K_TOKENS
import multiprocessing as mp
from zsvision.zs_multiproc import starmap_with_kwargs
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore


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
