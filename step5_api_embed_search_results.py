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
from pipeline_paths import PIPELINE_PATHS
from llm_api_utils import (
    init_openai_with_api_key,
    EMBEDDING_DIMENSIONS,
    PRICE_PER_1K_TOKENS,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore


class EmbedResults:
    def __init__(
        self,
        embedding_model="ada",
        limit=0,
        refresh=False,
        refresh_faiss_db=False,
        text_embedding_chunk_size=500,
        filter_str="",
    ):
        self.embedding_model = embedding_model
        self.limit = limit
        self.refresh = refresh
        self.refresh_faiss_db = refresh_faiss_db
        self.text_embedding_chunk_size = text_embedding_chunk_size
        self.filter_str = filter_str

    @beartype
    def compute_embeddings_from_chunks(
        self, embedding_function: OpenAIEmbeddings, metadatas: list, faiss_db
    ):
        doc_chunks = []
        metadatas_without_chunks = []
        for metadata in metadatas:
            doc_chunk = metadata.pop("doc_chunk")
            doc_chunks.append(doc_chunk)
            metadatas_without_chunks.append(metadata)

        with BlockTimer(f"Embedding {len(metadatas)} fragments"):
            embeddings = embedding_function.embed_documents(doc_chunks)
            # account for name mangling in Python
            faiss_db._FAISS__add(doc_chunks, embeddings, metadatas_without_chunks)

        return faiss_db

    @beartype
    def parse_date_of_fetching(self, data: dict) -> str:
        evidence_keys = {
            "search_results_fetched",
            "results_fetched_from_wikipedia_1M_with_cohere-22-12",
        }
        for key in evidence_keys:
            if key in data["dates"]:
                evidence_fetched_date = data["dates"][key]
                return evidence_fetched_date
        raise ValueError(f"Could not find evidence fetched date in {data['dates']}")

    # TODO: embed_for_uuid and embed needs to be refactor
    def embed_for_uuid(self, srcs):
        init_openai_with_api_key()

        embedding_function = OpenAIEmbeddings()

        index = faiss.IndexFlatL2(EMBEDDING_DIMENSIONS[self.embedding_model])
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}
        faiss_db = FAISS(
            embedding_function=embedding_function.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        already_embedded_chunks = {
            doc.metadata["chunk_tag"] for doc in faiss_db.docstore._dict.values()
        }

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.text_embedding_chunk_size,
            chunk_overlap=0,
        )

        kwarg_list = []
        seen_links = set()
        metadatas = []
        total_chunks = 0
        chunks_to_embed = 0
        chunks_to_skip = 0

        # TODO, give this as input, not reading from disk!!
        for data in srcs:
            evidence_fetched_date = self.parse_date_of_fetching(data)

            for document in data["documents"]:
                for search_result in document["search_results"]:
                    # Don't embed the same link twice
                    if search_result["link"] in seen_links:
                        continue
                    seen_links.add(search_result["link"])

                    doc_chunks = [
                        doc.page_content
                        for doc in splitter.create_documents([search_result["text"]])
                    ]
                    chunk_tags = [
                        f"{search_result['link']}-chunk-{idx}-chunk_sz-{self.text_embedding_chunk_size}"
                        for idx in range(len(doc_chunks))
                    ]
                    for doc_chunk, chunk_tag in zip(doc_chunks, chunk_tags):
                        if chunk_tag not in already_embedded_chunks:
                            metadatas.append(
                                {
                                    "doc_chunk": doc_chunk,
                                    "link": search_result["link"],
                                    "chunk_tag": chunk_tag,
                                    "date_accessed": evidence_fetched_date,
                                    "query": document["claim"],
                                }
                            )
                            chunks_to_embed += 1
                        else:
                            chunks_to_skip += 1
                    total_chunks += len(doc_chunks)

        encoding = tiktoken.encoding_for_model(self.embedding_model)
        doc_chunks = [x["doc_chunk"] for x in metadatas]
        num_words = len(" ".join(doc_chunks).split())
        num_tokens = len(encoding.encode("".join(doc_chunks)))

        print(
            f"Created {total_chunks} chunks of text to answer from {len(seen_links)} websites"
        )
        print(
            f"Embedding {chunks_to_embed} (skipping {chunks_to_skip}) chunks of text from {len(seen_links)} websites)"
        )
        print(
            f"Embedding {num_tokens} tokens ({num_words} words) from {len(doc_chunks)} chunks"
        )
        print(
            f"Step5: Estimated cost: {num_tokens * PRICE_PER_1K_TOKENS[self.embedding_model]['embed'] / 1000:.2f} USD"
        )

        if metadatas:
            self.compute_embeddings_from_chunks(
                embedding_function=embedding_function,
                faiss_db=faiss_db,
                metadatas=metadatas,
            )

            return faiss_db
        return None

    def embed(self):
        init_openai_with_api_key()
        src_paths = []
        for evidence_key in (
            "google_search_results_evidence",
            "cohere_wikipedia_evidence",
        ):
            evidence_paths = list(PIPELINE_PATHS[evidence_key].glob("**/*.json"))
            src_paths.extend(evidence_paths)

        if self.filter_str:
            num_paths = len(src_paths)
            src_paths = [
                src_path for src_path in src_paths if self.filter_str in src_path.name
            ]
            print(
                f"Filtering for {self.filter_str} (from {num_paths} to {len(src_paths)})"
            )

        print(f"Found {len(src_paths)} collections of evidence")
        src_paths = sorted(src_paths)

        embedding_function = OpenAIEmbeddings()
        faiss_persist_dir = (
            PIPELINE_PATHS["faiss_db_embeddings_for_evidence"]
            / f"{self.embedding_model}_chunk_size_{self.text_embedding_chunk_size}"
        )

        if faiss_persist_dir.exists():
            if self.refresh_faiss_db:
                print(f"Deleting existing database at {faiss_persist_dir}")
                shutil.rmtree(faiss_persist_dir)

        # check which chunks we've already embedded to avoid duplication
        if faiss_persist_dir.exists() and not self.refresh_faiss_db:
            faiss_db = FAISS.load_local(
                folder_path=str(faiss_persist_dir), embeddings=embedding_function
            )
            print(f"Found existing database at {faiss_persist_dir}, using... ")
        else:
            index = faiss.IndexFlatL2(EMBEDDING_DIMENSIONS[self.embedding_model])
            docstore = InMemoryDocstore({})
            index_to_docstore_id = {}
            faiss_db = FAISS(
                embedding_function=embedding_function.embed_query,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
            )
            print(f"Persisting intialised database to {faiss_persist_dir}")
            faiss_db.save_local(folder_path=str(faiss_persist_dir))

        already_embedded_chunks = {
            doc.metadata["chunk_tag"] for doc in faiss_db.docstore._dict.values()
        }

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.text_embedding_chunk_size,
            chunk_overlap=0,
        )

        kwarg_list = []
        seen_links = set()
        metadatas = []
        total_chunks = 0
        chunks_to_embed = 0
        chunks_to_skip = 0

        for src_path in src_paths:
            with open(src_path, "r") as f:
                data = json.load(f)

            evidence_fetched_date = self.parse_date_of_fetching(data)

            for document in data["documents"]:
                for search_result in document["search_results"]:
                    # Don't embed the same link twice
                    if search_result["link"] in seen_links:
                        continue
                    seen_links.add(search_result["link"])

                    doc_chunks = [
                        doc.page_content
                        for doc in splitter.create_documents([search_result["text"]])
                    ]
                    chunk_tags = [
                        f"{search_result['link']}-chunk-{idx}-chunk_sz-{self.text_embedding_chunk_size}"
                        for idx in range(len(doc_chunks))
                    ]
                    for doc_chunk, chunk_tag in zip(doc_chunks, chunk_tags):
                        if chunk_tag not in already_embedded_chunks:
                            metadatas.append(
                                {
                                    "doc_chunk": doc_chunk,
                                    "link": search_result["link"],
                                    "chunk_tag": chunk_tag,
                                    "date_accessed": evidence_fetched_date,
                                    "query": document["claim"],
                                }
                            )
                            chunks_to_embed += 1
                        else:
                            chunks_to_skip += 1
                    total_chunks += len(doc_chunks)

        encoding = tiktoken.encoding_for_model(self.embedding_model)
        doc_chunks = [x["doc_chunk"] for x in metadatas]
        num_words = len(" ".join(doc_chunks).split())
        num_tokens = len(encoding.encode("".join(doc_chunks)))

        print(
            f"Created {total_chunks} chunks of text to answer from {len(seen_links)} websites"
        )
        print(
            f"Embedding {chunks_to_embed} (skipping {chunks_to_skip}) chunks of text from {len(seen_links)} websites)"
        )
        print(
            f"Embedding {num_tokens} tokens ({num_words} words) from {len(doc_chunks)} chunks"
        )
        print(
            f"Estimated cost: {num_tokens * PRICE_PER_1K_TOKENS[self.embedding_model]['embed'] / 1000:.2f} USD"
        )

        if metadatas:
            self.compute_embeddings_from_chunks(
                embedding_function=embedding_function,
                faiss_persist_dir=faiss_persist_dir,
                metadatas=metadatas,
            )


def main():
    args = parse_args()

    embed_results = EmbedResults(
        embedding_model=args.embedding_model,
        limit=args.limit,
        refresh=args.refresh,
        refresh_faiss_db=args.refresh_faiss_db,
        openai_api_key_path=args.openai_api_key_path,
        cohere_api_key_path=args.cohere_api_key_path,
        text_embedding_chunk_size=args.text_embedding_chunk_size,
        filter_str=args.filter_str,
    )
    # TODO: this togheter with embed need to be refactor
    embed_results.embed()


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
    parser.add_argument("--refresh_faiss_db", action="store_true")
    parser.add_argument("--openai_api_key_path", default="OPENAI_API_KEY.txt")
    parser.add_argument("--cohere_api_key_path", default="COHERE_API_KEY.txt")
    parser.add_argument("--text_embedding_chunk_size", default=500, type=int)
    parser.add_argument("--filter_str", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main()
