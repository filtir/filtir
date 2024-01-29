import argparse
import json
import multiprocessing as mp
from zsvision.zs_multiproc import starmap_with_kwargs
from pathlib import Path
from zsvision.zs_utils import BlockTimer
from text_utils import is_unique_verbatim_quote, parse_passage_quote_and_claim
from llm_api_utils import (
    call_openai_with_exponetial_backoff,
    estimate_cost_of_text_generation_api_call,
    init_openai_with_api_key,
)
from pipeline_paths import PIPELINE_PATHS
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ClaimExtractor:
    def __init__(
        self,
        temperature=0,
        model="gpt-3.5-turbo",
        filter_str="",
        processes=1,
        refresh=False,
    ):
        """Initializes ClaimExtractor with the provided arguments"""
        self.temperature = temperature
        self.model = model
        self.filter_str = filter_str
        self.processes = processes
        self.refresh = refresh

    def extract_claims_from_passage(
        self,
        idx: int,
        total: int,
        passage: str,
    ):
        init_openai_with_api_key()
        print(f"Processing passage {idx + 1} of {total}")
        prompt = f"""\
    Task:
    Enumerate all the discrete factual claims or logical assertions stated in the passage that follows the dashed horizontal line below. \
    To allow the claims to be linked to the passage, use the format: `VERBATIM_PASSAGE_QUOTE_FOR_CLAIM: <verbatim passage quote for claim>, CLAIM: <claim>` on each line. \
    The <verbatim passage quote for claim> must be A SINGLE UNEDITED SUBSTRING from the passage that uniquely identifies the claim. \
    The <verbatim passage quote for claim> must carefully preserve all punctuation and clauses from the original passage. \
    This text will be used in the final national exam.

    ----------
    Here is an example passage, together with the verbatim passage quotes and claims that should be extracted from it:

    Passage:
    Immanuel Kant was born in 1724 into a modest, devoutly religious family, with his father working as a saddle-maker. \
    He was one of nine children, but only five, including Kant, survived to adulthood. \
    His upbringing was steeped in the Pietist tradition, emphasizing intense religious devotion, a literal interpretation of the Bible, and a strong focus on personal morality. \
    Kant attended the University of Königsberg, studying various subjects, including theology, metaphysics, and natural science. \
    After completing his studies, Kant worked as a private tutor for nine years before returning to the University of Königsberg as a lecturer in 1755. \
    In his works Groundwork of the Metaphysics of Morals (1785) and Critique of Practical Reason (1788), Kant argues that morality is not contingent upon personal desires or cultural norms. \


    Extracted source phrases and claims:
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] Immanuel Kant was born in 1724 into a modest, devoutly religious family [CLAIM] Immanuel Kant was born in 1724.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] Immanuel Kant was born in 1724 into a modest, devoutly religious family [CLAIM] Immanuel Kant was born into a modest family.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] Immanuel Kant was born in 1724 into a modest, devoutly religious family [CLAIM] Immanuel Kant was born into a devoutly religious family.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] with his father working as a saddle-maker [CLAIM] Immnauel Kant's father worked as a saddle-maker.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] He was one of nine children [CLAIM] Immanuel Kant was one of nine children.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] but only five, including Kant survived to adulthood [CLAIM] Only five of Immanuel Kant's parents' children survived to adulthood.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] His upbringing was steeped in the Pietist tradition [CLAIM] Immanuel Kant's upbringing was steeped in the Pietist tradition.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] emphasizing intense religious devotion [CLAIM] Immanuel Kant's upbringing emphasized intense religious devotion.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] a literal interpretation of the Bible [CLAIM] Immanuel Kant's upbringing emphasized a literal interpretation of the Bible.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] a strong focus on personal morality [CLAIM] Immanuel Kant's upbringing emphasized a strong focus on personal morality.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] Kant attended the University of Königsberg [CLAIM] Immanuel Kant attended the University of Königsberg.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] studying various subjects, including theology, metaphysics, and natural science [CLAIM] Immanuel Kant studied theology.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] studying various subjects, including theology, metaphysics, and natural science [CLAIM] Immanuel Kant studied metaphysics.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] studying various subjects, including theology, metaphysics, and natural science [CLAIM] Immanuel Kant studied natural science.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] After completing his studies [CLAIM] Immanuel Kant completed his studies.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] After completing his studies, Kant worked as a private tutor for nine years [CLAIM] After completing his studies, Immanuel Kant worked as a private tutor.
    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] before returning to the University of Königsberg as a lecturer in 1755 [CLAIM] Immanuel Kant returned to the University of Königsberg as a lecturer in 1755.

    ----------
    Passage:
    {passage}

    Extracted source phrases and claims:\
    """
        persona = "You are a careful research assistant who helps with fact-checking and editing informative articles."
        system_message = {"role": "system", "content": persona}
        user_message = {"role": "user", "content": prompt}
        messages = [system_message, user_message]

        with BlockTimer(f"Using OpenAI API to extract claims with {self.model}"):
            response = call_openai_with_exponetial_backoff(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
            )

        cost = estimate_cost_of_text_generation_api_call(
            model=self.model, response=response, verbose=True
        )
        content = response.choices[0].message.content
        content = content.strip()
        quotes_and_claims = content.split("\n")

        parsed_claims = []
        for quote_and_claim in quotes_and_claims:
            quote_and_claim = quote_and_claim.strip()
            if "[VERBATIM_PASSAGE_QUOTE_FOR_CLAIM]" not in quote_and_claim:
                quote_and_claim = quote_and_claim.replace(
                    "VERBATIM_PASSAGE_QUOTE_FOR_CLAIM: ",
                    "[VERBATIM_PASSAGE_QUOTE_FOR_CLAIM]: ",
                )
            if "[CLAIM]" not in quote_and_claim:
                quote_and_claim = quote_and_claim.replace(" CLAIM:", " [CLAIM]:")

            if "[VERBATIM_PASSAGE_QUOTE_FOR_CLAIM]" not in quote_and_claim:
                continue
            quote_and_claim = quote_and_claim.strip()
            parsed = parse_passage_quote_and_claim(quote_and_claim)
            is_unique_and_verbatim = is_unique_verbatim_quote(
                verbatim_quote=parsed["verbatim_quote"], original_passage=passage
            )
            parsed["is_unique_and_verbatim"] = is_unique_and_verbatim
            parsed_claims.append(parsed)

        return {"claims": parsed_claims, "cost": cost}

    def extract_claims(self, text_input):
        """
        Extracts claims from text_input and return the extracted claims in a json file
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents([text_input])
        print(f"Split text into {len(docs)} documents")
        all_claims = []

        kwarg_list = []
        for idx, doc in enumerate(docs):
            # remove newlines from the passage to avoid a confusing prompt format
            passage = doc.page_content.replace("\n", " ")
            kwarg_list.append(
                {
                    "idx": idx,
                    "total": len(docs),
                    "passage": passage,
                }
            )

        # TODO: this won't work with Celery
        if self.processes == 1:
            results = []
            for kwargs in kwarg_list:
                results.append(self.extract_claims_from_passage(**kwargs))
        else:  # multiprocess
            func = self.extract_claims_from_passage
            with mp.Pool(processes=self.processes) as pool:
                results = starmap_with_kwargs(
                    pool=pool, func=func, kwargs_iter=kwarg_list
                )

        cost = sum([result["cost"] for result in results])
        all_claims = []
        for result in results:
            all_claims.extend(result["claims"])

        print(f"Returning {len(all_claims)} claims (cost: {cost} USD)")
        return all_claims


# Keep the original functionality
def main():
    args = parse_args()
    # Create an object of ClaimExtractor
    extractor = ClaimExtractor(
        temperature=args.temperature,
        model=args.model,
        filter_str=args.filter_str,
        api_key_path=args.api_key_path,
        processes=args.processes,
        refresh=args.refresh,
    )

    src_dir = PIPELINE_PATHS["source_document_dir"]
    src_paths = list(src_dir.glob("*.txt"))

    if args.filter_str:
        num_paths = len(src_paths)
        src_paths = [
            src_path for src_path in src_paths if args.filter_str in src_path.name
        ]
        print(f"Filtering for {args.filter_str} (from {num_paths} to {len(src_paths)})")
    else:
        print(f"Found {len(src_paths)} files in {src_dir}")

    for src_path in src_paths:
        with open(src_path, "r") as f:
            text_input = f.read()
        rel_path = src_path.relative_to(src_dir).with_suffix(".json")
        dest_path = PIPELINE_PATHS["extracted_claims_dir"] / rel_path
        if not dest_path.exists() or args.refresh:
            claims = extractor.extract_claims(text_input)
            dest_path.parent.mkdir(exist_ok=True, parents=True)
            with open(dest_path, "w") as f:
                json.dump(claims, f, indent=4, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument(
        "--model", default="gpt-3.5-turbo", choices=["gpt-4", "gpt-3.5-turbo"]
    )
    parser.add_argument("--filter_str", default="")
    parser.add_argument("--api_key_path", default="OPENAI_API_KEY.txt")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
