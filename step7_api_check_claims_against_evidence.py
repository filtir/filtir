import argparse
import json
import multiprocessing as mp
from zsvision.zs_multiproc import starmap_with_kwargs
from pathlib import Path
from zsvision.zs_utils import BlockTimer
from typing import List, Dict
from llm_api_utils import (
    call_openai_with_exponetial_backoff,
    estimate_cost_of_text_generation_api_call,
    init_openai_with_api_key,
)
from pipeline_paths import PIPELINE_PATHS


class CheckClaimAgainstEvidence:
    def __init__(
        self,
        temperature=0.0,
        max_num_evidences=2,
        model="gpt-3.5-turbo",
        src_dir=Path("data/raw"),
        dest_dir=Path("data/extracted_claims"),
        filter_str="",
        processes=8,
        refresh=False,
    ):
        self.temperature = temperature
        self.max_num_evidences = max_num_evidences
        self.model = model
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        self.filter_str = filter_str
        self.processes = processes
        self.refresh = refresh

    def check_claim_against_evidence(
        self,
        claim: str,
        evidences: List[Dict[str, str]],
    ):
        init_openai_with_api_key()
        evidence_str = ""
        for evidence in evidences:
            # avoid linebreaks in each piece of evidence, else it can create a confusing prompt
            text_evidence = evidence["text"].replace("\n", " ")
            evidence_str += f"{text_evidence}\n"
            evidence_str += f"URL: {evidence['link']}'\n"
            evidence_str += f"Date accessed: {evidence['date_accessed']}\n\n"

        prompt = f"""\
Your task is to assess whether a claim is correct based on the given pieces of evidence.

Your answer should be in json format as follows:
{{
    "verdict": "<verdict>",
    "justification": "<justification for the verdict>",
    "quotes": ["<most relevant verbatim quotes from evidence>"],
    "URLs": "<URL sources for verbatim quotes>",
    "date_accessed": "<access dates for URL quotes>"
}}
The <verdict> label should be one of the following:
"Fully supported", "Partially supported", "Unsupported"

When quoting the relevant sentence from the evidence, be careful to copy it **EXACTLY** (with no edits).
---
## Example

**Claim**:
Hannah Arendt was born in 1906.

**Pieces of evidence**:
Hannah Arendt was a 20th-century German-Jewish political thinker and philosopher. She was born in Linden, Hanover, Germany in 1906. When she was three her family moved to Königsberg so that her father’s syphilis could be treated. He died when she was seven years old. Königsberg was where Immanuel Kant was born, right?

Königsberg was where Immanuel Kant was born, right?
URL: https://fivebooks.com/best-books/hannah-arendt-samantha-rose-hill/'
Date accessed: 2023-05-10

Hannah Arendt was born as Johanna Arendt in 1906, in the Wilhelmine period. Her German Jewish family were comfortable, educated and secular in Linden, Prussia (now a part of Hanover). They were merchants of Russian extraction from Königsberg.[a] Her grandparents were members of the Reform Jewish community. Her paternal grandfather, Max Arendt [de] (1843–1913), was a prominent businessman, local politician, a leader of the Königsberg Jewish community and a member of the Centralverein deutscher
URL: https://en.wikipedia.org/wiki/Hannah_Arendt'
Date accessed: 2023-05-10


**Assessment**:
{{
    "verdict": "Fully supported",
    "justification": "The claim about Hannah Arendt's birth date is directly supported by the evidence."
    "quote": "Hannah Arendt was born as Johanna Arendt in 1906, in the Wilhelmine period.",
    "URL": "https://en.wikipedia.org/wiki/Hannah_Arendt",
    "date_accessed": "2023-05-10"
}}
---
**Claim**:
{claim}

**Pieces of evidence**:
{evidence_str}
**Assessment**:\
    """
        persona = "You are a careful research assistant who helps with fact-checking and editing informative articles."
        system_message = {"role": "system", "content": persona}
        user_message = {"role": "user", "content": prompt}
        messages = [system_message, user_message]

        with BlockTimer(
            f"Using OpenAI API to check claims against evidence {self.model}"
        ):
            response = call_openai_with_exponetial_backoff(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                response_format={"type": "json_object"},
            )

        cost = estimate_cost_of_text_generation_api_call(
            model=self.model, response=response, verbose=True
        )

        assessment = response.choices[0].message.content
        assessment_dict = json.loads(assessment)
        return {"assessment": assessment_dict, "cost": cost}

    def check_claims_against_evidence(self, claims_with_evidence):
        """
        Checks claims against evidence.
        """
        kwarg_list = []
        results = []
        for idx, item in enumerate(claims_with_evidence):
            kwarg_list.append(
                {
                    "claim": item["claim"],
                    "evidences": item["evidences"][: self.max_num_evidences],
                }
            )
        if self.processes == 1:
            for kwargs in kwarg_list:
                results.append(self.check_claim_against_evidence(**kwargs))
        else:  # multiprocess
            func = self.check_claim_against_evidence
            with mp.Pool(processes=self.processes) as pool:
                results = starmap_with_kwargs(
                    pool=pool, func=func, kwargs_iter=kwarg_list
                )
        costs = [result["cost"] for result in results]
        print(f"Total cost: {sum(costs)} USD")
        assessed_claims = []
        for result, item in zip(results, claims_with_evidence):
            item["assessment"] = result["assessment"]
            item["verdict_model"] = self.model
            assessed_claims.append(item)

        print(f"Writing {len(assessed_claims)} assessed claims")
        return assessed_claims


def main():
    args = parse_args()
    check_claim_against_evidence = CheckClaimAgainstEvidence(
        temperature=args.temperature,
        max_num_evidences=args.max_num_evidences,
        model=args.model,
        src_dir=args.src_dir,
        dest_dir=args.dest_dir,
        api_key_path=args.api_key_path,
        filter_str=args.filter_str,
        processes=args.processes,
        refresh=args.refresh,
    )

    src_paths = list(PIPELINE_PATHS["web_evidence_chunks"].glob("*.json"))
    print(f"Found {len(src_paths)} files in {PIPELINE_PATHS['web_evidence_chunks']}")

    if args.filter_str:
        num_paths = len(src_paths)
        src_paths = [
            src_path for src_path in src_paths if args.filter_str in src_path.name
        ]
        print(f"Filtering for {args.filter_str} (from {num_paths} to {len(src_paths)})")

    for src_path in src_paths:
        dest_path = PIPELINE_PATHS["evaluated_claims_dir"] / src_path.name
        if not dest_path.exists() or args.refresh:
            with open(src_path, "r") as f:
                claims_with_evidence = json.load(f)
            assessed_claims = (
                check_claim_against_evidence.check_claims_against_evidence(
                    claims_with_evidence
                )
            )
            dest_path.parent.mkdir(exist_ok=True, parents=True)
            with open(dest_path, "w") as f:
                json.dump(assessed_claims, f, indent=4, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_num_evidences", type=int, default=2)
    parser.add_argument(
        "--model", default="gpt-3.5-turbo", choices=["gpt-4", "gpt-3.5-turbo"]
    )
    parser.add_argument("--src_dir", default="data/raw", type=Path)
    parser.add_argument("--dest_dir", default="data/extracted_claims", type=Path)
    parser.add_argument("--api_key_path", default="OPENAI_API_KEY.txt")
    parser.add_argument("--filter_str", default="")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
