import json
import argparse
import multiprocessing as mp
from zsvision.zs_multiproc import starmap_with_kwargs
from typing import List, Dict
import numpy as np
from zsvision.zs_utils import BlockTimer
from llm_api_utils import (
    call_openai_with_exponetial_backoff,
    estimate_cost_of_text_generation_api_call,
    init_openai_with_api_key,
)
import random


class ClassifyClaims:
    def __init__(
        self,
        temperature=0,
        model="gpt-3.5-turbo",
        max_claims_per_api_call=10,
        processes=8,
        filter_str="",
        refresh=False,
    ):
        self.temperature = temperature
        self.model = model
        self.max_claims_per_api_call = max_claims_per_api_call
        self.processes = processes
        self.filter_str = filter_str
        self.refresh = refresh
        self.objective_claims_file = "objective_claims.txt"
        self.subjective_claims_file = "subjective_claims.txt"

    def parse_classification_label(self, text: str) -> str:
        raw = text.strip()
        if raw.endswith("[objective]"):
            label = "objective"
        elif raw.endswith("[subjective]"):
            label = "subjective"
        else:
            raise ValueError(f"Invalid label: {raw}")
        return label

    def read_file(self, file_name):
        with open(file_name, "r") as f:
            lines = []
            for line in f:
                parsed_line = line.strip()
                lines.append(parsed_line)
        return lines

    def create_few_shot_learning_prompt(self) -> str:
        objective_list = self.read_file(self.objective_claims_file)
        subjective_list = self.read_file(self.subjective_claims_file)
        merged_list = list(
            zip(objective_list, ["[objective]"] * len(objective_list))
        ) + list(zip(subjective_list, ["[subjective]"] * len(subjective_list)))

        # Randomizing the merged list with a specific seed
        seed = 1234
        random.seed(seed)
        random.shuffle(merged_list)
        prompt = "Claims:\n"
        for claim, _ in merged_list:
            prompt += claim + "\n"
        prompt += "\nClassifications:\n"
        for claim, classif in merged_list:
            prompt += claim + " " + classif + "\n"
        return prompt

    def classify_claim_batch(
        self,
        idx: int,
        total: int,
        claims_and_sources_batch: List[Dict[str, str]],
    ):
        print(
            f"Processing batch {idx+1} of {total} (containing {len(claims_and_sources_batch)} claims)"
        )

        claim_str = "\n".join([claim["claim"] for claim in claims_and_sources_batch])
        num_batch_claims = len(claims_and_sources_batch)
        few_shot = self.create_few_shot_learning_prompt()
        prompt = f"""\
Objective claims can be verified based on factual data (such as those that could be verified by \
referencing an encyclopedia), whereas subjective claims involve a personal interpretation of \
the data and are more open to debate. \
For each of the following claims given below the dashed horizontal line, classify them as \
[subjective] or [objective] by suffixing the claim with the appropriate label. OUTPUT ONLY the class, either subjective or objective for each claim!

Here are some examples:

{few_shot}
----------
Claims:
{claim_str}

Classifications:\
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

        proposed_classified_claims = response.choices[0].message.content
        batch_classified_claims = proposed_classified_claims.split("\n")

        content = response.choices[0].message.content
        batch_classified_claims = content.split("\n")
        assert (
            len(batch_classified_claims) == num_batch_claims
        ), f"Expected {num_batch_claims} claims, but got {len(batch_classified_claims)}"
        print(f"Generated {len(batch_classified_claims)} claims (cost: {cost:.4f} USD)")

        claims_with_labels = []
        for claim_and_source, classified_claim in zip(
            claims_and_sources_batch, batch_classified_claims
        ):
            claim_label = self.parse_classification_label(classified_claim)
            claim_and_source["label"] = claim_label
            claims_with_labels.append(claim_and_source)
        return {"claims_with_labels": claims_with_labels, "cost": cost}

    def classify_claims(self, claims_and_sources):
        """
        Classify claims as being either subjective or objective, and write the results to a file.
        """
        init_openai_with_api_key()
        num_claims = len(claims_and_sources)

        # we limit the number of claims per api call (otherwise GPT-4 can choke)
        num_batches = int(np.ceil(num_claims / self.max_claims_per_api_call))
        claims_and_sources_batches = [
            batch.tolist() for batch in np.array_split(claims_and_sources, num_batches)
        ]

        kwarg_list = []
        for idx, claims_and_sources_batch in enumerate(claims_and_sources_batches):
            # remove newlines from the passage to avoid a confusing prompt format
            kwarg_list.append(
                {
                    "idx": idx,
                    "total": len(claims_and_sources_batches),
                    "claims_and_sources_batch": claims_and_sources_batch,
                }
            )

        if self.processes == 1:
            batch_results = []
            for kwargs in kwarg_list:
                batch_results.append(self.classify_claim_batch(**kwargs))
        else:  # multiprocess
            func = self.classify_claim_batch
            with mp.Pool(processes=self.processes) as pool:
                batch_results = starmap_with_kwargs(
                    pool=pool, func=func, kwargs_iter=kwarg_list
                )

        cost = sum([result["cost"] for result in batch_results])
        labelled_claims = []
        for batch in batch_results:
            labelled_claims.extend(batch["claims_with_labels"])

        print(f"Returning {len(labelled_claims)} claims (cost: {cost} USD)")
        return labelled_claims

    def filter_to_objective_claims(self, claims):
        """Filter claims to only those that are objective."""

        objective_claims = [claim for claim in claims if claim["label"] == "objective"]

        print(f"Returning {len(objective_claims)} objective claims")
        return objective_claims
