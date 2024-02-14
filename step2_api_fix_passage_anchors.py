import argparse
import json
import multiprocessing as mp
from zsvision.zs_multiproc import starmap_with_kwargs
from zsvision.zs_utils import BlockTimer
from text_utils import is_unique_verbatim_quote, parse_passage_quote_and_claim
from llm_api_utils import (
    call_openai_with_exponetial_backoff,
    estimate_cost_of_text_generation_api_call,
    init_openai_with_api_key,
)


class FixAnchors:
    def __init__(
        self,
        temperature=0,
        model="gpt-3.5-turbo",
        filter_str="",
        processes=8,
        refresh=False,
    ):
        self.temperature = temperature
        self.model = model
        self.filter_str = filter_str
        self.processes = processes
        self.refresh = refresh

    def fix_passage_anchor(
        self,
        idx: int,
        total: int,
        original_passage: str,
        claim_with_metadata: dict,
    ):
        init_openai_with_api_key()
        print(f"Processing claim with metadata {idx + 1} of {total}")
        # we remove newlines
        original_passage = original_passage.replace("\n", " ")
        assert not claim_with_metadata[
            "is_unique_and_verbatim"
        ], "We should only fix broken passage anchors"

        prompt = f"""\
Task:
A co-worker was tasked with identifying a unique, verbatim quote from a passage that underpins a particular claim. \
Unfortunately, they made a mistake and the quote they identified is not unique and verbatim. \
Your task is to fix their quote so that it is both verbatim and unique.

-----
Here is an example passage, together with the claim and the erroneous quote.

Passage:
In 1940, she was interned in a French camp as an enemy alien, but managed to escape and eventually make her way to the United States in 1941.  \
Arendt's experiences during this time would deeply influence her work on totalitarianism and human rights. \
In New York, she began to immerse herself in academic life, working as an editor, journalist, and lecturer. \
Her first major work, *The Origins of Totalitarianism*, published in 1951, explored the common roots of Nazism and Stalinism, and established her as a significant voice in political philosophy. \
## A Life Of Controversial, Influential Works  \
Throughout her career, Arendt wrote a number of seminal, and controversial, works. *The Human Condition* (1958) examined the role of politics in modern societies and introduced the concept of "the public realm" – the space where individuals act and participate in political life. \
This exploration of freedom and action would become a recurring theme in her writings.  \
Her 1963 publication, *Eichmann in Jerusalem: A Report on the Banality of Evil*, based on her coverage of Adolf Eichmann's trial, ignited significant controversy. \
Arendt argued that Eichmann, a key architect of the Holocaust, was not a monster but rather an ordinary bureaucrat who unquestioningly followed orders. \
The idea of the "banality of evil" continues to influence discussions on the nature of evil and moral responsibility.  \
Arendt's later works, such as *On Revolution* (1963) and *Between Past and Future* (1968), sought to further unravel the complexities of power, authority, and rebellion. \
Her writings on these subjects continue to resonate with present-day political struggles, as well as with the works of other philosophers like [Immanuel Kant](/philosophy/2023-immanuel-kant-life-and-work) and [Edmund Husserl](/philosophy/2023-edmund-husserl-his-life-and-work).  \

Claim:
*The Origins of Totalitarianism* established Arendt as a significant voice in political philosophy.

Initial attempt at a unique and verbatim quote:
[The Origins of Totalitarianism] established her as a significant voice in political philosophy.

Correct (unique and verbatim) quote:
Her first major work, *The Origins of Totalitarianism*, published in 1951, explored the common roots of Nazism and Stalinism, and established her as a significant voice in political philosophy.
-----
Passage:
{original_passage}

Claim:
{claim_with_metadata["claim"]}

Initial attempt at a unique verbatim quote:
{claim_with_metadata["verbatim_quote"]}

Correct (unique and verbatim) quote:\
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
        verbatim_quote = content.rstrip()
        is_unique_and_verbatim = is_unique_verbatim_quote(
            verbatim_quote=verbatim_quote, original_passage=original_passage
        )
        assert (
            is_unique_and_verbatim
        ), f"Failed to fix passage anchor: {claim_with_metadata['verbatim_quote']} was updated to {verbatim_quote} but is not unique and verbatim"

        claim_with_metadata["verbatim_quote"] = verbatim_quote
        return {"claim_with_metadata": claim_with_metadata, "cost": cost}

    def fix_passage_anchors(self, claims_with_metadata, original_passage: str):
        kwarg_list = []
        valid_claims_with_metadata = []
        invalid_claims_with_metadata = []
        for idx, claim_with_metadata in enumerate(claims_with_metadata):
            # remove newlines from the passage to avoid a confusing prompt format
            if not claim_with_metadata["is_unique_and_verbatim"]:
                invalid_claims_with_metadata.append(claim_with_metadata)
            else:
                valid_claims_with_metadata.append(claim_with_metadata)

        for idx, claim_with_metadata in enumerate(invalid_claims_with_metadata):
            kwarg_list.append(
                {
                    "idx": idx,
                    "total": len(invalid_claims_with_metadata),
                    "claim_with_metadata": claim_with_metadata,
                    "original_passage": original_passage,
                }
            )

        if self.processes == 1:
            results = []
            for kwargs in kwarg_list:
                try:
                    results.append(self.fix_passage_anchor(**kwargs))
                except Exception as e:
                    print(f"Exception in step2: {e}, model: {self.model}")
                    print("Skipping this claim!")
                    if self.model == "gpt-4":
                        pass
                    else:
                        raise e
        else:  # multiprocess
            func = self.fix_passage_anchor
            with mp.Pool(processes=self.processes) as pool:
                results = starmap_with_kwargs(
                    pool=pool, func=func, kwargs_iter=kwarg_list
                )

        cost = sum([result["cost"] for result in results])
        for result in results:
            valid_claims_with_metadata.append(result["claim_with_metadata"])

        # remove the is_unique_and_verbatim field (no longer needed)
        for claim_with_metadata in valid_claims_with_metadata:
            del claim_with_metadata["is_unique_and_verbatim"]

        print(
            f"Returning {len(valid_claims_with_metadata)} claims with metadat (cost: {cost} USD)"
        )
        return valid_claims_with_metadata
