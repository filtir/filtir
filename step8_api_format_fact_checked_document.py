import argparse
import re
from collections import defaultdict
import json
from text_utils import find_matching_indices
from pathlib import Path
from pipeline_paths import PIPELINE_PATHS


class FormatDocument:
    def __init__(
        self,
        footnote_style: str,
        temperature=0.0,
        model="gpt-4",
        dest_dir=Path("data/extracted_claims"),
        filter_str="",
        refresh=False,
    ):
        self.temperature = temperature
        self.model = model
        self.dest_dir = dest_dir
        self.filter_str = filter_str
        self.refresh = refresh
        self.footnote_style = footnote_style

    def cleanup_explanation(self, claim_assessment: dict, mode: str) -> str:
        claim = claim_assessment["claim"]
        assessment = claim_assessment["assessment"]
        justification = assessment["justification"]
        category = assessment["verdict"]
        urls = assessment["URLs"]
        date_accessed = assessment["date_accessed"]

        prefixes = {
            "Fully supported": "✅",
            "Partially supported": "❓",
            "Unsupported": "❗",
        }
        prefix = prefixes[category]
        quotes = ",".join(f'"{quote}"' for quote in assessment["quotes"])
        # Sometimes, the verdict justification contains newlines , which messes up the formatting of footnotes.
        justification = justification.replace("\n", "")

        if mode == "terse":
            footnote = f"Claim: {claim} 👉 {category} {urls}"
        elif mode == "verbose":
            footnote = f"Claim: {claim} 👉 {category} {quotes} {justification}, URLs: {urls}, date accessed: {date_accessed}"
        footnote = f"{prefix} {footnote}"
        return footnote

    def reformat_document_to_include_claims(
        self,
        original_text,
        fact_verdicts,
        footnote_style=None,
    ):
        bibliography = []
        footnote_markers_to_insert = []
        statistics = defaultdict(int)
        number_of_facts_checked = 0
        if footnote_style:
            self.footnote_style = footnote_style
        for fact_idx, claim_assessment in enumerate(fact_verdicts):
            if self.footnote_style == "terse":
                footnote_str = f"{fact_idx + 1}"
            elif self.footnote_style == "verbose":
                footnote_str = claim_assessment["claim"].replace(" ", "-")
                # footnote markers cannot contain much punctuation or commas in Jekyll
                # (even though this is valid in GitHub-flavoured markdown)
                for char in [
                    ",",
                    ".",
                    '"',
                    "'",
                    ":",
                    ";",
                    "(",
                    ")",
                    "[",
                    "]",
                    "{",
                    "}",
                    "*",
                ]:
                    footnote_str = footnote_str.replace(char, "")

            explanation = self.cleanup_explanation(
                claim_assessment, mode=self.footnote_style
            )
            footnote_marker = f"[^{footnote_str}]"
            query = claim_assessment["verbatim_quote"]

            assert (
                original_text.count(query) == 1
            ), f"Found {original_text.count(query)} matches for {query}, rather than 1"
            start_pos = original_text.find(query)
            assert start_pos != -1, f"Could not find {query} in {original_text}"
            end_pos = start_pos + len(query)
            footnote_markers_to_insert.append((end_pos, footnote_marker))
            verdict_category = claim_assessment["assessment"]["verdict"]
            statistics[verdict_category] += 1
            number_of_facts_checked += 1
            bibliography.append(f"{footnote_marker}: {explanation} ")

        # perform insertions in reverse order so that the indices don't get messed up
        modified_text = original_text
        for char_pos, footnote_marker in sorted(
            footnote_markers_to_insert, reverse=True
        ):
            modified_text = (
                modified_text[:char_pos] + footnote_marker + modified_text[char_pos:]
            )

        modified_text += "\n\n"
        modified_text += "\n".join(bibliography)

        # assert number_of_facts_checked != 0, "No facts were checked"
        if number_of_facts_checked == 0:
            print("No objective facts were found.")
            modified_text = "No clear-cut objective claims were detected."
        return modified_text
