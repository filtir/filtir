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
            print("!!!!NO facts!!!", original_text)
            modified_text = "No clear-cut objective claims were detected."
        print(f"Returned fact-checked document [statistics: {statistics}]")
        return modified_text


def main():
    args = parse_args()
    format_document = FormatDocument(
        temperature=args.temperature,
        model=args.model,
        dest_dir=args.dest_dir,
        filter_str=args.filter_str,
        refresh=args.refresh,
        footnote_style=args.footnote_style,
    )

    src_dir = PIPELINE_PATHS["source_document_dir"]
    src_paths = list(reversed(list(src_dir.glob("*.txt"))))
    print(f"Found {len(src_paths)} documents in {src_dir}")

    if args.filter_str:
        num_paths = len(src_paths)
        src_paths = [
            src_path for src_path in src_paths if args.filter_str in src_path.name
        ]
        print(f"Filtering for {args.filter_str} (from {num_paths} to {len(src_paths)})")

    for src_path in src_paths:
        rel_path = src_path.relative_to(src_dir)
        fact_verdicts_path = PIPELINE_PATHS[
            "evaluated_claims_dir"
        ] / rel_path.with_suffix(".json")
        dest_path = PIPELINE_PATHS["fact_checked_document_dir"] / rel_path.with_suffix(
            ".md"
        )

        with open(src_path) as f:
            original_text = f.read()
        with open(fact_verdicts_path) as f:
            fact_verdicts = json.load(f)

        if not dest_path.exists() or args.refresh:
            modified_text = format_document.reformat_document_to_include_claims(
                original_text=original_text,
                fact_verdicts=fact_verdicts,
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "w") as f:
                f.write(modified_text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--model", default="gpt-4", choices=["gpt-4", "gpt-3.5-turbo"])
    parser.add_argument("--dest_dir", default="data/extracted_claims", type=Path)
    parser.add_argument("--api_key_path", default="OPENAI_API_KEY.txt")
    parser.add_argument(
        "--footnote_style", default="terse", choices=["terse", "verbose"]
    )
    parser.add_argument("--filter_str", default="")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
