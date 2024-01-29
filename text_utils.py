import re
from typing import Dict
import unittest


def parse_passage_quote_and_claim(passage_quote_and_claim: str) -> Dict[str, str]:
    """Parse the quote and claim from a string, where the string is of the form:

    [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] <passage quote for claim> [CLAIM] <claim>
    """

    if not passage_quote_and_claim.startswith("[VERBATIM_PASSAGE_QUOTE_FOR_CLAIM]"):
        raise ValueError(f"Invalid input format: {passage_quote_and_claim}")

    parts = passage_quote_and_claim.split("[VERBATIM_PASSAGE_QUOTE_FOR_CLAIM]")
    source_parts = parts[1].split("[CLAIM]")

    # If there aren't exactly two parts after splitting by [CLAIM], the format is invalid
    if len(source_parts) != 2:
        raise ValueError(f"Invalid input format: {passage_quote_and_claim}")

    passage_quote_for_claim = source_parts[0].strip()
    claim = source_parts[1].strip()
    return {"verbatim_quote": passage_quote_for_claim, "claim": claim}


def is_unique_verbatim_quote(verbatim_quote: str, original_passage: str):
    """Check if the verbatim quote is an exact quote from the original passage."""
    return original_passage.count(verbatim_quote) == 1


def find_matching_indices(query: str, original_text: str):
    # Function to remove markdown links and create an index map
    def remove_links(text):
        index_map = []
        result = []
        markdown_links = re.finditer(r"\[([^\]]+)\]\([^)]+\)", text)

        prev_end = 0
        for match in markdown_links:
            result.append(text[prev_end : match.start()])
            index_map.extend(range(prev_end, match.start()))
            result.append(match.group(1))
            index_map.extend(range(match.start(1), match.end(1)))
            prev_end = match.end()

        result.append(text[prev_end:])
        index_map.extend(range(prev_end, len(text)))

        return "".join(result), index_map

    # Remove markdown links from the original text and create an index map
    cleaned_text, index_map = remove_links(original_text)

    # Remove markdown links from the query
    cleaned_query, _ = remove_links(query)

    # Find the start index of the cleaned query in the cleaned text
    start = cleaned_text.find(cleaned_query)

    # If the query is not found, return an empty list
    if start == -1:
        return []

    # Add the query length to get the end index
    end = start + len(cleaned_query)

    # Use the index map to find the corresponding start and end indices in the original text
    original_start = index_map[start]
    original_end = index_map[end - 1] + 1

    return [(original_start, original_end)]


# def find_matching_indices(query: str, original_text: str):
#     # Function to remove markdown links and create an index map
#     def remove_links(text):
#         index_map = []
#         result = []
#         markdown_links = re.finditer(r'\[([^\]]+)\]\([^)]+\)', text)

#         prev_end = 0
#         for match in markdown_links:
#             result.append(text[prev_end:match.start()])
#             index_map.extend(range(prev_end, match.start()))
#             result.append(match.group(1))
#             index_map.extend(range(match.start(1), match.end(1)))
#             prev_end = match.end()

#         result.append(text[prev_end:])
#         index_map.extend(range(prev_end, len(text)))

#         return ''.join(result), index_map

#     # Remove markdown links from the original text and create an index map
#     cleaned_text, index_map = remove_links(original_text)

#     # Find the start index of the query in the cleaned text
#     start = cleaned_text.find(query)

#     # If the query is not found, return an empty list
#     if start == -1:
#         return []

#     # Add the query length to get the end index
#     end = start + len(query)

#     # Use the index map to find the corresponding start and end indices in the original text
#     original_start = index_map[start]
#     original_end = index_map[end - 1] + 1

#     return [(original_start, original_end)]


class TestCases(unittest.TestCase):
    def test_find_matching_indices(self):
        """Test the find_matching_indices() function.
        This function should return a list of matches, where each match is a tuple of (start, end) indices.

        The start and end indices should be the character positions of the query in the original_text, accounting
        for the fact that markdown links should be ignored when performing the match.

        """
        test_cases = [
            {
                "query": "Her writings on these subjects continue to resonate with present-day political struggles, as well as with the works of other philosophers like Immanuel Kant and Edmund Husserl.",
                "original": "Arendt's later works, sought to further unravel the complexities of power and rebellion. Her writings on these subjects continue to resonate with present-day political struggles, as well as with the works of other philosophers like [Immanuel Kant](/philosophy/2023-immanuel-kant-life-and-work) and [Edmund Husserl](/philosophy/2023-edmund-husserl-his-life-and-work).\n\n## A Lasting Legacy",
                "expected": "Her writings on these subjects continue to resonate with present-day political struggles, as well as with the works of other philosophers like [Immanuel Kant](/philosophy/2023-immanuel-kant-life-and-work) and [Edmund Husserl](/philosophy/2023-edmund-husserl-his-life-and-work).",
            },
            {
                "query": "I went to the sea side (at the weekend).",
                "original": "I woke up. Then I went to the sea side (at the weekend). Then I went home.",
                "expected": "I went to the sea side (at the weekend).",
            },
            {
                "query": "no merger with the [solar farm] company",
                "original": "There would be no merger with the [solar farm] company.",
                "expected": "no merger with the [solar farm] company",
            },
            {
                "query": "with the works of other philosophers like [Immanuel Kant](/philosophy/2023-immanuel-kant-life-and-work) and [Edmund Husserl](/philosophy/2023-edmund-husserl-his-life-and-work)",
                "original": "\n\n## Fleeing Germany and the Road to Academia\n\nWith the rise of the Nazi regime in the 1930s, Arendt's Jewish heritage put her in grave danger. She fled Germany in 1933 and settled in Paris, where she became involved with a number of political and social organizations advocating for Jewish refugees. In 1940, she was interned in a French camp as an enemy alien, but managed to escape and eventually make her way to the United States in 1941.\n\nArendt's experiences during this time would deeply influence her work on totalitarianism and human rights. In New York, she began to immerse herself in academic life, working as an editor, journalist, and lecturer. Her first major work, *The Origins of Totalitarianism*, published in 1951, explored the common roots of Nazism and Stalinism, and established her as a significant voice in political philosophy.\n\n## A Life Of Controversial, Influential Works\n\nThroughout her career, Arendt wrote a number of seminal, and controversial, works. *The Human Condition* (1958) examined the role of politics in modern societies and introduced the concept of \"the public realm\" – the space where individuals act and participate in political life. This exploration of freedom and action would become a recurring theme in her writings.\n\nHer 1963 publication, *Eichmann in Jerusalem: A Report on the Banality of Evil*, based on her coverage of Adolf Eichmann's trial, ignited significant controversy. Arendt argued that Eichmann, a key architect of the Holocaust, was not a monster but rather an ordinary bureaucrat who unquestioningly followed orders. The idea of the \"banality of evil\" continues to influence discussions on the nature of evil and moral responsibility.\n\nArendt's later works, such as *On Revolution* (1963) and *Between Past and Future* (1968), sought to further unravel the complexities of power, authority, and rebellion. Her writings on these subjects continue to resonate with present-day political struggles, as well as with the works of other philosophers like [Immanuel Kant](/philosophy/2023-immanuel-kant-life-and-work) and [Edmund Husserl](/philosophy/2023-edmund-husserl-his-life-and-work).\n\n## A Lasting Legacy\n\nHannah Arendt died in 1975, but her work remains as relevant as ever.",
                "expected": "with the works of other philosophers like [Immanuel Kant](/philosophy/2023-immanuel-kant-life-and-work) and [Edmund Husserl](/philosophy/2023-edmund-husserl-his-life-and-work)",
            },
        ]

        for test_case in test_cases:
            matches = find_matching_indices(
                query=test_case["query"], original_text=test_case["original"]
            )
            assert (
                len(matches) == 1
            ), f"Expected exactly one match, but found {len(matches)}"
            result = test_case["original"][matches[0][0] : matches[0][1]]
            msg = (
                f"Expected\n\n{test_case['expected']}\n\nbut instead found\n\n{result}"
            )
            self.assertEqual(result, test_case["expected"], msg)
        print(f"Passed all tests for find_matching_indices()")

    def test_parse_passage_quote_and_claim(self):
        """Test the following function:
            parse_passage_quote_and_claim(passage_quote_and_claim: str) -> {"verbatim_quote": str, "claim": str}

        The passage quote and claim should take the form:
        [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] <passage quote for claim> [CLAIM] <claim>
        """
        test_cases = [
            {
                "passage_quote_and_claim": "[VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] Hannah Arendt [was born in] 1906 in Linden, Germany [CLAIM] Hannah Arendt was born in Linden, Germany.",
                "expected": {
                    "verbatim_quote": "Hannah Arendt [was born in] 1906 in Linden, Germany",
                    "claim": "Hannah Arendt was born in Linden, Germany.",
                },
            },
            {
                "passage_quote_and_claim": "Something [VERBATIM_PASSAGE_QUOTE_FOR_CLAIM] Hannah Arendt [was born in] 1906 in Linden, Germany [CLAIM] Hannah Arendt was born in Linden, Germany.",
                "expected": "Exception",
            },
        ]
        for test_case in test_cases:
            expected = test_case["expected"]
            if expected == "Exception":
                self.assertRaises(
                    ValueError,
                    parse_passage_quote_and_claim,
                    test_case["passage_quote_and_claim"],
                )
            else:
                parsed = parse_passage_quote_and_claim(
                    passage_quote_and_claim=test_case["passage_quote_and_claim"]
                )
                self.assertEqual(parsed["verbatim_quote"], expected["verbatim_quote"])

    def test_is_unique_verbatim_quote_check(self):
        """Test the following function:
            is_unique_verbatim_quote_check(verbatim_quote: str) -> bool

        This function should return True if the verbatim quote is indeed a quote and is unique, and false otherwise.

        """
        test_cases = [
            {
                "verbatim_quote": "Hannah Arendt [was born in] 1906 in Linden, Germany",
                "original_passage": "Hannah Arendt [was born in] 1906 in Linden, Germany at a time when...",
                "expected": True,
            },
            {
                "verbatim_quote": "Hannah Arendt [was born in] 1906 in Linden, Germany",
                "original_passage": "Hannah Arendt [wasn't born in] 1906 in Linden, Germany at a time when...",
                "expected": False,
            },
            {
                "verbatim_quote": "Hannah Arendt [was born in] 1906 in Linden, Germany. Hannah Arendt was a person.",
                "original_passage": "Hannah Arendt",
                "expected": False,
            },
        ]
        for test_case in test_cases:
            result = is_unique_verbatim_quote(
                verbatim_quote=test_case["verbatim_quote"],
                original_passage=test_case["original_passage"],
            )
            self.assertEqual(result, test_case["expected"])


if __name__ == "__main__":
    unittest.main()
