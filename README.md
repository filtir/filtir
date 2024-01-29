# Filtir - fact checking pipeline

This repo contains the Filtir pipeline for claim extraction and fact-checking.

## Prerequisites

### Create and prepare venv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements
```

### Setup keys
In order to run the code you need to set up the following keys and add them to .env:

- OPENAI_API_KEY - used to call the OpenAI API
- COHERE_API_KEY and WEAVIATE_API_KEY - used for Wikipedia search
- GOOGLE_CLOUD_API_KEY and GOOGLE_CUSTOM_SEARCH_ENGINE_ID - used for Google search

## Run the pipeline

```bash
python run_pipeline.py --file <input-file> --model <model>
```

The `<input-file>` contains the text that needs to be fact-checked while `<model>` represents the OpenAI model used in the pipeline.

## Step-by-step usage

### Step 1: Claim extraction

We extract claims from a folder of text files.

```bash
python step1_api_claim_extractor.py
```

### Step 2: Fix passage anchors

Sometimes the first step does not correctly quote passages to anchor the claims on.

```bash
python step2_api_fix_passage_anchors.py
```

### Step 3: classify each claim as objective or subjective

This script classifies each claim as being either objective or subjective.
Objective claims are copied into a new directory.

```bash
python step3_api_identify_objective_claims.py
```

### Step 4: Fetch evidence

We fetch evidence by:

- Searching the claims against wikipedia (with semantic search on 1M cohere embeddings)
- Searching google

```bash
python step41_api_fetch_cohere_wikipedia_evidence.py
python step42_api_fetch_google_search_evidence.py
```

### Step 5: Embed the retrieved documents

```bash
python step5_api_embed_search_results.py
```

### Step 6: Link the claims to supporting evidence

```bash
python step6_api_link_claims_to_evidence.py
```

### Step 7: Evaluate the evidence in support of the claim

```bash
python step7_api_check_claims_against_evidence.py --processes 10
```

### Step 8: Insert fact-checking results into the original text

```bash
python step8_api_format_fact_checked_document.py
```
