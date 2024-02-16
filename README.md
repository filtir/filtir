---
title: Filtir
app_file: app.py
sdk: gradio
sdk_version: 4.19.0
---
# Filtir - fact checking pipeline

This repo contains the Filtir pipeline for claim extraction and fact-checking.

## Prerequisites

### Create and prepare venv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Setup keys
In order to run the code you need to set up the following keys and add them to .env:

- OPENAI_API_KEY - used to call the OpenAI API
- COHERE_API_KEY and WEAVIATE_API_KEY - used for Wikipedia search
- GOOGLE_CLOUD_API_KEY and GOOGLE_CUSTOM_SEARCH_ENGINE_ID - used for Google search

## Run the pipeline

```bash
python run_pipeline.py --file example.txt --model gpt-4-1106-preview
```

## Run Gradio app locally

```bash
python app.py
```

## Demo
Demo available [here](https://huggingface.co/spaces/vladbogo/Filtir)