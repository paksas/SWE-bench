DATASET_NAME := princeton-nlp/SWE-bench
DATASET_DIR  := ./datasets
SPLIT        := test
API_URL      := http://localhost:5301
MODEL_NAME   := pavo
PREDICTIONS  := ./outputs-pavo/predictions.jsonl
RUN_ID       := pavo-eval

# Step 1: Prepare the dataset (creates the "text" column)
prepare-dataset:
	uv run python -m swebench.inference.make_datasets.create_text_dataset \
		--dataset_name_or_path $(DATASET_NAME) \
		--output_dir $(DATASET_DIR) \
		--prompt_style style-3 \
		--file_source oracle \
		--splits $(SPLIT)

# Step 2: Run inference via custom API
generate-pavo-predictions: 
	uv run python -m swebench.inference.run_custom_api \
		--dataset_name_or_path $(DATASET_DIR)/SWE-bench__style-3__fs-oracle \
		--split $(SPLIT) \
		--api_url $(API_URL) \
		--model_name $(MODEL_NAME) \
		--predictions_path $(PREDICTIONS)

# Step 3: Evaluate predictions
eval-pavo:
	uv run python -m swebench.harness.run_evaluation \
		--dataset_name $(DATASET_NAME) \
		--split $(SPLIT) \
		--predictions_path $(PREDICTIONS) \
		--run_id $(RUN_ID) \
		--max_workers 4
