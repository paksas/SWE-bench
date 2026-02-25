#!/usr/bin/env python3

"""Runs inference on a dataset by calling a custom API endpoint.

The API is expected to expose a POST /run endpoint that accepts
{"user_message": str, "skills_dir": str | None} and returns
{"text": str, "error_code": int}.

Outputs are written incrementally to a JSONL file so the script can be
stopped and restarted without losing progress.
"""

import json
import os
import traceback
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import requests
from datasets import load_dataset, load_from_disk, Dataset  # type: ignore
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm.auto import tqdm

from swebench.inference.make_datasets.utils import extract_diff

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _make_call_custom_api(max_retries: int):
    """Build a retry-decorated API caller with the given max retries."""

    @retry(
        wait=wait_random_exponential(min=30, max=300),
        stop=stop_after_attempt(max_retries),
    )
    def call_custom_api(
        api_url: str,
        user_message: str,
        skills_dir: str | None,
        timeout: int,
    ) -> str | None:
        """POST to {api_url}/run and return the response text.

        Returns None when the API signals an error via error_code != 0.
        Raises on HTTP / connection errors (handled by tenacity retry).
        """
        url = f"{api_url.rstrip('/')}/run"
        payload = {
            "user_message": user_message,
            "skills_dir": skills_dir,
        }
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()

        error_code = body.get("error_code", -1)
        if error_code != 0:
            logger.warning(
                "API returned error_code=%d for request (message truncated to 200 chars): %s",
                error_code,
                user_message[:200],
            )
            return None

        return body["text"]

    return call_custom_api


def custom_api_inference(
    test_dataset,
    api_url: str,
    skills_dir: str | None,
    output_file: Path,
    existing_ids: set,
    timeout: int,
    retries: int,
):
    call_api = _make_call_custom_api(retries)
    logger.info(
        "Running inference against %s (%d instances)", api_url, len(test_dataset)
    )

    with open(str(output_file), "a+") as f:
        for datum in tqdm(test_dataset, desc=f"Inference via {api_url}"):
            instance_id = datum["instance_id"]
            if instance_id in existing_ids:
                continue

            try:
                response_text = call_api(
                    api_url=api_url,
                    user_message=datum["text"],
                    skills_dir=skills_dir,
                    timeout=timeout,
                )
            except Exception as e:
                logger.error("Failed for %s: %s", instance_id, e)
                traceback.print_exc()
                continue

            if response_text is None:
                logger.warning("Skipping %s (API returned error)", instance_id)
                continue

            output_dict = {
                "instance_id": instance_id,
                "full_output": response_text,
                "model_patch": extract_diff(response_text),
            }
            print(json.dumps(output_dict), file=f, flush=True)


def main(
    dataset_name_or_path: str,
    split: str,
    api_url: str,
    skills_dir: str | None,
    model_name: str,
    shard_id: int | None,
    num_shards: int | None,
    output_dir: str,
    timeout: int,
    retries: int,
):
    if shard_id is None and num_shards is not None:
        logger.warning(
            "Received num_shards=%d but shard_id is None, ignoring", num_shards
        )
    if shard_id is not None and num_shards is None:
        logger.warning(
            "Received shard_id=%d but num_shards is None, ignoring", shard_id
        )

    # Build output path
    output_file_str = f"{model_name}__{dataset_name_or_path.split('/')[-1]}__{split}"
    if shard_id is not None and num_shards is not None:
        output_file_str += f"__shard-{shard_id}__num_shards-{num_shards}"
    output_file = Path(output_dir) / (output_file_str + ".jsonl")
    logger.info("Will write to %s", str(output_file))

    # Collect already-processed ids for resumability
    existing_ids: set[str] = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                existing_ids.add(data["instance_id"])
    logger.info("Read %d already completed ids from %s", len(existing_ids), output_file)

    # Load dataset
    if Path(dataset_name_or_path).exists():
        dataset_or_dict = load_from_disk(dataset_name_or_path)
        if isinstance(dataset_or_dict, Dataset):
            dataset_dict = {split: dataset_or_dict}
        else:
            dataset_dict = dataset_or_dict
    else:
        dataset_dict = load_dataset(dataset_name_or_path)
    if split not in dataset_dict:
        raise ValueError(f"Invalid split {split} for dataset {dataset_name_or_path}")
    dataset = dataset_dict[split]

    # Sort by length for efficiency
    lens = np.array(list(map(len, dataset["text"])))
    dataset = dataset.select(np.argsort(lens))

    # Filter out already-processed instances
    if len(existing_ids) > 0:
        dataset = dataset.filter(
            lambda x: x["instance_id"] not in existing_ids,
            desc="Filtering out existing ids",
            load_from_cache_file=False,
        )

    # Shard if requested
    if shard_id is not None and num_shards is not None:
        dataset = dataset.shard(num_shards, shard_id, contiguous=True)

    custom_api_inference(
        test_dataset=dataset,
        api_url=api_url,
        skills_dir=skills_dir,
        output_file=output_file,
        existing_ids=existing_ids,
        timeout=timeout,
        retries=retries,
    )
    logger.info("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        required=True,
        help="Base URL of the custom API (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--skills_dir",
        type=str,
        default=None,
        help="Value to send as skills_dir in the API request",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="custom_api",
        help="Label used in output file naming and JSONL records",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="Shard id to process. If None, process all shards.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Number of shards. If None, process all shards.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts per instance",
    )
    args = parser.parse_args()
    main(**vars(args))
