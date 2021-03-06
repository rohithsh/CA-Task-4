import argparse
import json
import pandas as pd
import rouge

from os import path


def main():
    # Read train-test-splits and get test ids
    splits = pd.read_csv(SPLITS_PATH, sep=";")
    test_ids = sorted([int(fn[-3:]) for fn in splits[splits.SET == "TEST"].ID.values])

    #  Read data files from disk
    with open(ESSAYS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # Extract prediction labels
    # Also, make sure they are sorted based on the integer value of their id
    predictions = sorted(predictions, key=lambda x: int(x["id"]))
    predicted_texts = [p["prompt"] for p in predictions]

    # Extract true labels
    # Also, make sure they are sorted based on their id
    test_split = list(filter(lambda x: x["id"] in test_ids, data))
    test_split = sorted(test_split, key=lambda x: x["id"])
    true_prompts = [p['prompt'] for p in test_split]

    # Instantiate ROUGE evaluator
    evaluator = rouge.Rouge(metrics=['rouge-l'])

    print(len(true_prompts))
    print(len(predicted_texts))
    # Calculate the scores and print them nicely
    scores = evaluator.get_scores(predicted_texts, true_prompts, avg=True)
    print(scores)


if __name__ == "__main__":
    # Parse cli arguments
    SCRIPT_DESCRIPTION = "Simple script to evaluate the prediction output"
    parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)
    parser.add_argument(
        "--corpus",
        "-c",
        type=str,
        required=True,
        help="Path to the corpus file.",
        metavar="CORPUS")
    parser.add_argument(
        "--predictions",
        "-p",
        type=str,
        required=True,
        help="Path to the predictions file.",
        metavar="PREDICTIONS")
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        required=True,
        help="Path to the train-test split definition file.",
        metavar="SPLIT")
    args = parser.parse_args()

    # Define all data files paths
    PREDICTIONS_PATH = args.predictions
    SPLITS_PATH = args.split
    ESSAYS_PATH = args.corpus

    main()
