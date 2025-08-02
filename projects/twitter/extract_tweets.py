import argparse
import json

from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run char-level GPT generation")

    parser.add_argument(
        "--tweets",
        type=Path,
        default="tweets.js",
        help="Path to input text file containing tweets for training dataset",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default="input.txt",
        help="Path to output text file for generated tweets",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    prefix = "window.YTD.tweets.part0 = "
    _tweet_data = Path(args.tweets).read_text()[len(prefix) :]
    tweet_data = json.loads(_tweet_data)

    output_path = Path(args.output)

    results = []

    for entry in tweet_data:
        tweet = entry.get("tweet", {})

        full_text = tweet.get("full_text", "")

        if full_text.startswith("RT @"):
            username, content = full_text[4:].split(":", 1)
            results.append(f"@{username}:\n{content.strip()}\n\n")
        elif full_text.startswith("@"):
            username, content = full_text[1:].split(" ", 1)
            full_text = f"{content.strip()}\n\n"
        else:
            full_text = full_text.strip()
            results.append(f"{full_text}\n\n")

    output_path.write_text("\n".join(results))


if __name__ == "__main__":
    main()
