import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run char-level GPT generation")

    parser.add_argument(
        "--tweets",
        type=str,
        default="tweets.txt",
        help="Path to input text file containing tweets for training dataset",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.txt",
        help="Path to output text file for generated tweets",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)


if __name__ == "__main__":
    main()
