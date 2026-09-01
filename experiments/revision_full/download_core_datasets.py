"""Download the two datasets required before the revision-v2 preflight."""

from datasets import load_dataset


def main() -> None:
    requests = [
        ("openai/gsm8k", "main", ("train", "test")),
        ("wikitext", "wikitext-2-raw-v1", ("train",)),
    ]
    for dataset, config, splits in requests:
        for split in splits:
            loaded = load_dataset(dataset, config, split=split)
            print(f"cached {dataset}/{config}/{split}: {len(loaded)} rows", flush=True)


if __name__ == "__main__":
    main()
