import os
import sys
import argparse

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN


def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = "./out/chargpt"

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = "gpt-mini"

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = (
        5e-4  # the model we're using is so small that we can go a bit faster
    )

    return C


class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.config.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


def parse_args():
    parser = argparse.ArgumentParser(description="Run char-level GPT generation")

    parser.add_argument(
        "--prompt",
        type=str,
        default="O God, O God!",
        help="Prompt string to start generation",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="out/model.pt",
        help="Path to model checkpoint file (default: out/model.pt)",
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default="input.txt",
        help="Path to input text file for training dataset",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # Load training text
    text = open(args.input_file, "r").read()
    train_dataset = CharDataset(config.data, text)

    # Setup model config
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()

    model = GPT(config.model)

    # Determine checkpoint path
    ckpt_path = (
        args.checkpoint
        if args.checkpoint is not None
        else os.path.join(config.system.work_dir, "model.pt")
    )
    if os.path.exists(ckpt_path):
        print(f"loading model from {ckpt_path}")
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model.to("cuda")
        print("Model moved to GPU.")

    model.eval()

    with torch.no_grad():
        context = args.prompt

        # Convert the context string to a tensor of token indices
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[
            None, ...
        ].to(device)

        # Generate a sequence of 500 characters
        y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]

        # Convert the generated token indices back to a string
        characters = [train_dataset.itos[int(i)] for i in y]
        completion = "".join(characters)

        print(completion)
