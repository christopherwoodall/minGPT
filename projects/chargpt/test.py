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
        # print([repr(c) for c in chars])

        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
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
        default="./out/chargpt/model.pt",
        help="Path to model checkpoint file (default: out/model.pt)",
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default="input.txt",
        help="Path to input text file for training dataset",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (default: cuda if available, else cpu)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # get default config and overrides from the command line, if any
    config = get_config()
    # config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # Set the device
    device = args.device
    # device = torch.device(args.device)

    # Set the context for generation
    context = args.prompt

    # construct the training dataset
    input_file = args.input_file if args.input_file else "input.txt"
    text = open(input_file, "r").read()
    train_dataset = CharDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()

    # Load the model
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
        # Load the state dictionary into the model
        model.load_state_dict(state_dict)

    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
        print("Model moved to GPU.")

    model.eval()

    with torch.no_grad():
        # Convert the context string to a tensor of token indices - on the same device as the model
        x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[
            None, ...
        ].to(device)

        # Generate a sequence of 500 characters
        y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]

        # Convert the generated token indices back to a string
        characters = [train_dataset.itos[int(i)].strip("'\"") for i in y]

        completion = "".join(characters)

        # Print the completed text
        print(completion)


if __name__ == "__main__":
    main()
