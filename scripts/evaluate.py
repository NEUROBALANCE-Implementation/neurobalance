from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from neurobalance.data.vqa_datasets import ToyVQADataset
from neurobalance.data.collators import toy_vqa_collate
from neurobalance.models.llava_next_wrapper import LlavaNextMiniWrapper, LlavaMiniConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ToyVQADataset(n=args.max_samples)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=toy_vqa_collate)

    model = LlavaNextMiniWrapper(LlavaMiniConfig(text_model_name="distilgpt2")).to(device)
    model.eval()

    with torch.no_grad():
        for batch in dl:
            pixel_values = batch["pixel_values"].to(device)
            questions = batch["questions"]

            out = model(pixel_values=pixel_values, questions=questions, answers=None, max_new_tokens=20)
            for q, gen in zip(questions, out["generated_text"]):
                print("Q:", q)
                print("A:", gen)
                print("-" * 60)


if __name__ == "__main__":
    main()
