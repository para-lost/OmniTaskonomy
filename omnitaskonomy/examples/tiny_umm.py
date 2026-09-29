"""CPU adapter example: RGB regression and four-class visual question answering."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from omnitaskonomy.data.common import resolve_image
from omnitaskonomy.umm import Loss, ParameterSpec


class TinyUMM(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 8)
        self.generation = nn.Linear(8, 3)
        self.understanding = nn.Linear(8, 4)
        self.scale = nn.Parameter(torch.ones(8), requires_grad=False)

    def forward(self, inputs, objective):
        hidden = torch.tanh(self.shared(inputs)) * self.scale
        return self.generation(hidden) if objective == "i2i" else self.understanding(hidden)


def image_mean(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) / 255


class TinyAdapter:
    def __init__(self, model_path, checkpoint, device):
        weights = checkpoint or model_path
        if weights.is_dir():
            weights = weights / "model.pt"
        self.model = TinyUMM().to(device)
        self.model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        self.checkpoint_files = (weights,)
        self.device = device

    def parameter_specs(self):
        for name, parameter in self.model.named_parameters():
            role = name.split(".")[0] if name != "scale" else "fixed"
            yield ParameterSpec(name, role, role if role == "shared" else None, layer=0,
                                trainable=parameter.requires_grad)

    def _features(self, images, prompt):
        color = np.mean([image_mean(path) for path in images], axis=0) if images else np.zeros(3)
        text = sum(prompt.encode()) % 257 / 256
        return torch.tensor([*color, text], dtype=torch.float32, device=self.device)

    def loss(self, records, objective, context):
        if objective not in {"i2i", "i2t"}:
            raise ValueError(f"Unknown objective: {objective}")
        inputs, targets = [], []
        rng = torch.Generator(device=self.device).manual_seed(context.seed)
        for record in records:
            if objective == "i2i":
                images, prompt = [record["source_image"]], record["prompt"]
                targets.append(image_mean(resolve_image(context.manifest, record["target_image"])))
            else:
                images = record["images"] if "images" in record else [record["image"]]
                if "gradient_mcq" in record:
                    prompt, answer = record["gradient_mcq"]["prompt"], record["gradient_mcq"]["answer"]
                else:
                    prompt = record["conversations"][0]["value"]
                    answer = record["conversations"][-1]["value"]
                if answer not in "ABCD" or len(answer) != 1:
                    raise ValueError("The tiny example supports answers A, B, C or D")
                targets.append("ABCD".index(answer))
            feature = self._features([resolve_image(context.manifest, path) for path in images], prompt)
            if torch.rand((), generator=rng, device=self.device).item() < context.condition_dropout:
                feature = torch.zeros_like(feature)
            inputs.append(feature)
        predictions = self.model(torch.stack(inputs), objective)
        if objective == "i2i":
            target = torch.tensor(np.asarray(targets), device=self.device)
            total = (predictions - target).square().mean(dim=-1).sum()
        else:
            total = F.cross_entropy(predictions, torch.tensor(targets, device=self.device), reduction="sum")
        return Loss(total, len(records))

    def generate(self, messages, *, output="text", **kwargs):
        images = [item["value"] for item in messages if item["type"] == "image"]
        prompt = "\n".join(item["value"] for item in messages if item["type"] == "text")
        self.model.eval()
        with torch.inference_mode():
            features = self._features(images, prompt).unsqueeze(0)
            if output == "text":
                return "ABCD"[self.model(features, "i2t").argmax().item()]
            if output == "image":
                rgb = self.model(features, "i2i")[0].clamp(0, 1).mul(255).byte().tolist()
                return Image.new("RGB", (8, 8), tuple(rgb))
        raise ValueError("Output must be text or image")

    def save_checkpoint(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")


def create_adapter(*, model_path, checkpoint=None, device="cpu", options=None):
    if options:
        raise ValueError("The tiny example has no model options")
    return TinyAdapter(Path(model_path), Path(checkpoint) if checkpoint else None, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.save(TinyUMM().state_dict(), args.output / "model.pt")
