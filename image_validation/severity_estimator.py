"""Shared zero-shot severity features for local and Lambda inference."""
import torch

WASTE_SCALES = ['small', 'medium', 'large']
SCALE_PROMPTS = [
    'a photo of a few scattered pieces of litter or a small amount of waste on the ground',
    'a photo of several bags of garbage or a localized pile of waste on the ground',
    'a photo of many bags of garbage or a large pile of waste covering a substantial part of a street or public area',
]
OBSTRUCTION_LEVELS = ['none', 'partial', 'complete']
OBSTRUCTION_PROMPTS = [
    'a photo of trash or waste lying cleanly off to the side, in a bin, or in an open area without blocking any road, path, sidewalk, or doorway',
    'a photo of garbage or debris spilling onto the sidewalk, curb, or road shoulder, partially obstructing pedestrian passage or narrow street traffic',
    'a photo of garbage, fallen debris, or uncollected waste completely blocking a road, sidewalk, drainage canal, or alleyway, preventing people or vehicles from passing',
]


class SeverityEstimator:
    def __init__(self, clip_model, tokenize, device):
        self.features = []
        with torch.no_grad():
            for prompts in (SCALE_PROMPTS, OBSTRUCTION_PROMPTS):
                features = clip_model.encode_text(tokenize(prompts).to(device)).float()
                self.features.append(features / features.norm(dim=-1, keepdim=True))

    def _classify(self, embedding, labels, features):
        with torch.no_grad():
            # Relative CLIP scores, not measured real-world accuracy probabilities.
            probabilities = torch.softmax((embedding @ features.T).squeeze(0) * 100.0, dim=-1)
            index = int(probabilities.argmax().item())
        return labels[index], round(probabilities[index].item(), 4), {
            label: round(probabilities[i].item(), 4) for i, label in enumerate(labels)
        }

    def classify_scale(self, embedding):
        return self._classify(embedding, WASTE_SCALES, self.features[0])

    def classify_obstruction(self, embedding):
        return self._classify(embedding, OBSTRUCTION_LEVELS, self.features[1])
