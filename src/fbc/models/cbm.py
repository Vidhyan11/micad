"""Concept Bottleneck Model + image-only baseline.

CBM: emb --concept_head--> concept logits --sigmoid--> concept probs
                                                 |
                                        diagnosis_head (SEES ONLY CONCEPTS)
                                                 |
                                              dx logits

The diagnosis head takes ONLY concept probabilities — never the embedding. This
true bottleneck is what makes the concept-counterfactual faithfulness test (MF1)
meaningful: the decision can only change via the concepts.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .heads import MLP


class CBM(nn.Module):
    def __init__(self, emb_dim: int, n_concepts: int, n_classes: int,
                 concept_hidden: int = 256, dx_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.concept_head = MLP(emb_dim, concept_hidden, n_concepts, dropout)
        self.diagnosis_head = MLP(n_concepts, dx_hidden, n_classes, dropout)

    def forward(self, emb):
        concept_logits = self.concept_head(emb)
        concept_probs = torch.sigmoid(concept_logits)
        dx_logits = self.diagnosis_head(concept_probs)
        return concept_logits, concept_probs, dx_logits

    def predict_concepts(self, emb):
        return torch.sigmoid(self.concept_head(emb))

    def diagnose_from_concepts(self, concept_probs):
        """Run only the bottleneck head — used by the counterfactual test."""
        return self.diagnosis_head(concept_probs)


class ImageOnly(nn.Module):
    """Baseline: emb -> dx directly (no concepts). Accuracy ceiling, no bottleneck."""

    def __init__(self, emb_dim: int, n_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = MLP(emb_dim, hidden, n_classes, dropout)

    def forward(self, emb):
        return self.net(emb)
