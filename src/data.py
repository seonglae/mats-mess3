"""Dataset construction for non-ergodic Mess3 mixture."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .mess3 import generate_mess3_sequences_fast


# Default component parameterizations
DEFAULT_COMPONENTS = [
    (0.90, 0.05),  # High clarity, high persistence
    (0.60, 0.15),  # Medium clarity, medium persistence
    (0.85, 0.30),  # High clarity, low persistence
    (0.50, 0.10),  # Low clarity, high persistence
]

# Simpler 2-component version
TWO_COMPONENTS = [
    (0.85, 0.05),  # Standard Mess3
    (0.60, 0.15),  # Diffuse Mess3
]


class NonErgodicMess3Dataset(Dataset):
    """Non-ergodic dataset: each sequence from one Mess3 component.

    The model sees only tokens {BOS=0, token0=1, token1=2, token2=3}.
    Component labels are stored for analysis but not given to the model.
    """

    def __init__(
        self,
        components: list[tuple[float, float]],
        n_sequences_per_component: int,
        seq_length: int,
        seed: int = 42,
    ):
        self.components = components
        self.seq_length = seq_length
        self.n_components = len(components)
        rng = np.random.default_rng(seed)

        all_tokens = []
        all_labels = []
        all_states = []

        for k, (alpha, x) in enumerate(components):
            tokens, states = generate_mess3_sequences_fast(
                alpha, x, n_sequences_per_component, seq_length, rng
            )
            all_tokens.append(tokens)
            all_states.append(states)
            all_labels.append(np.full(n_sequences_per_component, k))

        self.raw_tokens = np.concatenate(all_tokens, axis=0)  # values in {0,1,2}
        self.states = np.concatenate(all_states, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)

        # Shift tokens by 1 to make room for BOS=0
        # Model vocabulary: {0=BOS, 1=token0, 2=token1, 3=token2}
        self.tokens = self.raw_tokens + 1

        # Shuffle
        perm = rng.permutation(len(self.tokens))
        self.tokens = self.tokens[perm]
        self.raw_tokens = self.raw_tokens[perm]
        self.states = self.states[perm]
        self.labels = self.labels[perm]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> dict:
        # Prepend BOS token
        bos = np.array([0], dtype=np.int64)
        seq = np.concatenate([bos, self.tokens[idx]])

        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "label": self.labels[idx],
            "raw_tokens": torch.tensor(self.raw_tokens[idx], dtype=torch.long),
            "states": torch.tensor(self.states[idx], dtype=torch.long),
        }


def create_dataloaders(
    components: list[tuple[float, float]],
    n_sequences_per_component: int = 20000,
    seq_length: int = 16,
    batch_size: int = 4096,
    seed: int = 42,
) -> tuple:
    """Create train dataset and dataloader."""
    dataset = NonErgodicMess3Dataset(
        components=components,
        n_sequences_per_component=n_sequences_per_component,
        seq_length=seq_length,
        seed=seed,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return dataset, loader
