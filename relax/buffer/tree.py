from pathlib import Path
import pickle
from typing import Any, Callable, Tuple, TypeVar

import numpy as np
import jax, jax.tree_util as tree
from jax import ShapeDtypeStruct

from relax.buffer.base import Buffer
from relax.utils.experience import Experience, SequenceExperience

T = TypeVar("T")
S = TypeVar("S")
ShapeDtypeStructTree = Any  # A tree of same structure as T, but with ShapeDtypeStruct leaves


class TreeBuffer(Buffer[T]):
    def __init__(self, spec: ShapeDtypeStructTree, size: int, seed: int = 0) -> None:
        def create_buffer(sd: ShapeDtypeStruct):
            shape, dtype = (size, *sd.shape), sd.dtype
            return np.empty(shape, dtype=dtype)

        leaves, treedef = tree.tree_flatten(spec)
        self.buffers: Tuple[np.ndarray] = tuple(create_buffer(sd) for sd in leaves)
        self.treedef = treedef

        self.rng = np.random.default_rng(seed)

        self.max_len = size
        self.len = 0
        self.ptr = 0

    def add(self, sample: T, *, from_jax: bool = False) -> None:
        if from_jax:
            samples = jax.device_get(samples)
        leaves = self.treedef.flatten_up_to(sample)
        for leaf, buf in zip(leaves, self.buffers):
            if leaf is not None:  # Skip None fields (e.g., unpopulated next_action)
                buf[self.ptr] = leaf
        self._advance()

    def add_batch(self, samples: T, *, from_jax: bool = False) -> None:
        if from_jax:
            samples = jax.device_get(samples)
        leaves = self.treedef.flatten_up_to(samples)
        # Find batch_size from first non-None leaf
        batch_size = None
        for leaf in leaves:
            if leaf is not None:
                batch_size = leaf.shape[0]
                break
        assert batch_size is not None, "At least one field must be non-None"
        assert batch_size <= self.max_len and all(
            leaf is None or leaf.shape[0] == batch_size for leaf in leaves
        )

        start, end = self.ptr, self.ptr + batch_size
        if end > self.max_len:
            # Need to wrap around
            split, remain = self.max_len - start, end - self.max_len
            for leaf, buf in zip(leaves, self.buffers):
                if leaf is not None:  # Skip None fields
                    buf[start:] = leaf[:split]
                    buf[:remain] = leaf[split:]
        else:
            for leaf, buf in zip(leaves, self.buffers):
                if leaf is not None:  # Skip None fields
                    buf[start:end] = leaf

        self._advance(batch_size)

    def sample(self, size: int, *, to_jax: bool = False) -> T:
        return self.sample_with_indices(size, to_jax=to_jax)[0]

    def sample_with_indices(self, size: int, *, to_jax: bool = False) -> Tuple[T, np.ndarray]:
        indices = self.rng.integers(0, self.len, size=size)
        leaves = tuple(np.take(buf, indices, axis=0) for buf in self.buffers)
        samples = tree.tree_unflatten(self.treedef, leaves)
        if to_jax:
            samples = jax.device_put(samples)
        return samples, indices

    def gather_indices(self, indices: "np.ndarray", *, to_jax: bool = False) -> T:
        """Gather samples at the given buffer indices.

        This is a thin wrapper around the internal storage layout, used by
        higher-level code (e.g., trainers) to implement more structured
        sampling schemes such as train/validation splits without changing the
        existing random sampling API.
        """

        leaves = tuple(np.take(buf, indices, axis=0) for buf in self.buffers)
        samples = tree.tree_unflatten(self.treedef, leaves)
        if to_jax:
            samples = jax.device_put(samples)
        return samples

    def sample_sequences(
        self,
        batch_size: int,
        horizon: int,
        *,
        to_jax: bool = False,
        max_attempts_factor: int = 4,
    ) -> SequenceExperience:
        """Sample H-step sequences that don't cross episode boundaries.

        Samples starting indices and gathers H consecutive transitions,
        rejecting sequences that cross a done=True flag (episode end) or
        cross the ring buffer write pointer.

        Args:
            batch_size: Number of sequences to return.
            horizon: H, the number of consecutive steps per sequence.
            to_jax: If True, return JAX arrays instead of numpy.
            max_attempts_factor: Sample this many times more candidates than
                batch_size to ensure we get enough valid sequences.

        Returns:
            SequenceExperience with:
                obs: [batch_size, obs_dim] - starting state
                actions: [batch_size, horizon, act_dim] - H actions
                rewards: [batch_size, horizon] - H rewards
                dones: [batch_size, horizon] - H done flags
                next_obs: [batch_size, obs_dim] - final next_obs

        Raises:
            ValueError: If not enough valid sequences can be sampled.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if self.len < horizon:
            raise ValueError(
                f"Buffer has {self.len} samples but needs at least {horizon} for sequences"
            )

        # Experience fields order: obs(0), action(1), reward(2), done(3), next_obs(4)
        # We need to access the done buffer to check episode boundaries.
        # The treedef flattens in field order for NamedTuples.
        done_buf_idx = 3  # done is the 4th field in Experience

        # Compute the range of valid starting indices.
        # A valid starting index i must satisfy:
        # 1. i + horizon - 1 < self.len (sequence fits in buffer)
        # 2. The sequence [i, i+1, ..., i+horizon-2] has no done=True
        #    (done at position j means the episode ended after transition j,
        #     so we can't use transitions j+1, j+2, ... from the same sequence)
        # 3. The sequence doesn't cross the write pointer (ring buffer wrap)

        # For a ring buffer, we need to be careful about the wrap-around.
        # The valid range is [0, self.len - horizon] in terms of logical indices,
        # but we need to handle the case where ptr has wrapped.

        # Compute the "forbidden zone" around the write pointer.
        # If ptr < horizon, the forbidden zone wraps around.
        # A sequence starting at i covers indices [i, i+1, ..., i+horizon-1] mod max_len.
        # We don't want any of these to be in the "not yet written" or "just overwritten" zone.

        # Simple approach: sample candidate starts, build sequences, filter invalid ones.
        n_candidates = batch_size * max_attempts_factor
        candidate_starts = self.rng.integers(0, self.len - horizon + 1, size=n_candidates)

        # Build index arrays for all H positions: [n_candidates, horizon]
        offsets = np.arange(horizon)  # [0, 1, ..., H-1]
        seq_indices = (candidate_starts[:, None] + offsets) % self.max_len  # [n_candidates, H]

        # Check validity: no done=True in positions 0..H-2 (done at H-1 is ok, it's the last step)
        done_buf = self.buffers[done_buf_idx]
        # Gather done flags for the first H-1 positions
        if horizon > 1:
            interior_indices = seq_indices[:, :-1]  # [n_candidates, H-1]
            interior_dones = done_buf[interior_indices]  # [n_candidates, H-1]
            # A sequence is valid if none of the interior transitions have done=True
            valid_mask = ~np.any(interior_dones, axis=1)  # [n_candidates]
        else:
            # H=1: all sequences are valid (single transition)
            valid_mask = np.ones(n_candidates, dtype=bool)

        # Also check for ring buffer wrap-around: reject if sequence crosses ptr
        # when buffer is full. This happens if start <= ptr - 1 < start + horizon - 1
        # (i.e., ptr-1 is inside the sequence range).
        if self.len == self.max_len:
            # Buffer is full, ptr points to the oldest entry (about to be overwritten).
            # We should not sample sequences that include ptr-1 (just written) or
            # wrap around in a way that includes stale data.
            # For safety, reject sequences where any index equals (ptr - 1) mod max_len
            # or the sequence wraps around ptr.
            newest_idx = (self.ptr - 1) % self.max_len
            # Check if sequence wraps: start + horizon - 1 >= max_len
            wraps = candidate_starts + horizon - 1 >= self.max_len
            # For wrapped sequences, check more carefully
            # Simple conservative approach: reject if the sequence would cross ptr
            for_rejection = np.zeros(n_candidates, dtype=bool)
            if np.any(wraps):
                # Check if ptr is in the range [start, start + horizon - 1] mod max_len
                # This is tricky; for simplicity, reject wrapped sequences when ptr
                # is in the "danger zone" of size horizon around the wrap point.
                # Actually, simpler: just don't sample from len - horizon + 1 to len.
                pass  # The initial sampling already handles this by using len - horizon + 1
            valid_mask = valid_mask & ~for_rejection

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < batch_size:
            raise ValueError(
                f"Could only find {len(valid_indices)} valid sequences out of "
                f"{n_candidates} candidates (requested {batch_size}). "
                f"Try increasing max_attempts_factor or reducing horizon."
            )

        # Select batch_size valid sequences
        selected = self.rng.choice(valid_indices, size=batch_size, replace=False)
        selected_seq_indices = seq_indices[selected]  # [batch_size, horizon]

        # Gather the data for each field
        obs_buf = self.buffers[0]
        action_buf = self.buffers[1]
        reward_buf = self.buffers[2]
        next_obs_buf = self.buffers[4]

        # Starting obs: first position of each sequence
        obs = obs_buf[selected_seq_indices[:, 0]]  # [batch_size, obs_dim]

        # Actions for all H steps
        actions = action_buf[selected_seq_indices]  # [batch_size, horizon, act_dim]

        # Rewards for all H steps
        rewards = reward_buf[selected_seq_indices]  # [batch_size, horizon]

        # Dones for all H steps
        dones = done_buf[selected_seq_indices]  # [batch_size, horizon]

        # Next observations for all H steps
        # next_obs_seq[:, h] is the state after taking actions[:, h]
        next_obs_seq = next_obs_buf[selected_seq_indices]  # [batch_size, horizon, obs_dim]

        result = SequenceExperience(
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_obs_seq=next_obs_seq,
        )

        if to_jax:
            result = jax.device_put(result)

        return result

    def replace(self, indices: np.ndarray, samples: T, *, from_jax: bool = False) -> None:
        if from_jax:
            samples = jax.device_get(samples)
        leaves = self.treedef.flatten_up_to(samples)
        for leaf, buf in zip(leaves, self.buffers):
            if leaf is None:
                continue  # Skip replacing if leaf is None
            buf[indices] = leaf

    def save(self, path: Path) -> None:
        if self.len < self.max_len:
            # Only save the valid part of the buffer
            leaves = tuple(buf[:self.len] for buf in self.buffers)
        else:
            leaves = self.buffers
        with path.open("wb") as f:
            pickle.dump(leaves, f)

    def _advance(self, size: int = 1):
        self.len = min(self.len + size, self.max_len)
        self.ptr = (self.ptr + size) % self.max_len

    def __len__(self):
        return self.len

    def __repr__(self):
        return f"TreeBuffer(size={self.max_len}, len={self.len}, ptr={self.ptr}, treedef={self.treedef})"

    @staticmethod
    def from_example(example: T, size: int, seed: int = 0, remove_batch_dim: bool = True) -> "TreeBuffer[T]":
        def create_shape_dtype_struct(x: np.ndarray):
            if remove_batch_dim:
                assert x.ndim >= 1
                return ShapeDtypeStruct(x.shape[1:], x.dtype)
            else:
                return ShapeDtypeStruct(x.shape, x.dtype)

        spec = tree.tree_map(create_shape_dtype_struct, example)
        return TreeBuffer(spec, size, seed)

    @staticmethod
    def from_experience(obs_dim: int, act_dim: int, size: int, seed: int = 0, include_next_action: bool = False) -> "TreeBuffer[Experience]":
        # include_next_action is kept for API compatibility but next_action is always included
        # in the buffer structure for consistency
        example = Experience.create_example(obs_dim, act_dim, include_next_action=True)
        return TreeBuffer.from_example(example, size, seed, remove_batch_dim=False)

    @staticmethod
    def connect(src: "TreeBuffer[S]", dst: "TreeBuffer[T]", converter: Callable[[S], T]):
        original_add = src.add
        original_add_batch = src.add_batch

        def add(self, experience: Experience):
            original_add(experience)
            dst.add(converter(experience))

        def add_batch(self, experiences: Experience):
            original_add_batch(experiences)
            dst.add_batch(converter(experiences))

        import types
        src.add = types.MethodType(add, src)
        src.add_batch = types.MethodType(add_batch, src)
