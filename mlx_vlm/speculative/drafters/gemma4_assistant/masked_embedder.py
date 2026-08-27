"""Centroid-routed sparse LM head for Gemma 4 E2B / E4B drafters.

Mirrors HF's ``Gemma4AssistantMaskedEmbedder``
(``transformers/models/gemma4_assistant/modeling_gemma4_assistant.py:43``).

Idea: rather than computing ``hidden @ embed.T`` over the full 262144-vocab
(expensive for a tiny drafter with ``hidden_size=256``), the drafter learns a
``centroids`` Linear that scores 2048 token clusters, and a ``token_ordering``
buffer that maps each cluster to a contiguous block of canonical token IDs.
At inference, the top-K clusters' tokens (typ. 32×128 = 4096 of 262144) are
materialized and scored densely; the rest of the vocab is filled with a
sentinel ``min - 1`` so it loses any argmax / sampling competition.
"""

from typing import Any

import mlx.core as mx
import mlx.nn as nn


class MaskedEmbedder(nn.Module):
    """Centroid-routed sparse softmax for the assistant drafter's LM head."""

    def __init__(self, config: Any):
        super().__init__()
        text_cfg = config.text_config
        self.hidden_size = text_cfg.hidden_size
        self.vocab_size = text_cfg.vocab_size
        self.num_centroids = config.num_centroids
        self.top_k = config.centroid_intermediate_top_k
        self.vocab_size_per_centroid = self.vocab_size // self.num_centroids

        self.centroids = nn.Linear(self.hidden_size, self.num_centroids, bias=False)
        # ``token_ordering[c * vocab_size_per_centroid : (c+1) * vocab_size_per_centroid]``
        # holds the canonical token IDs assigned to centroid ``c``.
        # Loaded from checkpoint as int64; cast to int32 for indexing. This is a
        # static checkpoint buffer, not a learnable parameter.
        self.token_ordering = mx.zeros((self.vocab_size,), dtype=mx.int32)
        self._freeze_static_buffers()

    def _freeze_static_buffers(self):
        self.freeze(keys="token_ordering", recurse=False, strict=True)

    def unfreeze(self, *, recurse: bool = True, keys=None, strict: bool = False):
        super().unfreeze(recurse=recurse, keys=keys, strict=strict)
        self._freeze_static_buffers()
        return self

    def __call__(self, hidden_states: mx.array, embed_tokens: nn.Module) -> mx.array:
        """Compute sparse logits over the full vocab.

        ``hidden_states``: ``[B, L, hidden_size]``.
        ``embed_tokens``: the drafter's tied embedding MODULE (``nn.Embedding``
        or ``nn.QuantizedEmbedding``) — the module, not its raw ``weight``,
        because a quantized table is bit-packed and must be dequantized
        row-wise (see ``_gather_embedding_rows``).
        Returns: ``[B, L, vocab_size]`` with non-selected positions masked
        to ``min(selected_logits) - 1``.
        """
        B, L = hidden_states.shape[:2]
        selected_canonical, selected_logits = self._selected_logits(
            hidden_states, embed_tokens
        )

        mask_value = float(selected_logits.min().item()) - 1.0

        # Scatter selected_logits into a full-vocab tensor at canonical positions.
        scatter_idx = selected_canonical.reshape(B, L, -1)  # [B, L, top_k*vsc]
        out = mx.full(
            (B, L, self.vocab_size),
            vals=mask_value,
            dtype=hidden_states.dtype,
        )
        # mlx.put_along_axis writes ``src`` at ``index`` along ``axis``.
        return mx.put_along_axis(out, scatter_idx, selected_logits, axis=-1)

    def argmax(self, hidden_states: mx.array, embed_tokens: nn.Module) -> mx.array:
        """Return greedy tokens without materializing full-vocab logits."""
        selected_canonical, selected_logits = self._selected_logits(
            hidden_states, embed_tokens
        )
        best = mx.argmax(selected_logits, axis=-1)[..., None]
        selected_canonical = selected_canonical.reshape(*hidden_states.shape[:2], -1)
        return mx.take_along_axis(selected_canonical, best, axis=-1).squeeze(-1)

    @staticmethod
    def _gather_embedding_rows(embed_tokens: nn.Module, flat_idx: mx.array) -> mx.array:
        """Rows ``flat_idx`` of the embedding table, dense ``[N, hidden_size]``.

        A ``QuantizedEmbedding``'s ``weight`` is bit-PACKED (4-bit × 256 hidden
        = 32 uint32 words per row) — indexing it raw and reshaping to
        ``hidden_size`` is exactly how the ``-4bit`` assistant checkpoints
        crashed (``[reshape] cannot reshape 131072 into (1,1,4096,256)``).
        Gather the packed rows plus their per-group scales/biases and
        dequantize just the selection (top_k·vsc rows) — never the whole
        262k-row table, which would defeat the sparse head's purpose.
        """
        if isinstance(embed_tokens, mx.array):  # raw [vocab, hidden] table
            return embed_tokens[flat_idx]
        rows = embed_tokens.weight[flat_idx]
        if isinstance(embed_tokens, nn.QuantizedEmbedding):
            biases = getattr(embed_tokens, "biases", None)
            return mx.dequantize(
                rows,
                embed_tokens.scales[flat_idx],
                biases[flat_idx] if biases is not None else None,
                group_size=embed_tokens.group_size,
                bits=embed_tokens.bits,
                mode=getattr(embed_tokens, "mode", "affine"),
            )
        return rows

    def _selected_logits(
        self, hidden_states: mx.array, embed_tokens: nn.Module
    ) -> tuple[mx.array, mx.array]:
        B, L = hidden_states.shape[:2]
        # Cluster scores → top-K cluster indices.
        centroid_logits = self.centroids(hidden_states)  # [B, L, num_centroids]
        topk_idx = mx.argpartition(centroid_logits, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]  # [B, L, top_k]

        # Reshape token_ordering to [num_centroids, vocab_size_per_centroid].
        ordering = self.token_ordering.reshape(
            self.num_centroids, self.vocab_size_per_centroid
        )

        # For each selected cluster, fetch its canonical token IDs.
        # selected_canonical: [B, L, top_k, vocab_size_per_centroid]
        selected_canonical = ordering[topk_idx]

        # Gather embeddings (dequantizing if packed) → [B, L, top_k * vsc, hidden].
        # Cast to the activations' dtype: dequantize yields the scales' dtype,
        # which needn't match (and put_along_axis wants uniform dtypes anyway).
        flat_idx = selected_canonical.reshape(-1)
        selected_emb = (
            self._gather_embedding_rows(embed_tokens, flat_idx)
            .reshape(B, L, self.top_k * self.vocab_size_per_centroid, self.hidden_size)
            .astype(hidden_states.dtype)
        )

        # selected_logits = (h @ E.T)
        selected_logits = mx.matmul(
            hidden_states[..., None, :],  # [B, L, 1, hidden]
            selected_emb.swapaxes(-1, -2),  # [B, L, hidden, top_k*vsc]
        ).squeeze(
            -2
        )  # [B, L, top_k*vsc]
        return selected_canonical, selected_logits
