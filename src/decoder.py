"""Dynamic Mask Refinement (DMR) decoder.

Implements Sec. II-C of the letter: L cascaded layers that jointly update the
instance queries and the superpoint features. Query initialization follows the
region-guided scheme of Sec. II-B.
"""

import math

import torch
import torch.nn as nn

from ..builder import MODELS
from .query_sampling import confidence_weighted_fps


class CrossAttentionLayer(nn.Module):
    """Masked cross-attention from instance queries to superpoint features."""

    def __init__(self, d_model=384, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source, query, attn_mask=None):
        outputs = []
        for i in range(len(source)):
            mask = attn_mask[i] if attn_mask is not None else None
            output, _ = self.attn(query[i], source[i], source[i], attn_mask=mask)
            output = self.dropout(output) + query[i]
            output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(nn.Module):
    """Self-attention that lets instance queries exchange instance-level cues."""

    def __init__(self, d_model=384, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outputs = []
        for i in range(len(x)):
            output, _ = self.attn(x[i], x[i], x[i])
            output = self.dropout(output) + x[i]
            output = self.norm(output)
            outputs.append(output)
        return outputs


class FFN(nn.Module):
    """Position-wise feed-forward network on the instance queries."""

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn="relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        outputs = []
        for i in range(len(x)):
            output = self.net(x[i])
            output = output + x[i]
            output = self.norm(output)
            outputs.append(output)
        return outputs


class ReverseCrossAttention(nn.Module):
    """Gated residual update of the superpoint features, i.e. Eq. (2).

    Reverse cross-attention uses X_{l-1} as queries and Q_l as keys and values,
    and a learnable scalar alpha_l modulates the residual contribution:
    X_l = X_{l-1} + alpha_l * ReverseCrossAttention(X_{l-1}, Q_l).
    alpha_l is initialized to zero to preserve the backbone-derived features
    during early training. The branch itself applies the reverse attention
    update and a feed-forward residual to the superpoint features.
    """

    def __init__(self, d_model, nhead, hidden_dim, dropout, activation_fn, gated=True):
        super().__init__()
        self.memory_norm = nn.LayerNorm(d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        if gated:
            self.gate = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("gate", None)

    def forward(self, memory, query):
        instance_context = self.query_norm(query)
        update, _ = self.attn(
            self.memory_norm(memory),
            instance_context,
            instance_context,
            need_weights=False,
        )
        update = self.dropout(update)
        update = update + self.ffn(self.ffn_norm(memory + update))
        scale = 1.0 if self.gate is None else self.gate.to(dtype=update.dtype)
        return memory + scale * update


@MODELS.register_module("BRQDMRDecoder")
class BRQDMRDecoder(nn.Module):
    """Decoder with region-guided queries and dynamic mask refinement.

    Args:
        num_layer: number of cascaded decoder layers L.
        num_query: query budget Q.
        num_class: number of foreground classes; the head adds one no-object logit.
        in_channel: dimension of the superpoint features Fs.
        d_model: decoder feature dimension D.
        use_region_guided_query: initialize queries with region-guided query
            generation instead of learnable embeddings.
        guided_score_thr: building-probability threshold tau of the candidate set.
        use_dynamic_mask_refinement: feed updated instance context back into the
            superpoint features used by the next mask prediction.
        gate_residual: use the learnable scalar alpha_l of Eq. (2); disable it for
            the ungated reverse-attention variant.
    """

    def __init__(
        self,
        num_layer=6,
        num_query=150,
        num_class=18,
        in_channel=64,
        d_model=384,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn="relu",
        iter_pred=True,
        use_region_guided_query=True,
        guided_score_thr=0.5,
        use_dynamic_mask_refinement=True,
        gate_residual=True,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.num_query = num_query
        self.d_model = d_model
        self.iter_pred = iter_pred
        self.use_region_guided_query = use_region_guided_query
        self.use_dynamic_mask_refinement = use_dynamic_mask_refinement
        if not 0 <= guided_score_thr <= 1:
            raise ValueError("guided_score_thr must lie in [0, 1]")
        self.guided_score_thr = guided_score_thr

        # X0: linear projection of Fs into the decoder feature space.
        self.input_proj = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        # H0: mask embedding layer that derives the initial mask features.
        self.x_mask = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.query = nn.Embedding(num_query, d_model)
        # Projection of the gathered superpoint features into query space.
        self.query_seed_proj = nn.Linear(in_channel, d_model)

        self.cross_attn_layers = nn.ModuleList(
            [CrossAttentionLayer(d_model, nhead, dropout) for _ in range(num_layer)]
        )
        self.self_attn_layers = nn.ModuleList(
            [SelfAttentionLayer(d_model, nhead, dropout) for _ in range(num_layer)]
        )
        self.ffn_layers = nn.ModuleList(
            [FFN(d_model, hidden_dim, dropout, activation_fn) for _ in range(num_layer)]
        )
        if self.use_dynamic_mask_refinement:
            self.reverse_attn_layers = nn.ModuleList(
                [
                    ReverseCrossAttention(
                        d_model, nhead, hidden_dim, dropout, activation_fn,
                        gated=gate_residual,
                    )
                    for _ in range(num_layer)
                ]
            )
            # Maps the feature increment X_l - X0 into the mask embedding space.
            # Without a bias the zero increment leaves H_l = H0 exactly.
            self.mask_increment_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_class + 1)
        )
        # Separately learned projections for the superpoint and query branches of
        # the scaled-dot-product mask head.
        self.mask_super_proj = nn.Linear(d_model, d_model)
        self.mask_query_proj = nn.Linear(d_model, d_model)

    def _forward_head(self, query, mask_feats):
        """Predict labels, mask logits and the attention mask of the next layer."""
        pred_labels = []
        pred_masks = []
        attn_masks = []
        for i in range(len(query)):
            norm_query = self.out_norm(query[i])
            pred_labels.append(self.out_cls(norm_query))
            # Scaled dot product of the projected queries and superpoint features.
            mask_query = self.mask_query_proj(norm_query)
            mask_super = self.mask_super_proj(mask_feats[i])
            pred_mask = torch.einsum("nd,md->nm", mask_query, mask_super)
            pred_mask = pred_mask / math.sqrt(self.d_model)
            pred_masks.append(pred_mask)
            # Superpoints with mask probability below 0.5 are excluded from the
            # cross-attention of the next layer.
            attn_mask = (pred_mask.sigmoid() < 0.5).bool()
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            attn_masks.append(attn_mask.detach())
        return pred_labels, pred_masks, attn_masks

    def _get_query(self, input_dict, sp_feat, batch_size):
        if not self.use_region_guided_query:
            return self.query.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        return self._get_region_guided_query(input_dict, sp_feat)

    def _get_region_guided_query(self, input_dict, sp_feat):
        """Initialize queries from spatially dispersed, high-confidence seeds."""
        coordinates = input_dict.get("sp_seed_coord")
        confidence = input_dict.get("sp_building_confidence")
        candidates = input_dict.get("sp_building_candidate")
        if coordinates is None or confidence is None or candidates is None:
            raise ValueError(
                "Region-guided queries require sp_seed_coord, "
                "sp_building_confidence and sp_building_candidate"
            )
        if len(coordinates) != len(sp_feat) or len(confidence) != len(sp_feat):
            raise ValueError("Query confidence/coordinates must be provided per scene")
        if len(candidates) != len(sp_feat):
            raise ValueError("Query candidates must be provided per scene")

        queries, indices = [], []
        for feat, coord, score, keep in zip(sp_feat, coordinates, confidence, candidates):
            if len(coord) != len(feat) or len(score) != len(feat) or len(keep) != len(feat):
                raise ValueError("Query seeds must align with superpoint features")
            selected = confidence_weighted_fps(
                coord, score, self.num_query, candidate=keep
            ).to(device=feat.device)
            # Gathered superpoint features are projected into the D-dimensional
            # query space; learnable embeddings fill the remaining slots.
            seeds = self.query_seed_proj(feat[selected])
            learned = self.query.weight[len(selected):].to(dtype=seeds.dtype)
            queries.append(torch.cat((seeds, learned), dim=0))
            indices.append(selected)
        input_dict["query_seed_indices"] = indices
        return torch.stack(queries)

    def forward(self, input_dict):
        sp_feat = input_dict["sp_feat"]
        batch_size = len(sp_feat)

        inst_feats = [self.input_proj(x) for x in sp_feat]
        mask_feats = [self.x_mask(x) for x in sp_feat]
        if self.use_dynamic_mask_refinement:
            initial_inst_feats = inst_feats
            initial_mask_feats = mask_feats
        query = self._get_query(input_dict, sp_feat, batch_size)

        pred_labels, pred_masks = [], []
        pred_label, pred_mask, _ = self._forward_head(query, mask_feats)
        pred_labels.append(pred_label)
        pred_masks.append(pred_mask)

        # The first layer attends to the full superpoint set: the mask of layer l
        # only constrains layers l' > l.
        attn_mask = None
        for i in range(self.num_layer):
            query = self.cross_attn_layers[i](inst_feats, query, attn_mask)
            query = self.self_attn_layers[i](query)
            query = self.ffn_layers[i](query)
            if self.use_dynamic_mask_refinement:
                inst_feats = [
                    self.reverse_attn_layers[i](memory, scene_query)
                    for memory, scene_query in zip(inst_feats, query)
                ]
                # H_l = H0 + W(X_l - X0) updates the mask embedding space, so the
                # refreshed features serve both the mask prediction of this layer
                # and the cross-attention of the next one.
                mask_feats = [
                    initial_mask + self.mask_increment_proj(memory - initial_memory)
                    for initial_mask, memory, initial_memory in zip(
                        initial_mask_feats, inst_feats, initial_inst_feats
                    )
                ]
            pred_label, pred_mask, attn_mask = self._forward_head(query, mask_feats)
            pred_labels.append(pred_label)
            pred_masks.append(pred_mask)

        if not self.iter_pred:
            return {"labels": pred_label, "masks": pred_mask}

        return {
            "labels": pred_label,
            "masks": pred_mask,
            "aux_outputs": [
                {"labels": a, "masks": b}
                for a, b in zip(pred_labels[:-1], pred_masks[:-1])
            ],
        }
