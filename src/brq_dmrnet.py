"""BRQ-DMRNet: region-guided queries and dynamic mask refinement.

The wrapper turns a voxelized scene into superpoints, derives the building
candidates used by region-guided query generation, and maps the predicted
superpoint masks back to the input points.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from ..builder import MODELS, build_model
from ..utils.structure import Point
from .misc import process_instance, process_label, split_offset
from .nms import mask_matrix_nms
from .query_sampling import superpoint_candidates


@MODELS.register_module("BRQDMRNet")
class BRQDMRNet(nn.Module):
    def __init__(
        self,
        backbone,
        decoder,
        criterion=None,
        semantic_num_classes=18,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 2),
        instance_ignore_index=-1,
        topk_insts=100,
        score_thr=0.0,
        npoint_thr=100,
        sp_score_thr=0.5,
        nms=True,
        voxel_semantic_head=None,
        semantic_only=False,
        use_voxel_semantic_loss=True,
        freeze_backbone=False,
    ):
        super().__init__()

        self.semantic_num_classes = semantic_num_classes
        self.semantic_ignore_index = semantic_ignore_index
        self.segment_ignore_index = tuple(segment_ignore_index)
        self.instance_ignore_index = instance_ignore_index

        ignored = sorted([i for i in self.segment_ignore_index if i >= 0])
        total = self.semantic_num_classes + len(ignored)
        kept = [i for i in range(total) if i not in ignored]
        self.register_buffer(
            "class_map", torch.tensor(kept, dtype=torch.long), persistent=False
        )

        self.backbone = build_model(backbone)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        self.decoder = build_model(decoder)
        self.criterion = build_model(criterion) if criterion is not None else None

        self.topk_insts = topk_insts
        self.score_thr = score_thr
        self.npoint_thr = npoint_thr
        self.sp_score_thr = sp_score_thr
        self.nms = nms
        self.voxel_semantic_head = None
        self.semantic_only = semantic_only
        self.use_voxel_semantic_loss = use_voxel_semantic_loss
        if semantic_only and not use_voxel_semantic_loss:
            raise ValueError("semantic_only requires use_voxel_semantic_loss=True")
        if voxel_semantic_head is not None:
            self.voxel_semantic_head = nn.Linear(voxel_semantic_head["in_channels"], 2)
            self.voxel_semantic_loss_weight = voxel_semantic_head.get("loss_weight", 0.1)
            self.voxel_semantic_ignore_index = voxel_semantic_head.get("ignore_index", -1)
            class_weight = voxel_semantic_head.get("class_weight")
            if class_weight is not None:
                class_weight = torch.as_tensor(class_weight, dtype=torch.float32)
                if class_weight.shape != (2,) or not torch.isfinite(class_weight).all() or (class_weight <= 0).any():
                    raise ValueError("Voxel semantic class_weight must contain two positive finite weights [background, building]")
            self.register_buffer("voxel_semantic_class_weight", class_weight, persistent=False)
            if self.voxel_semantic_loss_weight < 0:
                raise ValueError("Voxel semantic loss weight must be nonnegative")
            if not use_voxel_semantic_loss:
                self.voxel_semantic_head.requires_grad_(False)
        elif semantic_only:
            raise ValueError("semantic_only requires a voxel semantic head")
        if semantic_only:
            self.decoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # The trainer calls model.train() each epoch; frozen features must stay
        # deterministic rather than re-enabling the backbone's DropPath layers.
        if self.freeze_backbone:
            self.backbone.eval()
        if self.voxel_semantic_head is not None and not any(
            parameter.requires_grad for parameter in self.voxel_semantic_head.parameters()
        ):
            self.voxel_semantic_head.eval()
        return self

    def forward(self, input_dict):
        vx_offset = input_dict["offset"].int()
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.backbone(Point(input_dict))
        else:
            feats = self.backbone(Point(input_dict))
        input_dict["feats"] = feats
        semantic_loss = None
        semantic_logits = None
        # The voxel semantic head supplies the building probabilities consumed by
        # region-guided query generation, so it runs even when its loss is off.
        if self.voxel_semantic_head is not None:
            semantic_logits = self.voxel_semantic_head(feats)
        if (
            semantic_logits is not None
            and self.use_voxel_semantic_loss
            and "segment" in input_dict
        ):
            semantic_loss = self.voxel_semantic_loss(semantic_logits, input_dict["segment"])
        if self.semantic_only:
            if semantic_loss is None and self.training:
                raise ValueError("Semantic training requires voxel segment labels")
            # Do not run or update the pretrained instance decoder in this stage.
            result = {}
            if semantic_loss is not None:
                result.update(loss=semantic_loss, loss_semantic=semantic_loss)
            if not self.training:
                result["seg_logits"] = semantic_logits
                if "segment" in input_dict:
                    result["voxel_semantic_confusion"] = self.voxel_semantic_confusion(
                        semantic_logits, input_dict["segment"]
                    )
            return result

        pt_offset = input_dict["origin_offset"].int()

        inv = split_offset(input_dict["inverse"], pt_offset)
        sp_raw = split_offset(input_dict["superpoint"], pt_offset)
        sp = [torch.unique(_sp, return_inverse=True)[1] for _sp in sp_raw]

        vx_feat = split_offset(feats, vx_offset)
        vx_coord = split_offset(input_dict["coord"], vx_offset)
        vx_grid_coord = split_offset(input_dict["grid_coord"], vx_offset)

        input_dict["sp_feat"] = [
            scatter_mean(_feat[_inv], _sp, dim=0)
            for _feat, _inv, _sp in zip(vx_feat, inv, sp)
        ]
        input_dict["sp_coord"] = [
            scatter_mean(torch.floor(_coord[_inv] * 50), _sp, dim=0)
            for _coord, _inv, _sp in zip(vx_coord, inv, sp)
        ]
        input_dict["sp_grid_coord"] = [
            scatter_mean(_coord[_inv].float(), _sp, dim=0)
            for _coord, _inv, _sp in zip(vx_grid_coord, inv, sp)
        ]
        input_dict["sp"] = sp
        if getattr(self.decoder, "use_region_guided_query", False):
            point_confidence = self.building_probability(
                input_dict, semantic_logits, inv, pt_offset
            )
            point_coordinates = split_offset(input_dict["origin_coord"], pt_offset)
            threshold = self.decoder.guided_score_thr
            confidences, candidates, centroids = [], [], []
            for points, coord, group in zip(point_confidence, point_coordinates, sp):
                coord = coord.float()
                num_superpoints = int(group.max()) + 1 if len(group) else 0
                candidate, confidence = superpoint_candidates(
                    points.to(device=coord.device),
                    group,
                    threshold,
                    num_superpoints,
                )
                candidates.append(candidate)
                confidences.append(confidence)
                centroids.append(scatter_mean(coord, group, dim=0))
            input_dict["sp_building_confidence"] = confidences
            input_dict["sp_building_candidate"] = candidates
            input_dict["sp_seed_coord"] = centroids

        out = self.decoder(input_dict)

        return_dict = {}
        if self.criterion is not None and "origin_segment" in input_dict:
            return_dict.update(self.criterion(out, self.prepare_target(input_dict)))
        else:
            return_dict["loss"] = torch.tensor(
                0.0, device=feats.device, requires_grad=self.training
            )

        if semantic_loss is not None:
            return_dict["loss_instance"] = return_dict["loss"]
            return_dict["loss_semantic"] = semantic_loss
            return_dict["loss"] = (
                semantic_loss if self.semantic_only
                else return_dict["loss_instance"] + self.voxel_semantic_loss_weight * semantic_loss
            )

        if not self.training:
            return_dict = self.prediction(out, return_dict, sp)
            if semantic_logits is not None and "segment" in input_dict:
                return_dict["voxel_semantic_confusion"] = self.voxel_semantic_confusion(
                    semantic_logits, input_dict["segment"]
                )
        return return_dict

    def building_probability(self, input_dict, semantic_logits, inv, pt_offset):
        """Pointwise building probability p_i of Section II-B.

        Cached point-aligned confidences are used when available; otherwise the
        voxel probabilities of the pretrained semantic head are assigned to every
        point inside the corresponding voxel.
        """
        if "building_confidence" in input_dict:
            confidence = input_dict["building_confidence"].float()
            if confidence.shape != input_dict["superpoint"].shape:
                raise ValueError("Guided queries require point-aligned building_confidence")
            return split_offset(confidence, pt_offset)
        if semantic_logits is None:
            raise ValueError(
                "Region-guided queries require a voxel semantic head or cached "
                "building_confidence"
            )
        probability = semantic_logits.float().softmax(dim=-1)[:, 1]
        return [probability[_inv] for _inv in inv]

    def voxel_semantic_loss(self, logits, target):
        """One CE target per sampled voxel; no inverse or superpoint expansion."""
        target = target.long().reshape(-1)
        if logits.shape[0] != target.numel():
            raise ValueError("Voxel semantic logits and segment labels must align")
        if not torch.all((target == 0) | (target == 1) | (target == self.voxel_semantic_ignore_index)):
            raise ValueError("Voxel semantic labels must be background=0, building=1 or ignore")
        if not torch.any(target != self.voxel_semantic_ignore_index):
            return logits.sum() * 0.0
        return F.cross_entropy(logits.float(), target, weight=self.voxel_semantic_class_weight,
                               ignore_index=self.voxel_semantic_ignore_index)

    def voxel_semantic_confusion(self, logits, target):
        target = target.long().reshape(-1)
        valid = target != self.voxel_semantic_ignore_index
        prediction = logits.argmax(dim=1)
        return torch.bincount(target[valid] * 2 + prediction[valid], minlength=4).reshape(2, 2)

    @torch.no_grad()
    def prepare_target(self, input_dict):
        pt_offset = input_dict["origin_offset"].int()
        pt_ins = split_offset(input_dict["origin_instance"], pt_offset)
        pt_sem = split_offset(input_dict["origin_segment"], pt_offset)
        sp = input_dict["sp"]

        target = {"inst_gt": []}
        for p_ins, p_cls, p_sp in zip(pt_ins, pt_sem, sp):
            p_ins = process_instance(
                p_ins.clone(),
                p_cls.clone(),
                self.segment_ignore_index,
                self.instance_ignore_index,
            )
            p_cls = process_label(
                p_cls.clone(),
                self.segment_ignore_index,
                self.semantic_ignore_index,
            )

            p_ins_mask = p_ins.clone()
            if torch.sum(p_ins_mask == self.instance_ignore_index) != 0:
                p_ins_mask[p_ins_mask == self.instance_ignore_index] = (
                    torch.max(p_ins_mask) + 1
                )
                p_ins_mask = torch.nn.functional.one_hot(p_ins_mask)[:, :-1]
            else:
                p_ins_mask = torch.nn.functional.one_hot(p_ins_mask)

            if p_ins_mask.shape[1] != 0:
                p_ins_mask = p_ins_mask.T
                sp_ins_mask = scatter_mean(p_ins_mask.float(), p_sp, dim=-1) > 0.5
            else:
                sp_ins_mask = p_ins_mask.new_zeros(
                    (0, p_sp.max() + 1), dtype=torch.bool
                )

            unique_insts = p_ins.unique()
            insts = unique_insts[unique_insts != self.instance_ignore_index]
            gt_labels = insts.new_zeros(len(insts))
            for i, inst_id in enumerate(insts):
                gt_labels[i] = p_cls[p_ins == inst_id][0]

            target["inst_gt"].append(dict(labels=gt_labels, masks=sp_ins_mask))
        return target

    def prediction(self, out, return_dict, sp):
        scores, masks, classes = self.predict_by_feat(out, sp)
        masks = masks.cpu().detach().numpy()
        classes = classes.cpu().detach().numpy()

        sort_scores = scores.sort(descending=True)
        sort_scores_index = sort_scores.indices.cpu().detach().numpy()
        return_dict["pred_scores"] = sort_scores.values.cpu().detach().numpy()
        return_dict["pred_masks"] = masks[sort_scores_index]
        return_dict["pred_classes"] = classes[sort_scores_index]
        return return_dict

    def predict_by_feat(self, out, sp):
        cls_preds = out["labels"][0]
        pred_masks = out["masks"][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]

        labels = (
            torch.arange(self.semantic_num_classes, device=scores.device)
            .unsqueeze(0)
            .repeat(len(cls_preds), 1)
            .flatten(0, 1)
        )
        scores, topk_idx = scores.flatten(0, 1).topk(self.topk_insts, sorted=False)
        labels = labels[topk_idx]
        topk_idx = torch.div(topk_idx, self.semantic_num_classes, rounding_mode="floor")

        mask_pred = pred_masks[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / (
            (mask_pred > 0).sum(1) + 1e-6
        )
        # Candidate instances are ranked by the product of classification and
        # mask confidence scores.
        scores = scores * mask_scores

        if self.nms:
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel="linear"
            )

        mask_pred = mask_pred_sigmoid[:, sp[0]] > self.sp_score_thr

        score_mask = scores > self.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        npoint_mask = mask_pred.sum(1) > self.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return scores, mask_pred, self.class_map[labels]
