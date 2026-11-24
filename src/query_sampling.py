"""Region-guided query seed selection (Sec. II-B).

A pretrained semantic head yields pointwise building probabilities, from which
the foreground candidate set and the per-superpoint building confidence are
derived. Confidence-weighted farthest point sampling (CW-FPS) then selects the
query seeds inside that set.
"""

import torch


@torch.no_grad()
def superpoint_candidates(
    point_confidence, point_to_superpoint, threshold=0.5, num_superpoints=None
):
    """Foreground candidate set C and building confidence p_s.

    A superpoint joins C when more than half of its constituent points satisfy
    p_i >= tau, and its building confidence is the mean pointwise probability.

    Args:
        point_confidence: pointwise building probabilities in [0, 1], shape (P,).
        point_to_superpoint: superpoint index of every point, shape (P,).
        threshold: confidence threshold tau.
        num_superpoints: number of superpoints; inferred from the indices if None.

    Returns:
        (candidate, confidence), both with shape (S,). ``candidate`` is boolean.
    """
    if point_confidence.ndim != 1:
        raise ValueError("Point confidence must have shape (P,)")
    if point_to_superpoint.shape != point_confidence.shape:
        raise ValueError("Point confidence and superpoint indices must align")
    if not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must lie in [0, 1]")
    if num_superpoints is None:
        num_superpoints = int(point_to_superpoint.max()) + 1 if len(point_to_superpoint) else 0
    device = point_confidence.device
    index = point_to_superpoint.to(device=device, dtype=torch.long)
    if len(index) and (int(index.min()) < 0 or int(index.max()) >= num_superpoints):
        raise ValueError("Superpoint indices must lie in [0, num_superpoints)")

    points = torch.ones_like(index)
    sizes = torch.zeros(num_superpoints, dtype=torch.long, device=device)
    sizes.scatter_add_(0, index, points)
    # Count candidates exactly instead of averaging, so "more than half" is never
    # blurred by floating-point rounding.
    selected = (point_confidence >= threshold).to(torch.long)
    counts = torch.zeros(num_superpoints, dtype=torch.long, device=device)
    counts.scatter_add_(0, index, selected)
    candidate = 2 * counts > sizes

    total = torch.zeros(num_superpoints, dtype=torch.float32, device=device)
    total.scatter_add_(0, index, point_confidence.to(device=device, dtype=torch.float32))
    confidence = total / sizes.clamp(min=1)
    return candidate, confidence


@torch.no_grad()
def confidence_weighted_fps(coordinates, confidence, num_samples, candidate=None):
    """Select seeds that maximise p_s * d(s) inside the candidate set.

    CW-FPS starts from the highest-confidence candidate and then, at every step,
    adds the candidate maximising the confidence-weighted distance to the nearest
    selected seed, i.e. d(s) = p_s * min_{s' in S} ||c_s - c_s'||_2.

    Indices are unique, including coincident coordinates. There is no padding
    here: the caller fills missing seeds with learnable query embeddings.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Query seed coordinates must have shape (S, 3)")
    if confidence.shape != (len(coordinates),):
        raise ValueError("Query seed confidence must have shape (S,)")
    if num_samples < 0:
        raise ValueError("Query count must be nonnegative")
    if candidate is not None and candidate.shape != confidence.shape:
        raise ValueError("Query candidates must have shape (S,)")
    coordinates = coordinates.detach().float()
    confidence = confidence.detach().to(device=coordinates.device, dtype=torch.float32)
    if not torch.isfinite(coordinates).all() or not torch.isfinite(confidence).all():
        raise ValueError("Query seed coordinates/confidence must be finite")
    if (confidence < 0).any() or (confidence > 1).any():
        raise ValueError("Building confidence must be a probability in [0, 1]")
    if candidate is None:
        candidate = torch.ones_like(confidence, dtype=torch.bool)
    else:
        candidate = candidate.to(device=coordinates.device, dtype=torch.bool)
    points = torch.nonzero(candidate, as_tuple=True)[0]
    count = min(num_samples, len(points))
    if count == 0:
        return points[:0]
    xyz, probabilities = coordinates[points], confidence[points]
    selected = torch.empty(count, dtype=torch.long, device=coordinates.device)
    used = torch.zeros(len(points), dtype=torch.bool, device=coordinates.device)
    distance_squared = torch.full_like(probabilities, torch.inf)
    current = probabilities.argmax()
    for index in range(count):
        selected[index] = points[current]
        used[current] = True
        delta = xyz - xyz[current]
        distance_squared = torch.minimum(distance_squared, delta.square().sum(-1))
        # argmax(p**2 * d**2) == argmax(p * d), avoiding a square root.
        priority = (probabilities.square() * distance_squared).masked_fill(used, -1)
        current = priority.argmax()
    return selected
