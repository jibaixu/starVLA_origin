"""Pure helpers used only by the independent Space2Act two-stage path."""

from __future__ import annotations

import math
from bisect import bisect_right

import torch


SPACE2ACT_CHANNELS = ("spatial", "future", "plan")


def is_enabled(value) -> bool:
    return value in (True, "True", "true", 1)


def normalize_confidence_aware_config(config) -> dict[str, float | bool]:
    config = config or {}
    min_weight = float(config.get("min_weight", 0.0))
    if not math.isfinite(min_weight) or not 0.0 <= min_weight <= 1.0:
        raise ValueError("confidence_aware_alignment min_weight must be finite and within [0, 1]")
    return {
        "enabled": is_enabled(config.get("enabled", False)),
        "min_weight": min_weight,
        "require_cached_confidence": is_enabled(config.get("require_cached_confidence", True)),
    }


def normalize_loss_schedule(schedule_cfg, active_channels) -> tuple[tuple[int, dict[str, float]], ...]:
    if not schedule_cfg or not is_enabled(schedule_cfg.get("enabled", False)):
        return ()
    if str(schedule_cfg.get("mode", "fixed")).lower() != "progressive":
        return ()
    expected = set(SPACE2ACT_CHANNELS)
    if set(active_channels) != expected:
        raise ValueError("Progressive loss_schedule requires spatial/future/plan alignment channels")
    knots = schedule_cfg.get("knots")
    if not knots:
        raise ValueError("Progressive loss_schedule requires non-empty knots")
    result, previous_step = [], None
    for index, knot in enumerate(knots):
        raw_step = knot.get("step")
        if raw_step is None or isinstance(raw_step, bool) or int(raw_step) != float(raw_step):
            raise ValueError(f"loss_schedule knot {index} has invalid step: {raw_step!r}")
        step = int(raw_step)
        if step < 0 or (previous_step is not None and step <= previous_step):
            raise ValueError("loss_schedule knot steps must be non-negative and strictly increasing")
        weights = knot.get("weights")
        if weights is None or set(weights) != expected:
            raise ValueError("Every loss_schedule knot must define spatial/future/plan weights")
        normalized = {channel: float(weights[channel]) for channel in SPACE2ACT_CHANNELS}
        if any(not math.isfinite(weight) or weight < 0.0 for weight in normalized.values()):
            raise ValueError("loss_schedule weights must be finite and non-negative")
        result.append((step, normalized))
        previous_step = step
    if result[0][0] != 0:
        raise ValueError("loss_schedule must start at step 0")
    return tuple(result)


def interpolate_loss_schedule(knots, training_step: int) -> dict[str, float]:
    if isinstance(training_step, bool) or int(training_step) != float(training_step):
        raise ValueError(f"training_step must be an integer, got {training_step!r}")
    step = max(0, int(training_step))
    steps = [knot_step for knot_step, _ in knots]
    right = bisect_right(steps, step)
    if right == 0:
        return dict(knots[0][1])
    if right >= len(knots):
        return dict(knots[-1][1])
    left_step, left_weights = knots[right - 1]
    right_step, right_weights = knots[right]
    ratio = (step - left_step) / (right_step - left_step)
    return {
        channel: left_weights[channel] + ratio * (right_weights[channel] - left_weights[channel])
        for channel in SPACE2ACT_CHANNELS
    }


def normalize_gradnorm_config(schedule_cfg, active_channels) -> dict[str, float | int | bool]:
    mode = str(schedule_cfg.get("mode", "fixed")).lower() if schedule_cfg else "fixed"
    if mode != "gradnorm":
        return {"enabled": False, "mode": mode}
    if not is_enabled(schedule_cfg.get("enabled", False)):
        raise ValueError("GradNorm loss_schedule requires enabled=true")
    channels = tuple(active_channels)
    if set(channels) != set(SPACE2ACT_CHANNELS):
        raise ValueError("GradNorm requires spatial/future/plan alignment channels")
    if schedule_cfg.get("knots"):
        raise ValueError("GradNorm loss_schedule cannot define fixed knots")

    def finite(name, default, minimum, maximum=None):
        value = float(schedule_cfg.get(name, default))
        if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"GradNorm {name} is outside its supported range")
        return value

    def nonnegative_int(name, default):
        value = schedule_cfg.get(name, default)
        if isinstance(value, bool) or int(value) != float(value) or int(value) < 0:
            raise ValueError(f"GradNorm {name} must be a non-negative integer")
        return int(value)

    def positive_int(name, default):
        value = nonnegative_int(name, default)
        if not value:
            raise ValueError(f"GradNorm {name} must be a positive integer")
        return value

    probe_interval = positive_int("gradient_probe_interval", 20)
    update_interval = positive_int("gradient_update_interval", 200)
    if update_interval % probe_interval:
        raise ValueError("GradNorm gradient_update_interval must be divisible by gradient_probe_interval")
    alpha = finite("alpha", 0.0, 0.0)
    if not math.isclose(alpha, 0.0, abs_tol=1e-12):
        raise ValueError("This GradNorm implementation only supports alpha=0")
    # VLM-only training has no action loss. Match its fixed baseline where
    # S/F/P each have loss_weight=1.0, then let GradNorm redistribute that
    # total scale rather than treating it as an auxiliary-loss budget.
    budget = finite("auxiliary_budget", 3.0, 0.0)
    min_weight = finite("min_weight", 0.25, 0.0)
    max_weight = finite("max_weight", 2.0, min_weight)
    if budget < len(channels) * min_weight or budget > len(channels) * max_weight:
        raise ValueError("GradNorm auxiliary_budget must be feasible under min_weight/max_weight")
    return {
        "enabled": True,
        "mode": "gradnorm",
        "alpha": alpha,
        "auxiliary_budget": budget,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "gradient_epsilon": finite("gradient_epsilon", 1e-8, 1e-12),
        "gradient_ema_decay": finite("gradient_ema_decay", 0.9, 0.0, 1.0),
        "gradient_probe_interval": probe_interval,
        "gradient_update_interval": update_interval,
        "adaptive_weight_ramp_steps": nonnegative_int("adaptive_weight_ramp_steps", 0),
    }


def project_bounded_simplex(values: dict[str, float], total: float, lower: float, upper: float) -> dict[str, float]:
    channels = tuple(values)
    if not channels:
        return {}
    if lower < 0.0 or upper < lower or total < len(channels) * lower or total > len(channels) * upper:
        raise ValueError("Invalid bounded-simplex limits")
    raw = torch.tensor([float(values[channel]) for channel in channels], dtype=torch.float64)
    low, high = float(raw.min().item()) - upper, float(raw.max().item()) - lower
    for _ in range(80):
        shift = (low + high) * 0.5
        projected = raw.sub(shift).clamp(min=lower, max=upper)
        if float(projected.sum()) > total:
            low = shift
        else:
            high = shift
    projected = raw.sub((low + high) * 0.5).clamp(min=lower, max=upper)
    residual = total - float(projected.sum())
    if abs(residual) > 1e-12:
        for index, value in enumerate(projected.tolist()):
            room = (upper - value) if residual > 0.0 else (value - lower)
            delta = math.copysign(min(abs(residual), max(room, 0.0)), residual)
            projected[index] += delta
            residual -= delta
            if abs(residual) <= 1e-12:
                break
    return {channel: float(projected[index]) for index, channel in enumerate(channels)}
