"""Label smoothing utilities for training data preprocessing.

This module provides various smoothing strategies for converting step-function labels
into smooth, continuous values for better training stability.
"""

import math
from abc import ABC, abstractmethod
from typing import Literal

type LabelSmootherType = Literal["none", "linear", "exponential", "sigmoid", "ramp", "calibration"]


class LabelSmoother(ABC):
    """Base class for label smoothing strategies."""

    @abstractmethod
    def smooth(self, labels: list[float]) -> list[float]:
        """Apply smoothing to labels.

        Args:
            labels: List of label values (typically step-function like)

        Returns:
            Smoothed label values
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class NoSmoother(LabelSmoother):
    """No smoothing - returns labels as-is (step function)."""

    def smooth(self, labels: list[float]) -> list[float]:
        return labels[:]

    def __repr__(self) -> str:
        return "NoSmoother()"


class LinearSmoother(LabelSmoother):
    """Linear smoothing: connects jump points with straight lines.

    This smoother detects step changes (jumps) in the label sequence and
    connects consecutive jump points with linear interpolation, creating
    a smooth transition instead of step functions.

    For example, if we have:
        labels = [0, 0, 0, 1, 1, 1, 0.5, 0.5, 0.5]

    Jump points are at indices 2 (0→1) and 5 (1→0.5).
    The smoother will interpolate linearly between these jump points:
        - indices 0-2: remain 0
        - indices 3-5: interpolate from 1 to 0.5 (1.0, 0.75, 0.5)
        - indices 6-8: remain 0.5

    Args:
        min_segment_length: Minimum segment length to apply interpolation (default: 2).
                            If the gap between jump points is smaller than this,
                            no interpolation is applied.
    """

    def __init__(self, min_segment_length: int = 2):
        if min_segment_length < 1:
            raise ValueError("min_segment_length must be >= 1")
        self.min_segment_length = min_segment_length

    def smooth(self, labels: list[float]) -> list[float]:
        if len(labels) == 0:
            return []

        result = labels[:]
        n = len(labels)

        # Find all jump/change points (indices where change happens)
        jump_points = []
        for i in range(n - 1):
            if abs(labels[i + 1] - labels[i]) > 1e-9:
                jump_points.append(i)

        # No jumps
        if not jump_points:
            return result

        # Build key points: (index, value) pairs at jump locations
        # Start from index 0 with its value
        key_points = [(0, labels[0])]

        for jump_idx in jump_points:
            # Add the point AFTER the jump
            key_points.append((jump_idx + 1, labels[jump_idx + 1]))

        # Interpolate between consecutive key points
        for i in range(len(key_points) - 1):
            start_idx, start_val = key_points[i]
            end_idx, end_val = key_points[i + 1]

            # Skip if segment is too short
            if end_idx - start_idx < self.min_segment_length:
                continue

            # Linear interpolation for indices between start_idx and end_idx
            for j in range(start_idx, end_idx):
                if end_idx > start_idx:
                    t = (j - start_idx) / (end_idx - start_idx)
                    result[j] = start_val + t * (end_val - start_val)

        return result

    def __repr__(self) -> str:
        return f"LinearSmoother(min_segment_length={self.min_segment_length})"


class ExponentialSmoother(LabelSmoother):
    """Exponential smoothing: applies exponential moving average to labels.

    This smoother uses an exponential moving average approach to smooth
    the entire label sequence, which is particularly useful for reducing
    high-frequency noise while preserving overall trends.

    Args:
        alpha: Smoothing factor between 0 and 1. Lower values = smoother.
              - alpha=1.0: no smoothing (same as NoSmoother)
              - alpha=0.5: moderate smoothing
              - alpha=0.1: heavy smoothing
    """

    def __init__(self, alpha: float = 0.3):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha

    def smooth(self, labels: list[float]) -> list[float]:
        if len(labels) == 0:
            return []

        result: list[float] = []
        # Start with the first value
        ema = labels[0]
        result.append(ema)

        # Apply exponential moving average
        for i in range(1, len(labels)):
            ema = self.alpha * labels[i] + (1 - self.alpha) * ema
            result.append(ema)

        return result

    def __repr__(self) -> str:
        return f"ExponentialSmoother(alpha={self.alpha})"


class SigmoidSmoother(LabelSmoother):
    """Sigmoid smoothing: S-curve transitions between jump points.

    This smoother connects jump points with sigmoid (S-shaped) curves,
    creating gradual transitions that start slow, accelerate in the middle,
    and decelerate at the end. This provides smoother transitions than
    linear interpolation while maintaining the overall trend.

    Args:
        min_segment_length: Minimum segment length to apply interpolation (default: 2).
                            If the gap between jump points is smaller than this,
                            no interpolation is applied.
        steepness: Controls the steepness of the S-curve (default: 3.0).
                   Higher values create sharper transitions.
    """

    def __init__(self, min_segment_length: int = 2, steepness: float = 3.0):
        if min_segment_length < 1:
            raise ValueError("min_segment_length must be >= 1")
        if steepness <= 0:
            raise ValueError("steepness must be > 0")
        self.min_segment_length = min_segment_length
        self.steepness = steepness

    def smooth(self, labels: list[float]) -> list[float]:
        if len(labels) == 0:
            return []

        result = labels[:]
        n = len(labels)

        # Find all jump/change points (indices where change happens)
        jump_points = []
        for i in range(n - 1):
            if abs(labels[i + 1] - labels[i]) > 1e-9:
                jump_points.append(i)

        # No jumps
        if not jump_points:
            return result

        # Build key points: (index, value) pairs at jump locations
        # Start from index 0 with its value
        key_points = [(0, labels[0])]

        for jump_idx in jump_points:
            # Add the point AFTER the jump
            key_points.append((jump_idx + 1, labels[jump_idx + 1]))

        # Interpolate between consecutive key points using sigmoid
        for i in range(len(key_points) - 1):
            start_idx, start_val = key_points[i]
            end_idx, end_val = key_points[i + 1]

            # Skip if segment is too short
            if end_idx - start_idx < self.min_segment_length:
                continue

            # Sigmoid interpolation for indices between start_idx and end_idx
            for j in range(start_idx, end_idx):
                if end_idx > start_idx:
                    # Normalize position to [0, 1]
                    t = (j - start_idx) / (end_idx - start_idx)

                    # Use logistic sigmoid function for proper S-curve
                    # Map t from [0,1] to sigmoid output
                    # sigmoid with steepness k: 1 / (1 + exp(-k * (t - 0.5)))
                    # We then normalize back to [0, 1]

                    # Standard sigmoid centered at t=0.5
                    k = self.steepness * 10  # Scale factor for more intuitive range
                    if k > 0:
                        # Apply sigmoid: map t from [0,1] through [-k/2, k/2]
                        x = k * (t - 0.5)
                        # Clamp x to avoid overflow
                        x = max(-10, min(10, x))
                        sigmoid_raw = 1 / (1 + pow(2.71828, -x))  # e^(-x)

                        # Normalize sigmoid output from [1/(1+e^(k/2)), 1/(1+e^(-k/2))] to [0, 1]
                        sigmoid_min = 1 / (1 + pow(2.71828, k / 2))
                        sigmoid_max = 1 / (1 + pow(2.71828, -k / 2))
                        sigmoid_t = (sigmoid_raw - sigmoid_min) / (sigmoid_max - sigmoid_min)
                    else:
                        # Fallback to linear if steepness is 0 or negative
                        sigmoid_t = t

                    result[j] = start_val + sigmoid_t * (end_val - start_val)

        return result

    def __repr__(self) -> str:
        return f"SigmoidSmoother(min_segment_length={self.min_segment_length}, steepness={self.steepness})"


class RampSmoother(LabelSmoother):
    """Ramp smoothing: Linearly interpolates current segment towards the next value.

    Unlike LinearSmoother which only smooths the short gap between jumps,
    this smoother treats the entire constant segment (plateau) as a ramp,
    gradually transitioning from the current value to the value of the
    next segment.

    As seen in the provided image (Red line):
    - When the original label (Blue) stays at 0 and is about to jump to 0.5,
      this smoother generates a line from 0 to 0.5 across that duration.
    - For the last segment, it maintains the final value flat.
    """

    def smooth(self, labels: list[float]) -> list[float]:
        if len(labels) == 0:
            return []

        n = len(labels)
        result = [0.0] * n

        # 1. Find indices where the value changes (start of each segment)
        # Always start with index 0
        change_indices = [0]
        for i in range(1, n):
            # Using epsilon comparison for floats consistent with existing code
            if abs(labels[i] - labels[i - 1]) > 1e-9:
                change_indices.append(i)

        # Add the total length as the closing boundary
        change_indices.append(n)

        # 2. Iterate through each segment
        for k in range(len(change_indices) - 1):
            start_idx = change_indices[k]
            end_idx = change_indices[k + 1]

            current_val = labels[start_idx]

            # Determine target value (value of the NEXT segment)
            # If it's the last segment, there is no next value, so target = current
            if k + 1 < len(change_indices) - 1:
                next_seg_start = change_indices[k + 1]
                target_val = labels[next_seg_start]
            else:
                target_val = current_val

            segment_len = end_idx - start_idx

            # 3. Interpolate from current_val to target_val over the segment length
            for i in range(segment_len):
                global_idx = start_idx + i

                # t goes from 0.0 at start to 1.0 (approaching) at the end of segment
                # We want result[end_idx] to effectively be target_val, so we ramp up.
                t = i / segment_len

                result[global_idx] = current_val + t * (target_val - current_val)

        return result

    def __repr__(self) -> str:
        return "RampSmoother()"


class CalibrationSmoother(LabelSmoother):
    """Designed for Model Calibration / Self-Knowledge training.

    WARNING: This smoother is specifically designed for calibration tasks and
    does NOT use lookahead/ramping to avoid data leakage.

    Strategies:
    1. NO Lookahead ramping: Keeps transitions sharp to reflect reality.
    2. Label Clipping: Prevents overconfidence (0.0 -> 0.05, 1.0 -> 0.95).
    3. Slight Gaussian Blur: Handles boundary ambiguity/noise only.

    Args:
        epsilon: Label smoothing factor.
                 Target becomes [epsilon, 1-epsilon]. Default 0.05.
        sigma: Sigma for Gaussian kernel. Keep it small (e.g., 0.5 to 1.0)
               to avoid blurring 'wrong' states into 'correct' ones too much.

    Example:
        >>> labels = [0, 0, 0, 1, 1, 1]
        >>> smoother = CalibrationSmoother(epsilon=0.05, sigma=0.5)
        >>> smoothed = smoother.smooth(labels)
        # Result will have soft transitions, but values stay in [0.05, 0.95]
    """

    def __init__(self, epsilon: float = 0.05, sigma: float = 0.5):
        if epsilon < 0 or epsilon >= 0.5:
            raise ValueError("epsilon must be in [0, 0.5)")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.epsilon = epsilon
        self.sigma = sigma
        # Small radius because we don't want extensive smoothing
        self.radius = int(math.ceil(3 * sigma))

    def smooth(self, labels: list[float]) -> list[float]:
        if not labels:
            return []

        n = len(labels)
        result = [0.0] * n

        # 1. Apply Gaussian Convolution (Denoising)
        # -----------------------------------------
        # Precompute kernel
        kernel = []
        for x in range(-self.radius, self.radius + 1):
            val = math.exp(-(x * x) / (2 * self.sigma * self.sigma))
            kernel.append(val)
        kernel_sum = sum(kernel)
        kernel = [k / kernel_sum for k in kernel]

        for i in range(n):
            weighted_sum = 0.0

            for k_idx, offset in enumerate(range(-self.radius, self.radius + 1)):
                sample_idx = min(max(i + offset, 0), n - 1)  # Clamp to boundary
                weighted_sum += labels[sample_idx] * kernel[k_idx]

            result[i] = weighted_sum

        # 2. Apply Label Clipping (Calibration)
        # -------------------------------------
        # Map [0, 1] -> [epsilon, 1-epsilon]
        # Formula: x' = epsilon + x * (1 - 2*epsilon)
        lower = self.epsilon
        scale = 1.0 - 2 * self.epsilon

        for i in range(n):
            # Clip result to essentially step function levels,
            # but soft boundaries
            val = result[i]
            # Linear mapping to safe range
            result[i] = lower + val * scale

        return result

    def __repr__(self) -> str:
        return f"CalibrationSmoother(epsilon={self.epsilon}, sigma={self.sigma})"


def create_smoother(
    smoother_type: LabelSmootherType,
    **kwargs,
) -> LabelSmoother:
    """Factory function to create a LabelSmoother instance.

    Args:
        smoother_type: Type of smoother ("none", "linear", "exponential", "sigmoid", "ramp", "calibration")
        **kwargs: Additional arguments passed to the smoother constructor
            - For "linear": min_segment_length (int, default: 2)
            - For "exponential": alpha (float, default: 0.3)
            - For "sigmoid": min_segment_length (int, default: 2), steepness (float, default: 3.0)
            - For "calibration": epsilon (float, default: 0.05), sigma (float, default: 0.5)

    Returns:
        A LabelSmoother instance

    Examples:
        >>> smoother = create_smoother("none")
        >>> smoother = create_smoother("linear", min_segment_length=2)
        >>> smoother = create_smoother("exponential", alpha=0.5)
        >>> smoother = create_smoother("sigmoid", min_segment_length=2, steepness=3.0)

        >>> labels = [0, 0, 0, 1, 1, 1]
        >>> smoothed = smoother.smooth(labels)
    """
    smoothers: dict[LabelSmootherType, type[LabelSmoother]] = {
        "none": NoSmoother,
        "linear": LinearSmoother,
        "exponential": ExponentialSmoother,
        "sigmoid": SigmoidSmoother,
        "ramp": RampSmoother,
        "calibration": CalibrationSmoother,
    }

    if smoother_type not in smoothers:
        raise ValueError(
            f"Unknown smoother_type: {smoother_type}. "
            f"Must be one of {list(smoothers.keys())}"
        )

    smoother_class = smoothers[smoother_type]
    return smoother_class(**kwargs)
