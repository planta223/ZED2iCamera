"""Immutable ZED 2i sensor frames and a latest-frame buffer."""

from dataclasses import dataclass
import threading
from typing import Optional

import numpy as np


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    copied = np.array(array, copy=True, order="C")
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class ZED2iFrame:
    """One rectified-left RGB-D observation in application units."""

    source_frame_id: int
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    device_timestamp_ms: Optional[float]
    timestamp_domain: Optional[str]
    host_wall_time_s: float
    host_monotonic_time_s: float

    def __post_init__(self) -> None:
        if isinstance(self.source_frame_id, bool) or not isinstance(
            self.source_frame_id, (int, np.integer)
        ):
            raise TypeError("source_frame_id must be an integer.")
        if self.source_frame_id < 0:
            raise ValueError("source_frame_id must be non-negative.")
        if not isinstance(self.rgb, np.ndarray) or self.rgb.dtype != np.uint8:
            raise TypeError("rgb must be a uint8 NumPy array.")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"rgb shape must be (H, W, 3); got {self.rgb.shape}.")
        if not isinstance(self.depth_m, np.ndarray) or self.depth_m.dtype != np.float32:
            raise TypeError("depth_m must be a float32 NumPy array.")
        if self.depth_m.ndim != 2 or self.rgb.shape[:2] != self.depth_m.shape:
            raise ValueError("RGB and depth resolutions must match.")
        if not isinstance(self.K, np.ndarray) or self.K.shape != (3, 3):
            raise ValueError("K must be a 3x3 NumPy array.")
        if not np.issubdtype(self.K.dtype, np.floating) or not np.all(
            np.isfinite(self.K)
        ):
            raise TypeError("K must contain finite floating-point values.")
        if self.device_timestamp_ms is not None and not np.isfinite(
            self.device_timestamp_ms
        ):
            raise ValueError("device_timestamp_ms must be finite when provided.")
        if self.timestamp_domain is not None and not isinstance(
            self.timestamp_domain, str
        ):
            raise TypeError("timestamp_domain must be a string or None.")
        if not np.isfinite(self.host_wall_time_s) or not np.isfinite(
            self.host_monotonic_time_s
        ):
            raise ValueError("Host timestamps must be finite.")

        object.__setattr__(self, "rgb", _readonly_copy(self.rgb))
        object.__setattr__(self, "depth_m", _readonly_copy(self.depth_m))
        object.__setattr__(self, "K", _readonly_copy(self.K))

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


class LatestFrameBuffer:
    """Thread-safe single slot retaining only the newest ZED frame."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: Optional[ZED2iFrame] = None

    def publish(self, frame: ZED2iFrame) -> None:
        if not isinstance(frame, ZED2iFrame):
            raise TypeError("frame must be a ZED2iFrame instance.")
        with self._condition:
            if (
                self._latest is not None
                and frame.source_frame_id <= self._latest.source_frame_id
            ):
                raise ValueError(
                    "Published source_frame_id must increase strictly: "
                    f"latest={self._latest.source_frame_id}, "
                    f"received={frame.source_frame_id}."
                )
            self._latest = frame
            self._condition.notify_all()

    def clear(self) -> None:
        with self._condition:
            self._latest = None
            self._condition.notify_all()

    def wait_for_newer(
        self,
        frame_id: int,
        timeout_s: Optional[float] = None,
    ) -> Optional[ZED2iFrame]:
        with self._condition:
            available = self._condition.wait_for(
                lambda: (
                    self._latest is not None
                    and self._latest.source_frame_id > frame_id
                ),
                timeout=timeout_s,
            )
            return self._latest if available else None
