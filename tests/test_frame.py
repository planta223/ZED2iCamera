import numpy as np
import pytest

from zed2i_camera import LatestFrameBuffer, ZED2iFrame


def make_frame(frame_id: int) -> ZED2iFrame:
    return ZED2iFrame(
        source_frame_id=frame_id,
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_m=np.ones((2, 3), dtype=np.float32),
        K=np.eye(3, dtype=np.float64),
        device_timestamp_ms=12.5,
        timestamp_domain="zed_image_clock",
        host_wall_time_s=20.0,
        host_monotonic_time_s=10.0,
    )


def test_frame_owns_immutable_arrays() -> None:
    frame = make_frame(1)
    assert frame.rgb.flags.writeable is False
    assert frame.depth_m.flags.writeable is False
    assert frame.K.flags.writeable is False
    assert frame.width == 3
    assert frame.height == 2


def test_latest_frame_buffer_keeps_newest_frame() -> None:
    buffer = LatestFrameBuffer()
    first = make_frame(1)
    latest = make_frame(2)
    buffer.publish(first)
    buffer.publish(latest)
    assert buffer.wait_for_newer(0, timeout_s=0.0) is latest
    assert buffer.wait_for_newer(2, timeout_s=0.0) is None
    with pytest.raises(ValueError, match="increase strictly"):
        buffer.publish(first)
