import numpy as np
from types import SimpleNamespace

from zed2i_camera import ZED2iCamera, ZED2iConfig


class FakeLeftParameters:
    fx = 700.0
    fy = 710.0
    cx = 640.0
    cy = 360.0


def test_default_configuration() -> None:
    assert ZED2iConfig() == ZED2iConfig(
        resolution="HD720",
        fps=30,
        serial=None,
        depth_mode="NEURAL",
    )


def test_intrinsic_matrix_uses_rectified_left_parameters() -> None:
    np.testing.assert_allclose(
        ZED2iCamera._intrinsic_matrix(FakeLeftParameters()),
        [[700.0, 0.0, 640.0], [0.0, 710.0, 360.0], [0.0, 0.0, 1.0]],
    )


def test_bgra_and_metric_depth_conversion() -> None:
    bgra = np.array(
        [[[1, 2, 3, 255], [4, 5, 6, 255]], [[7, 8, 9, 255], [10, 11, 12, 255]]],
        dtype=np.uint8,
    )
    depth = np.array([[1.25, np.nan], [np.inf, -1.0]], dtype=np.float32)
    frame = ZED2iCamera._make_frame(
        source_frame_id=4,
        image_bgra=bgra,
        depth_m=depth,
        K=np.eye(3, dtype=np.float64),
        device_timestamp_ms=100.0,
        host_wall_time_s=2.0,
        host_monotonic_time_s=1.0,
    )

    np.testing.assert_array_equal(frame.rgb[0, 0], [3, 2, 1])
    assert frame.rgb.dtype == np.uint8
    assert frame.rgb.shape == (2, 2, 3)
    assert frame.depth_m.dtype == np.float32
    np.testing.assert_array_equal(frame.depth_m, [[1.25, 0.0], [0.0, 0.0]])
    assert frame.K.shape == (3, 3)
    assert frame.device_timestamp_ms == 100.0


def test_mock_sdk_start_configures_metric_left_aligned_stream(monkeypatch) -> None:
    class FakeMat:
        def __init__(self):
            self.data = None

        def get_data(self):
            return self.data

    class FakeInitParameters:
        def __init__(self):
            self.camera_resolution = None
            self.camera_fps = None
            self.depth_mode = None
            self.coordinate_units = None
            self.coordinate_system = None
            self.serial = None

        def set_from_serial_number(self, serial):
            self.serial = serial

    class FakeTimestamp:
        def get_nanoseconds(self):
            return 5_000_000

    left = FakeLeftParameters()
    configuration = SimpleNamespace(
        resolution=SimpleNamespace(width=2, height=2),
        fps=30,
        calibration_parameters=SimpleNamespace(left_cam=left),
    )

    class FakeCamera:
        latest = None

        def __init__(self):
            self.init = None
            self.closed = False
            FakeCamera.latest = self

        def open(self, init):
            self.init = init
            return 0

        def close(self):
            self.closed = True

        def get_camera_information(self):
            return SimpleNamespace(
                camera_configuration=configuration,
                camera_model="ZED 2i",
                serial_number=123456,
            )

        def grab(self, runtime):
            del runtime
            return 0

        def retrieve_image(self, mat, view, memory):
            del view, memory
            mat.data = np.zeros((2, 2, 4), dtype=np.uint8)
            return 0

        def retrieve_measure(self, mat, measure, memory):
            del measure, memory
            mat.data = np.ones((2, 2), dtype=np.float32)
            return 0

        def get_timestamp(self, reference):
            del reference
            return FakeTimestamp()

    fake_sdk = SimpleNamespace(
        Camera=FakeCamera,
        InitParameters=FakeInitParameters,
        RuntimeParameters=object,
        Mat=FakeMat,
        RESOLUTION=SimpleNamespace(HD720="HD720"),
        DEPTH_MODE=SimpleNamespace(NEURAL="NEURAL"),
        UNIT=SimpleNamespace(METER="METER"),
        COORDINATE_SYSTEM=SimpleNamespace(IMAGE="IMAGE"),
        ERROR_CODE=SimpleNamespace(SUCCESS=0),
        VIEW=SimpleNamespace(LEFT="LEFT"),
        MEASURE=SimpleNamespace(DEPTH="DEPTH"),
        MEM=SimpleNamespace(CPU="CPU"),
        TIME_REFERENCE=SimpleNamespace(IMAGE="IMAGE"),
    )
    monkeypatch.setattr(ZED2iCamera, "_load_sdk", staticmethod(lambda: fake_sdk))
    camera = ZED2iCamera(
        ZED2iConfig(
            resolution="HD720",
            fps=30,
            serial="123456",
            depth_mode="NEURAL",
        )
    )

    camera.start()
    frame = camera.get_next_frame()
    camera.stop()

    assert frame.rgb.shape == (2, 2, 3)
    assert frame.depth_m.dtype == np.float32
    assert FakeCamera.latest.init.camera_resolution == "HD720"
    assert FakeCamera.latest.init.depth_mode == "NEURAL"
    assert FakeCamera.latest.init.coordinate_units == "METER"
    assert FakeCamera.latest.init.coordinate_system == "IMAGE"
    assert FakeCamera.latest.init.serial == 123456
    assert FakeCamera.latest.closed is True
