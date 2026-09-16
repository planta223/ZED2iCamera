"""Threaded Stereolabs ZED 2i RGB-D acquisition."""

from dataclasses import dataclass
import threading
import time
from types import ModuleType
from typing import Optional

import numpy as np

from .frame import LatestFrameBuffer, ZED2iFrame


_START_TIMEOUT_S = 20.0
_FRAME_WAIT_TIMEOUT_S = 2.0
_STOP_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class ZED2iConfig:
    """ZED 2i stream and depth selection."""

    resolution: str = "HD720"
    fps: int = 30
    serial: Optional[str] = None
    depth_mode: str = "NEURAL"


@dataclass(frozen=True)
class ZED2iStreamInfo:
    width: int
    height: int
    fps: int
    format: str


class ZED2iCamera:
    """Publish rectified-left RGB and aligned metric depth from one thread."""

    def __init__(
        self,
        config: ZED2iConfig,
        frame_buffer: Optional[LatestFrameBuffer] = None,
    ) -> None:
        if not isinstance(config, ZED2iConfig):
            raise TypeError("config must be a ZED2iConfig instance.")
        if not config.resolution or not isinstance(config.resolution, str):
            raise ValueError("ZED resolution must be a non-empty enum name.")
        if not config.depth_mode or not isinstance(config.depth_mode, str):
            raise ValueError("ZED depth_mode must be a non-empty enum name.")
        if config.fps <= 0:
            raise ValueError("ZED FPS must be positive.")
        if config.serial is not None:
            if not isinstance(config.serial, str) or not config.serial.isdigit():
                raise ValueError("ZED serial must contain decimal digits.")

        self.config = config
        self.frame_buffer = frame_buffer or LatestFrameBuffer()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._producer_error: Optional[BaseException] = None
        self._last_returned_frame_id = -1
        self._device_name: Optional[str] = None
        self._device_serial: Optional[str] = None
        self._color_stream_info: Optional[ZED2iStreamInfo] = None
        self._depth_stream_info: Optional[ZED2iStreamInfo] = None
        self._K: Optional[np.ndarray] = None

    @staticmethod
    def _load_sdk() -> ModuleType:
        try:
            import pyzed.sl as sl
        except ImportError as error:
            raise ImportError(
                "pyzed.sl is required for live ZED 2i input. Install the "
                "Stereolabs ZED SDK and its Python API in this environment."
            ) from error
        return sl

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def producer_error(self) -> Optional[BaseException]:
        with self._state_lock:
            return self._producer_error

    @property
    def device_name(self) -> Optional[str]:
        with self._state_lock:
            return self._device_name

    @property
    def device_serial(self) -> Optional[str]:
        with self._state_lock:
            return self._device_serial

    @property
    def depth_scale(self) -> float:
        return 1.0

    @property
    def color_stream_info(self) -> Optional[ZED2iStreamInfo]:
        with self._state_lock:
            return self._color_stream_info

    @property
    def depth_stream_info(self) -> Optional[ZED2iStreamInfo]:
        with self._state_lock:
            return self._depth_stream_info

    def raise_if_failed(self) -> None:
        error = self.producer_error
        if error is not None:
            raise RuntimeError("ZED 2i acquisition thread failed.") from error

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("ZED2iCamera is already running.")
            self._producer_error = None
            self._device_name = None
            self._device_serial = None
            self._color_stream_info = None
            self._depth_stream_info = None
            self._K = None
            self._last_returned_frame_id = -1
            self._stop_event.clear()
            self._started_event.clear()
            self.frame_buffer.clear()
            self._thread = threading.Thread(
                target=self._acquisition_loop,
                name="zed2i-acquisition",
                daemon=True,
            )
            thread = self._thread

        thread.start()
        if not self._started_event.wait(timeout=_START_TIMEOUT_S):
            self._stop_event.set()
            raise TimeoutError(
                f"ZED camera did not start within {_START_TIMEOUT_S:.0f} s."
            )
        self.raise_if_failed()

    def stop(self) -> None:
        with self._state_lock:
            thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=_STOP_TIMEOUT_S)
        if thread.is_alive():
            raise RuntimeError(
                f"ZED acquisition did not stop within {_STOP_TIMEOUT_S:.0f} s."
            )
        with self._state_lock:
            if self._thread is thread:
                self._thread = None

    def get_next_frame(self) -> ZED2iFrame:
        if not self.is_running:
            self.raise_if_failed()
            raise RuntimeError("ZED2iCamera.start() must be called first.")
        frame = self.frame_buffer.wait_for_newer(
            self._last_returned_frame_id,
            timeout_s=_FRAME_WAIT_TIMEOUT_S,
        )
        if frame is None:
            self.raise_if_failed()
            raise TimeoutError("Timed out waiting for a ZED frame.")
        self._last_returned_frame_id = frame.source_frame_id
        return frame

    @staticmethod
    def _intrinsic_matrix(left_camera_parameters: object) -> np.ndarray:
        return np.array(
            [
                [left_camera_parameters.fx, 0.0, left_camera_parameters.cx],
                [0.0, left_camera_parameters.fy, left_camera_parameters.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _make_frame(
        source_frame_id: int,
        image_bgra: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
        device_timestamp_ms: Optional[float],
        host_wall_time_s: float,
        host_monotonic_time_s: float,
    ) -> ZED2iFrame:
        image_bgra = np.asarray(image_bgra)
        if (
            image_bgra.dtype != np.uint8
            or image_bgra.ndim != 3
            or image_bgra.shape[2] != 4
        ):
            raise ValueError(
                "Unexpected ZED left image: "
                f"shape={image_bgra.shape}, dtype={image_bgra.dtype}."
            )
        depth_values = np.asarray(depth_m)
        if depth_values.ndim != 2 or depth_values.shape != image_bgra.shape[:2]:
            raise ValueError(
                "ZED RGB/depth resolution mismatch: "
                f"image={image_bgra.shape[:2]}, depth={depth_values.shape}."
            )

        # VIEW.LEFT is BGRA. The application contract is three-channel RGB.
        rgb = np.ascontiguousarray(image_bgra[..., 2::-1], dtype=np.uint8)
        normalized_depth = np.array(
            depth_values,
            dtype=np.float32,
            copy=True,
            order="C",
        )
        invalid = ~np.isfinite(normalized_depth) | (normalized_depth < 0.001)
        normalized_depth[invalid] = np.float32(0.0)

        return ZED2iFrame(
            source_frame_id=source_frame_id,
            rgb=rgb,
            depth_m=normalized_depth,
            K=K,
            device_timestamp_ms=device_timestamp_ms,
            timestamp_domain="zed_image_clock",
            host_wall_time_s=host_wall_time_s,
            host_monotonic_time_s=host_monotonic_time_s,
        )

    def _store_metadata(self, camera_info: object) -> None:
        configuration = camera_info.camera_configuration
        resolution = configuration.resolution
        K = self._intrinsic_matrix(
            configuration.calibration_parameters.left_cam
        )
        width = int(resolution.width)
        height = int(resolution.height)
        fps = int(configuration.fps)
        with self._state_lock:
            self._device_name = str(camera_info.camera_model)
            self._device_serial = str(camera_info.serial_number)
            self._color_stream_info = ZED2iStreamInfo(
                width, height, fps, "RGB8"
            )
            self._depth_stream_info = ZED2iStreamInfo(
                width, height, fps, "F32_M"
            )
            self._K = K

    def _acquisition_loop(self) -> None:
        zed = None
        opened = False
        try:
            sl = self._load_sdk()
            zed = sl.Camera()
            init = sl.InitParameters()
            try:
                init.camera_resolution = getattr(
                    sl.RESOLUTION, self.config.resolution.upper()
                )
            except AttributeError as error:
                raise ValueError(
                    f"Unsupported ZED resolution: {self.config.resolution!r}."
                ) from error
            try:
                init.depth_mode = getattr(sl.DEPTH_MODE, self.config.depth_mode.upper())
            except AttributeError as error:
                raise ValueError(
                    f"Unsupported ZED depth mode: {self.config.depth_mode!r}."
                ) from error
            init.camera_fps = self.config.fps
            init.coordinate_units = sl.UNIT.METER
            init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
            if self.config.serial is not None:
                init.set_from_serial_number(int(self.config.serial))

            open_status = zed.open(init)
            if open_status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Failed to open ZED camera: {open_status}.")
            opened = True
            self._store_metadata(zed.get_camera_information())
            self._started_event.set()

            runtime = sl.RuntimeParameters()
            image = sl.Mat()
            depth = sl.Mat()
            frame_id = 0
            while not self._stop_event.is_set():
                grab_status = zed.grab(runtime)
                if grab_status != sl.ERROR_CODE.SUCCESS:
                    if self._stop_event.is_set():
                        break
                    if grab_status > sl.ERROR_CODE.SUCCESS:
                        raise RuntimeError(f"ZED grab failed: {grab_status}.")
                    # Negative codes are SDK warnings such as a corrupted frame.
                    continue

                host_wall_time_s = time.time()
                host_monotonic_time_s = time.monotonic()
                image_status = zed.retrieve_image(
                    image, sl.VIEW.LEFT, sl.MEM.CPU
                )
                if (
                    image_status is not None
                    and image_status != sl.ERROR_CODE.SUCCESS
                ):
                    raise RuntimeError(
                        f"ZED left-image retrieval failed: {image_status}."
                    )
                depth_status = zed.retrieve_measure(
                    depth, sl.MEASURE.DEPTH, sl.MEM.CPU
                )
                if (
                    depth_status is not None
                    and depth_status != sl.ERROR_CODE.SUCCESS
                ):
                    raise RuntimeError(
                        f"ZED depth retrieval failed: {depth_status}."
                    )
                timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                timestamp_ms = float(timestamp.get_nanoseconds()) / 1_000_000.0
                with self._state_lock:
                    K = self._K
                if K is None:
                    raise RuntimeError("ZED intrinsic matrix was not initialized.")
                frame = self._make_frame(
                    source_frame_id=frame_id,
                    image_bgra=image.get_data(),
                    depth_m=depth.get_data(),
                    K=K,
                    device_timestamp_ms=timestamp_ms,
                    host_wall_time_s=host_wall_time_s,
                    host_monotonic_time_s=host_monotonic_time_s,
                )
                self.frame_buffer.publish(frame)
                frame_id += 1
        except BaseException as error:
            with self._state_lock:
                self._producer_error = error
            self._started_event.set()
        finally:
            if zed is not None and opened:
                try:
                    zed.close()
                except BaseException as error:
                    with self._state_lock:
                        if self._producer_error is None:
                            self._producer_error = error
            self._started_event.set()
