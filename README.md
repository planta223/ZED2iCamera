# ZED2iCamera

Standalone Stereolabs ZED 2i RGB-D acquisition package. It publishes immutable
left-rectified RGB frames, left-aligned float32 depth in meters, the matching
rectified-left intrinsic matrix, and image timestamps. Acquisition runs in one
producer thread with a single-slot latest-frame buffer.

## ZED SDK prerequisite

`pyzed` is supplied by the Stereolabs ZED SDK rather than as a normal portable
PyPI dependency. Install a ZED SDK release compatible with the host CUDA,
driver, operating system, and Python version. On Linux, install its Python API
into the environment that runs FoundationPose:

```bash
cd /usr/local/zed
python3 get_python_api.py
python3 -c "import pyzed.sl as sl; print('pyzed OK')"
```

Then install this package:

```bash
cd /home/kkb/Workspace/MULTI_OBJECT_TRACKING
python3 -m pip install -e ../ZED2iCamera
```

The default profile is `HD720`, 30 FPS, `NEURAL` depth, `UNIT.METER`, and
`COORDINATE_SYSTEM.IMAGE`. Supported configuration names are resolved against
the enums exposed by the installed ZED SDK, so an unsupported SDK/profile
fails explicitly at startup.
