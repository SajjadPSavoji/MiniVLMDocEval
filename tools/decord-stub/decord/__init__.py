"""Local mac-arm64 stub for the `decord` package.

Why this exists
---------------
`decord` (video decoding) has no macOS-arm64 wheel, but VLMEvalKit imports it
eagerly through two *video* dataset modules (`dsrbench`, `stibench`). This
project evaluates **image-only** document benchmarks, so video is never decoded
locally. This stub satisfies the import so `import vlmeval` works for local
smoke tests.

Scope
-----
- Local mac development ONLY. On Linux/Colab install the real `decord`.
- Any attempt to actually decode video raises a clear error rather than
  silently returning wrong data.
"""

__version__ = "0.6.0"


class _Unavailable:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "decord is stubbed on mac-arm64 (no upstream wheel). Video decoding "
            "is unavailable locally; this project only evaluates image benchmarks. "
            "Install the real `decord` on Linux/Colab to use video benchmarks."
        )


class VideoReader(_Unavailable):
    pass


def cpu(*args, **kwargs):
    return None


def gpu(*args, **kwargs):
    return None


class bridge:
    @staticmethod
    def set_bridge(*args, **kwargs):
        return None
