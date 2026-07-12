import resource
import sys


def rss_mb() -> float:
    """Resident set size in MB -- temporary instrumentation to find exactly
    which stage spikes memory on Render's 512MB free tier, before deciding
    whether a batching/streaming change is actually needed there.
    ru_maxrss units differ by platform: KB on Linux (Render's container),
    bytes on macOS."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1024 if sys.platform == "linux" else raw / 1024 / 1024
