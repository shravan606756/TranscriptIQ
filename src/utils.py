import re
from urllib.parse import urlparse, parse_qs


def video_id_from_url(url: str) -> str:
    """
    Extract the YouTube video ID from a URL and return it as a canonical storage key.

    Supports the common URL forms:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID

    Raises ValueError if the video ID cannot be determined.
    """
    parsed = urlparse(url)

    # youtu.be short links
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        vid = parsed.path.lstrip("/").split("/")[0]
        if vid:
            return vid

    # Standard watch URL  (?v=...)
    if parsed.path == "/watch":
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]

    # /shorts/<id>, /embed/<id>, /v/<id>
    m = re.match(r"^/(?:shorts|embed|v)/([A-Za-z0-9_-]+)", parsed.path)
    if m:
        return m.group(1)

    raise ValueError(f"Cannot extract YouTube video ID from URL: {url!r}")
