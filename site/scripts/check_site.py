#!/usr/bin/env python3
"""Check that the combined Pages artifact has working local navigation."""

from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        self.urls.extend(values[key] for key in ("href", "src") if values.get(key))


def main(root: Path) -> int:
    root = root.resolve()
    landing = root / "index.html"
    docs = root / "docs" / "index.html"
    if not landing.is_file() or not docs.is_file():
        raise SystemExit("Missing landing page or Sphinx documentation index")

    links = Links()
    links.feed(landing.read_text(encoding="utf-8"))
    if "./docs/index.html" not in links.urls:
        raise SystemExit("Landing page has no documentation link")
    for url in links.urls:
        parts = urlsplit(url)
        if parts.scheme or parts.netloc or not parts.path:
            continue
        target = (root / unquote(parts.path)).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise SystemExit(f"Broken local landing-page link: {url}")

    docs_links = Links()
    docs_links.feed(docs.read_text(encoding="utf-8"))
    if "../index.html" not in docs_links.urls:
        raise SystemExit("Sphinx index has no project-home link")
    print("Combined site links are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
