#!/usr/bin/env python3
"""Build the static project page from a single JSON configuration."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    """Raised when the canonical site configuration is invalid."""


def validate_schema(value: Any, schema: dict | None = None, path: str = "config") -> None:
    """Validate the subset of JSON Schema used by our schema, without dependencies."""
    if schema is None:
        schema = json.loads((ROOT / "site.schema.json").read_text(encoding="utf-8"))
    if "$ref" in schema:
        definitions = json.loads((ROOT / "site.schema.json").read_text(encoding="utf-8"))["$defs"]
        validate_schema(value, definitions[schema["$ref"].removeprefix("#/$defs/")], path)
        return
    kind = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int}
    if kind and (not isinstance(value, types[kind]) or (kind == "integer" and isinstance(value, bool))):
        raise ConfigError(f"{path} must be {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ConfigError(f"{path} must be one of {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise ConfigError(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        for key, child in value.items():
            if key in properties:
                validate_schema(child, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise ConfigError(f"Unknown field: {path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_schema(child, schema.get("items", {}), f"{path}[{index}]")
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ConfigError(f"{path} cannot be empty")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ConfigError(f"{path} has invalid format")
    elif isinstance(value, int) and "minimum" in schema and value < schema["minimum"]:
        raise ConfigError(f"{path} must be at least {schema['minimum']}")


def expand_variables(config: dict[str, Any]) -> dict[str, Any]:
    """Expand {{site.field}} references, detecting typos and reference cycles."""
    pattern = re.compile(r"\{\{site\.([a-z_]+)\}\}")

    def expand(value: Any, stack: tuple = ()) -> Any:
        if isinstance(value, str):
            def replace(match: re.Match) -> str:
                key = match[1]
                replacement = config["site"].get(key)
                if not isinstance(replacement, str):
                    raise ConfigError(f"Unknown text variable: site.{key}")
                if key in stack:
                    raise ConfigError(f"Circular text variable: site.{key}")
                return expand(replacement, (*stack, key))
            return pattern.sub(replace, value)
        if isinstance(value, list):
            return [expand(item, stack) for item in value]
        if isinstance(value, dict):
            return {key: expand(item, stack) for key, item in value.items()}
        return value

    return expand(config)


def validate_urls(config: dict[str, Any]) -> None:
    """Keep local URLs portable; only HTTP(S) URLs can leave the site."""
    def check(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in {"url", "src", "poster", "image", "entry", "favicon", "template_url", "license_url"} and isinstance(item, str) and item:
                    parts = urlsplit(item)
                    if item.startswith(("/", "\\")) or "\\" in item or any(c.isspace() for c in item):
                        raise ConfigError(f"{key}: use a relative path or an absolute HTTP(S) URL: {item}")
                    if parts.scheme and (parts.scheme not in {"http", "https"} or not parts.netloc):
                        raise ConfigError(f"{key}: unsupported URL: {item}")
                # Application options are opaque data belonging to the app.
                if key != "options":
                    check(item)
    check(config)
    if config.get("social_preview", {}).get("enabled", False):
        parts = urlsplit(config["site"].get("url", ""))
        if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment:
            raise ConfigError("site.url must be an absolute HTTP(S) page URL when social_preview is enabled")


def check_assets(output: str) -> None:
    """Optional release check; blank placeholder sources never create URLs."""
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.urls: list[str] = []
            self.ids: set[str] = set()

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                self.ids.add(attrs["id"])
            self.urls.extend(attrs[key] for key in ("src", "href", "poster", "data-entry") if attrs.get(key))

    links = Links()
    links.feed(output)
    for url in links.urls:
        parts = urlsplit(url)
        if parts.scheme or parts.netloc:
            continue
        if parts.path:
            asset = (ROOT / unquote(parts.path)).resolve()
            if not asset.is_relative_to(ROOT) or not asset.is_file():
                raise ConfigError(f"Missing local asset: {url}")
            with asset.open("rb") as stream:
                if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise ConfigError(f"Git LFS asset is a pointer; run git lfs pull: {url}")
        elif parts.fragment and unquote(parts.fragment) not in links.ids:
            raise ConfigError(f"Missing section anchor: {url}")


def esc(value: Any, *, quote: bool = True) -> str:
    return html.escape(str(value), quote=quote)


def required(mapping: dict[str, Any], key: str, context: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "":
        raise ConfigError(f"{context}.{key} is required")
    return value


def placeholder(label: str) -> str:
    return (
        '<div class="module-placeholder" role="status">'
        f"<p>{esc(label)}</p>"
        "</div>"
    )


def render_media(media: dict[str, Any], *, eager: bool = False) -> str:
    kind = media.get("type", "image")
    src = media.get("src", "")
    if not src:
        return placeholder(f"{kind.title()} asset coming soon")
    if kind == "image":
        loading = "eager" if eager else "lazy"
        return f'<img src="{esc(src)}" alt="{esc(media.get("alt", ""))}" loading="{loading}">'
    if kind == "video":
        controls = " controls" if media.get("controls", False) else ""
        poster = f' poster="{esc(media["poster"])}"' if media.get("poster") else ""
        mime = {".webm": "video/webm", ".ogv": "video/ogg"}.get(Path(urlsplit(src).path).suffix, "video/mp4")
        return (
            f'<video autoplay muted loop playsinline{controls}{poster} '
            f'aria-label="{esc(media.get("alt", "Video"))}">'
            f'<source src="{esc(src)}" type="{esc(media.get("mime", mime))}">'
            "Your browser does not support HTML video.</video>"
        )
    raise ConfigError(f"Unsupported media type: {kind}")


def section_shell(section: dict[str, Any], body: str, extra_class: str = "") -> str:
    section_id = section.get("id", section.get("type", "section"))
    title = section.get("title", "")
    heading = f'<h2 class="title is-3">{esc(title)}</h2>' if title else ""
    description = section.get("description", "")
    intro = f'<p class="section-intro">{esc(description)}</p>' if description else ""
    section_class = f"section {extra_class}".strip()
    return f"""<section class="{esc(section_class)}" id="{esc(section_id)}">
  <div class="container is-max-desktop">
    {heading}
    {intro}
    {body}
  </div>
</section>"""


def render_teaser(section: dict[str, Any]) -> str:
    media = render_media(section.get("media", {}), eager=True)
    # Text stays escaped, including substituted configuration variables.
    caption = section.get("caption", "")
    caption_html = f'<p class="subtitle has-text-centered">{esc(caption)}</p>' if caption else ""
    return f"""<section class="hero teaser" id="{esc(section.get('id', 'teaser'))}">
  <div class="container is-max-desktop">
    <div class="hero-body">
      {media}
      {caption_html}
    </div>
  </div>
</section>"""


def render_content(section: dict[str, Any]) -> str:
    align = "has-text-justified" if section.get("align") == "justified" else ""
    body = "".join(f"<p>{esc(paragraph)}</p>" for paragraph in section.get("paragraphs", []))
    return section_shell(section, f'<div class="content {align}">{body or placeholder("Content coming soon")}</div>')


def render_media_section(section: dict[str, Any]) -> str:
    cards: list[str] = []
    for item in section.get("items", []):
        caption = item.get("caption", "")
        caption_html = f"<figcaption>{esc(caption)}</figcaption>" if caption else ""
        cards.append(f'<figure class="media-card">{render_media(item)}{caption_html}</figure>')
    body = f'<div class="media-grid">{"".join(cards)}</div>' if cards else placeholder("Media coming soon")
    return section_shell(section, body)


def render_embed(section: dict[str, Any]) -> str:
    src = section.get("src", "")
    if not src:
        body = placeholder(f'{section.get("title", "Embedded content")} coming soon')
    else:
        height = int(section.get("height", 640))
        body = (
            '<div class="embed-frame">'
            f'<iframe src="{esc(src)}" title="{esc(section.get("title", "Embedded content"))}" '
            f'height="{height}" loading="lazy" allowfullscreen></iframe></div>'
        )
    return section_shell(section, body)


def render_interactive(section: dict[str, Any]) -> str:
    entry = section.get("entry", "")
    export_name = section.get("export", "mount")
    options = json.dumps(section.get("options", {}), separators=(",", ":"))
    attrs = f' data-entry="{esc(entry)}" data-export="{esc(export_name)}" data-options="{esc(options)}"'
    fallback = placeholder("Loading interactive application…") if entry else placeholder("Interactive application coming soon")
    body = (
        f'<div class="interactive-app"{attrs}>'
        f'<div class="interactive-mount">{fallback}</div>'
        '<noscript>Enable JavaScript to use this interactive demo.</noscript>'
        '<p class="interactive-error" role="alert" hidden></p></div>'
    )
    return section_shell(section, body, "interactive-section")


def render_bibtex(section: dict[str, Any]) -> str:
    citation = section.get("citation", "")
    body = f"<pre><code>{esc(citation)}</code></pre>" if citation else placeholder("Citation coming soon")
    return section_shell(section, body)


SECTION_RENDERERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "teaser": render_teaser,
    "content": render_content,
    "media": render_media_section,
    "embed": render_embed,
    "interactive": render_interactive,
    "bibtex": render_bibtex,
}


def render_links(links: list[dict[str, Any]]) -> str:
    buttons: list[str] = []
    for link in links:
        if not link.get("enabled", True):
            continue
        url = link.get("url", "")
        icon = f'<span class="icon"><i class="{esc(link.get("icon", "fas fa-link"))}" aria-hidden="true"></i></span>'
        label = f'<span>{esc(link.get("label", "Link"))}</span>'
        if not url:
            buttons.append(
                '<span class="link-block">'
                '<span class="button is-normal is-rounded is-dark placeholder-link" aria-disabled="true">'
                f'{icon}{label}</span></span>'
            )
            continue
        external = url.startswith(("http://", "https://"))
        target = ' target="_blank" rel="noopener noreferrer"' if external else ""
        buttons.append(
            '<span class="link-block">'
            f'<a href="{esc(url)}" class="external-link button is-normal is-rounded is-dark"{target}>'
            f'{icon}{label}</a></span>'
        )
    return "".join(buttons)


def render_hero(config: dict[str, Any]) -> str:
    site = config["site"]
    authors: list[str] = []
    for author in config.get("authors", []):
        name = esc(required(author, "name", "authors[]"))
        if author.get("url"):
            name = f'<a href="{esc(author["url"])}">{name}</a>'
        markers = ",".join(esc(value) for value in author.get("affiliations", []))
        suffix = f"<sup>{markers}</sup>" if markers else ""
        authors.append(f'<span class="author-block">{name}{suffix}</span>')
    affiliations: list[str] = []
    for affiliation in config.get("affiliations", []):
        marker = f'<sup>{esc(affiliation["id"])}</sup>' if affiliation.get("id") is not None else ""
        affiliations.append(f'<span class="author-block">{marker}{esc(affiliation.get("name", ""))}</span>')
    venue = config.get("venue", "")
    venue_html = f'<div class="publication-venue">{esc(venue)}</div>' if venue else ""
    return f"""<section class="hero publication-header">
  <div class="hero-body">
    <div class="container is-max-desktop has-text-centered">
      <h1 class="title is-1 publication-title">{esc(site["title"])}</h1>
      <h2 class="subtitle is-3 publication-subtitle">{esc(site.get("subtitle", ""))}</h2>
      <div class="is-size-5 publication-authors">{' '.join(authors)}</div>
      <div class="is-size-5 publication-affiliations">{' '.join(affiliations)}</div>
      {venue_html}
      <div class="publication-links">{render_links(config.get("links", []))}</div>
    </div>
  </div>
</section>"""


def render_footer(footer: dict[str, Any]) -> str:
    if not footer.get("enabled", True):
        return ""
    return f"""<footer class="footer">
  <div class="container">
    <div class="content has-text-centered">
      <p>{esc(footer.get("text", "This website is adapted from"))}
        <a href="{esc(footer.get("template_url", "https://github.com/nerfies/nerfies.github.io"))}">{esc(footer.get("template_name", "Nerfies"))}</a>
        and is licensed under
        <a href="{esc(footer.get("license_url", "https://creativecommons.org/licenses/by-sa/4.0/"))}">{esc(footer.get("license_name", "CC BY-SA 4.0"))}</a>.
      </p>
    </div>
  </div>
</footer>"""


def absolute_url(base: str, path: str) -> str:
    if path.startswith(("http://", "https://")):
        return path
    return urljoin(base.rstrip("/") + "/", path.removeprefix("./"))


def render_page(config: dict[str, Any]) -> str:
    validate_schema(config)
    config = expand_variables(config)
    validate_urls(config)
    site = config.get("site", {})
    title = required(site, "title", "site")
    required(site, "description", "site")
    social = config.get("social_preview", {})
    social_meta = ""
    if social.get("enabled", False):
        base_url = required(site, "url", "site (required when social_preview is enabled)")
        image = social.get("image", "")
        image_meta = f'  <meta property="og:image" content="{esc(absolute_url(base_url, image))}">\n' if image else ""
        social_meta = f"""  <meta property="og:title" content="{esc(social.get("title", title))}">
  <meta property="og:description" content="{esc(social.get("description", site["description"]))}">
{image_meta}  <meta property="og:url" content="{esc(base_url)}">
  <meta property="og:type" content="website">
  <meta name="twitter:card" content="{esc(social.get("twitter_card", "summary_large_image"))}">
"""
    rendered_sections: list[str] = []
    section_ids: set[str] = set()
    for index, section in enumerate(config.get("sections", [])):
        if not section.get("enabled", True):
            continue
        section_type = required(section, "type", f"sections[{index}]")
        renderer = SECTION_RENDERERS.get(section_type)
        if renderer is None:
            supported = ", ".join(sorted(SECTION_RENDERERS))
            raise ConfigError(f"sections[{index}].type '{section_type}' is unsupported; choose: {supported}")
        section_id = section.get("id", section_type)
        if section_id in section_ids:
            raise ConfigError(f"Duplicate enabled section id: {section_id}")
        section_ids.add(section_id)
        rendered_sections.append(renderer(section))
    keywords = ", ".join(site.get("keywords", []))
    favicon = site.get("favicon", "./static/images/favicon.svg")
    body = "\n\n".join([render_hero(config), '<main>', *rendered_sections, '</main>', render_footer(config.get("footer", {}))])
    return f"""<!DOCTYPE html>
<!-- Generated by scripts/build_site.py. Edit the canonical JSON configuration, not this file. -->
<html lang="{esc(site.get("language", "en"))}">
<head>
  <meta charset="utf-8">
  <meta name="description" content="{esc(site["description"])}">
  <meta name="keywords" content="{esc(keywords)}">
  <meta name="viewport" content="width=device-width, initial-scale=1">
{social_meta}  <title>{esc(title)} — {esc(site.get("subtitle", "Project Page"))}</title>
  <link rel="icon" href="{esc(favicon)}">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro">
  <link rel="stylesheet" href="./static/css/bulma.min.css">
  <link rel="stylesheet" href="./static/css/index.css">
</head>
<body>
{body}
  <script defer src="./static/js/fontawesome.all.min.js"></script>
  <script type="module" src="./static/js/index.js"></script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "site.config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "index.html")
    parser.add_argument("--check", action="store_true", help="verify index.html is current without writing")
    parser.add_argument("--strict-assets", action="store_true", help="fail on missing local assets or anchors")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        output = render_page(config)
        if args.strict_assets:
            check_assets(output)
    except (OSError, json.JSONDecodeError, ConfigError, TypeError, ValueError) as error:
        print(f"Build failed: {error}", file=sys.stderr)
        return 1
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != output:
            print(f"{args.output} is stale; run: python3 scripts/build_site.py", file=sys.stderr)
            return 1
        print(f"OK: {args.output} matches {args.config}")
        return 0
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    except OSError as error:
        print(f"Build failed: {error}", file=sys.stderr)
        return 1
    print(f"Built {args.output} from {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
