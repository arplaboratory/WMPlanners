# Project landing page

The landing page is generated from `site.config.json`.

For a local build, install the Sphinx extensions listed in `pyproject.toml`,
then run from the repository root:

```bash
python3 site/scripts/build_site.py --config site/site.config.json --output _site/index.html
sphinx-build -W -b html docs _site/docs
cp -R site/static _site/static
touch _site/.nojekyll
python3 site/scripts/check_site.py _site
python3 -m http.server 8000 -d _site
```

Open `http://localhost:8000/` for the landing page or `/docs/` for the guides.
The deployed project URL is set in `site.config.json` for social previews; update
it if the repository owner or Pages domain changes.

The adapted template files and landing-page content are covered by
[the template license](LICENSE), CC BY-SA 4.0. Changes here include project
content, reduced assets, a project favicon, and the combined Sphinx deployment.
The rest of `wm_planner` remains under its own MIT license.
