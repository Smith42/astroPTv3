# Configuration file for the Sphinx documentation builder.
#
# The source directory is `astro/` itself, so the top-level prose
# (README.md, PLAN.md, EXPERIMENTS.md) is part of the site alongside
# docs/ guides and ADRs. Markdown-only (MyST); no autodoc -- the corpus
# is prose, and building API docs would drag the torch stack into the
# Read the Docs environment.

# Project information
project = "AstroPTv3"
copyright = "2026 Michael J. Smith"
author = "Michael J. Smith"
release = "0.1.0"

# Extensions
extensions = [
    "sphinx.ext.mathjax",
    "myst_parser",
]

# Markdown only
source_suffix = {
    ".md": "markdown",
}

# MyST: the PLAN/ADR corpus uses $...$ math, definition lists, and the
# README uses an HTML <img> for the centred logo
myst_enable_extensions = ["dollarmath", "amsmath", "deflist", "html_image"]
myst_heading_anchors = 3

# Never treat code or configs as documents
exclude_patterns = [
    ".venv",
    ".pytest_cache",
    "src",
    "tests",
    "configs",
    "scripts",
    "wandb",
    "docs/requirements.txt",
]

# Repo-relative links (GitHub-style) are used throughout the corpus;
# they are not Sphinx document targets and that is fine
suppress_warnings = ["myst.xref_missing"]

# Theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
    "style_external_links": True,
    "navigation_depth": 4,
}

# The shoggoth sticker checked in by the logo PR (astro/assets/)
html_logo = "assets/shoggoth_telescope_sticker_2.png"
html_favicon = "assets/shoggoth_telescope_sticker_2.png"
