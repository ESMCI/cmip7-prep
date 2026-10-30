"""Sphinx configuration for the cmip7-prep user's guide."""

project = "cmip7-prep"
project_copyright = "2026, NorESM"
author = "NorESM"

extensions = ["myst_parser", "sphinx_copybutton"]

myst_enable_extensions = ["colon_fence", "deflist"]

templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]
exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"
html_title = "cmip7-prep"
html_theme_options = {
    "github_url": "https://github.com/ESMCI/cmip7-prep",
    "use_edit_page_button": True,
    "show_prev_next": True,
    # How deep the page headings in the left sidebar go, and how many of
    # those levels are expanded rather than needing a click.
    "navigation_depth": 3,
    "show_toc_level": 2,
    # Nothing on the right, so there is one column to read rather than two.
    "secondary_sidebar_items": [],
}

# The top navigation bar already lists every page, because the guide is one
# flat set of pages with nothing nested under them. The left sidebar therefore
# holds the headings of the page being read, and the link that opens it on
# GitHub.
html_sidebars = {
    "**": ["page-toc", "edit-this-page"],
}

# Needed by use_edit_page_button, which builds the link to GitHub from these.
html_context = {
    "github_user": "ESMCI",
    "github_repo": "cmip7-prep",
    "github_version": "main",
    "doc_path": "docs",
}
