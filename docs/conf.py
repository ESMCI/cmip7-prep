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

html_theme = "sphinx_book_theme"
html_title = "cmip7-prep"
html_theme_options = {
    "repository_url": "https://github.com/ESMCI/cmip7-prep",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_prev_next": True,
    "show_navbar_depth": 2,
    "secondary_sidebar_items": ["page-toc"],
}
