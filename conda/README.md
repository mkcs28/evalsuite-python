# Publishing on conda-forge

1. After a release is on PyPI, copy `conda/recipe.yaml` to `recipes/evalsuite-python/recipe.yaml` in a fork of
   [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes).
2. Set `version`, and `sha256` to the hash of `evalsuite_python-<version>.tar.gz` from PyPI's "Download files" page.
3. Open the pull request; a conda-forge reviewer checks it (usually within days to a few weeks).
4. Once merged, conda-forge creates the `evalsuite-python-feedstock` repository with you as maintainer, and its
   bot opens a pull request automatically for every new PyPI release.

Users then install with `conda install -c conda-forge evalsuite-python`.
