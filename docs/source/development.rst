.. _development:

========================
Contributing to Borealis
========================

Development on the Borealis software is both appreciated and encouraged. If you have new features you would like added,
please reach out via email to superdarn@usask.ca and we would be happy to work with you to implement your request.

----------------------
Contribution Standards
----------------------

``pre-commit`` hooks are used to enforce coding standards for this project. To use ``pre-commit`` in your environment,
install ``cppcheck`` and ``clang-format`` with your native package manager and install ``borealis[dev]`` in your
Python virtual environment. Now, every time you commit, ``pre-commit`` will run the hooks for this project and only
proceed with the commit if all hooks pass. Note that some hooks modify the formatting of files, so the hook will fail
if it executes any reformatting. Inspect the changes before attempting to commit again; in most instances, the hooks
should pass on the second attempt without you needing to modify any files.
