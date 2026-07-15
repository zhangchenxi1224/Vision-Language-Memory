# Third-party source references

DreamLite is checked out at the immutable commit recorded in models.lock.json. Treat that working tree as read-only.
All differentiable training changes belong under src/vision_memory rather than in the upstream repository.
Recreate the checkout on a new machine with `python scripts/bootstrap/fetch_sources.py`; the script refuses to
overwrite a dirty or non-Git destination.

PrefEval-GPT56 is an existing nested repository with user changes and is intentionally left in place at the project
root. The top-level Git repository ignores it.
