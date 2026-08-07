import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(*parts):
    """Builds an absolute path from the repository root."""
    return os.path.join(REPO_ROOT, *parts)
