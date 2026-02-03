"""
Percorsi base per Task B: tutti gli script trovano config e data
indipendentemente dalla directory da cui vengono lanciati.
"""
import os

# Cartella taskB/ (parent della cartella code/)
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKB_ROOT = os.path.dirname(_CODE_DIR)


def path(*parts):
    """Costruisce un path assoluto sotto taskB/."""
    return os.path.join(TASKB_ROOT, *parts)
