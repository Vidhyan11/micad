"""Data layer: canonical concepts, per-dataset loaders, unified schema."""
from . import concepts, schema

# Loaders (import lazily-safe: they only touch disk when .load() is called).
from . import derm7pt, pad_ufes, fitzpatrick17k

LOADERS = {
    "derm7pt": derm7pt.load,
    "pad_ufes_20": pad_ufes.load,
    "fitzpatrick17k": fitzpatrick17k.load,
    # PH2 dropped: available mirrors have images XOR concept labels, never both.
}

__all__ = ["concepts", "schema", "derm7pt", "pad_ufes", "fitzpatrick17k", "LOADERS"]
