"""
Ultralytics 8.1 kaller np.trapz i validering; NumPy 2.x fjernet trapz (bruk trapezoid).

Importer denne modulen før `from ultralytics import ...` i skript som trener/validerer.
"""
from __future__ import annotations

import numpy as np

if not hasattr(np, "trapz"):
    if hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid  # type: ignore[method-assign, assignment]
    else:
        raise RuntimeError(
            "numpy mangler både trapz og trapezoid. Installer numpy>=1.26,<2 eller numpy>=2 med trapezoid."
        )
