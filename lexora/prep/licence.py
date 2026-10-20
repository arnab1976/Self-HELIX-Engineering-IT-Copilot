"""Licence detection (ScanCode Toolkit in production). Copyleft code is excluded from FIM training only."""
from __future__ import annotations

import re
from dataclasses import dataclass

SPDX = re.compile(r"SPDX-License-Identifier:\s*([\w.+-]+)")
TEXT_HINTS = [
    (re.compile(r"GNU AFFERO GENERAL PUBLIC LICENSE", re.I), "AGPL-3.0"),
    (re.compile(r"GNU LESSER GENERAL PUBLIC LICENSE", re.I), "LGPL"),
    (re.compile(r"GNU GENERAL PUBLIC LICENSE", re.I), "GPL"),
    (re.compile(r"Mozilla Public License", re.I), "MPL-2.0"),
    (re.compile(r"Apache License,? Version 2\.0", re.I), "Apache-2.0"),
    (re.compile(r"Permission is hereby granted, free of charge", re.I), "MIT"),
]
COPYLEFT = re.compile(r"GPL|AGPL|LGPL|MPL|EUPL|CDDL", re.I)


@dataclass
class LicenceResult:
    licence: str | None
    copyleft: bool
    third_party: bool

    @property
    def allow_fim(self) -> bool:
        return not self.copyleft

    @property
    def allow_rag(self) -> bool:
        return True   # shown with its licence tag, not memorised


def detect(text: str, path: str = "") -> LicenceResult:
    m = SPDX.search(text[:2000])
    lic = m.group(1) if m else None
    if not lic:
        for rx, name in TEXT_HINTS:
            if rx.search(text[:5000]):
                lic = name
                break
    third_party = bool(re.search(r"(^|/)(vendor|third_party|node_modules)/", path)) or lic is not None
    return LicenceResult(lic, bool(lic and COPYLEFT.search(lic)), third_party)
