# SPDX-License-Identifier: GPL-3.0-only
# Vendored third-party helper. Copyleft: excluded from FIM training, still searchable with its licence tag.


def dumps_fast(obj) -> str:
    import json

    return json.dumps(obj, separators=(",", ":"))
