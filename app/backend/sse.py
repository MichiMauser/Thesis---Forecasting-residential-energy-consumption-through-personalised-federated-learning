from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Optional

from fastapi import Request

from . import data_sources as ds

POLL_SECONDS = 1.0


def _sse(payload: dict, event: Optional[str] = None) -> str:
    line = f"event: {event}\n" if event else ""
    return f"{line}data: {json.dumps(payload)}\n\n"


async def event_stream(request: Request,
                       run_id: Optional[str] = None) -> AsyncGenerator[str, None]:
    records = ds.read_run_log()
    target  = run_id or (records[-1]["run_id"] if records else None)

    yield _sse({"run_id": target}, event="run")

    # recuperare: retrimitem înregistrările deja existente
    sent = 0
    for rec in records:
        if target is None:
            target = rec.get("run_id")
        if rec.get("run_id") == target:
            yield _sse(rec)
    sent = len(records)

    # urmărire live a fișierului
    idle = 0
    while True:
        if await request.is_disconnected():
            break
        records = ds.read_run_log()
        new = records[sent:]
        sent = len(records)
        if new:
            idle = 0
            for rec in new:
                if target is None:
                    target = rec.get("run_id")
                if rec.get("run_id") == target:
                    yield _sse(rec)
        else:
            idle += 1
            if idle % 15 == 0:                # la ~15s, ca să nu se închidă conexiunea
                yield ": keepalive\n\n"
        await asyncio.sleep(POLL_SECONDS)
