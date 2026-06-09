from __future__ import annotations

import io
import os
import sys
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path


class _TeeStream(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    @property
    def encoding(self) -> str:
        return getattr(self._streams[0], "encoding", "utf-8")


class RunLogger(AbstractContextManager["RunLogger"]):
    def __init__(self, log_dir: Path, run_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{timestamp}_{run_name}.log"
        self._file = self.log_path.open("w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._tee_out = _TeeStream(self._stdout, self._file)
        self._tee_err = _TeeStream(self._stderr, self._file)

    def __enter__(self) -> "RunLogger":
        sys.stdout = self._tee_out
        sys.stderr = self._tee_err
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._file.close()
        return None


def start_run_logging(*, project_root: Path, log_group: str, run_name: str) -> RunLogger:
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name).strip("_")
    safe_name = safe_name or "run"
    return RunLogger(project_root / "logs" / log_group, safe_name)


def command_string() -> str:
    return " ".join([os.path.basename(sys.executable), *sys.argv])
