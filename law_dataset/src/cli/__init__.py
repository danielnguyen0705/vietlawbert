"""Các entrypoint vận hành thống nhất cho VietLawBERT."""

import sys


# PowerShell có thể dùng CP1258 nhưng không biểu diễn hết ký tự tiếng Việt.
# Chuẩn hóa output CLI sang UTF-8; trên Ubuntu đây là no-op về mặt hiển thị.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")
