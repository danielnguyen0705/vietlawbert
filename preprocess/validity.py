"""Conservative eligibility for current-law retrieval, using source metadata."""
from datetime import date, datetime
from zoneinfo import ZoneInfo
import unicodedata


def is_currently_effective(metadata, today=None):
    today = today or datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date()
    status = unicodedata.normalize("NFC", str(metadata.get("status") or "")).strip().casefold()
    # Unknown, future, suspended and partially expired documents need review.
    if status != "còn hiệu lực":
        return False
    try:
        effective = date.fromisoformat(str(metadata.get("effective_date") or "")[:10])
    except ValueError:
        return False
    return effective <= today
