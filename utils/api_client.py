"""
api_client.py - Client kết nối cổng dữ liệu VBPL có cơ chế Pooling và Chịu lỗi.
Tích hợp urllib3 Retry Adapter để xử lý nghẽn mạng và Rate Limiting.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger("VietLawBERT_APIClient")


class VBPLApiClient:
    """Client giao tiếp HTTP chuẩn hóa kết nối với Gateway dữ liệu của Bộ Tư pháp (VBPL)."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        self.base_url = (base_url or "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc").rstrip("/")
        self.timeout = timeout
        self.session = self._init_resilient_session()

    @staticmethod
    def _init_resilient_session() -> requests.Session:
        """Khởi tạo requests.Session với Connection Pooling và cơ chế tự động thử lại."""
        session = requests.Session()
        retry_strategy = Retry(
            total=4,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,
            pool_maxsize=50,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Origin": "https://vbpl.vn",
            "Referer": "https://vbpl.vn/",
        })
        return session

    def get_document_detail(self, item_id: str | int) -> Optional[Dict[str, Any]]:
        """Lấy siêu dữ liệu chi tiết và toàn văn HTML của một văn bản theo Document ID."""
        clean_id = str(item_id).strip()
        url = f"{self.base_url}/{clean_id}"
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            logger.warning(f"API Chi tiết trả về status {response.status_code} cho văn bản ID {clean_id}")
            return None
        except requests.exceptions.RequestException as exc:
            logger.error(f"Lỗi mạng khi tải chi tiết văn bản {clean_id}: {exc}")
            return None

    def get_diagram(self, item_id: str | int) -> Optional[Dict[str, Any]]:
        """Lấy dữ liệu JSON Lược đồ quan hệ pháp lý (Knowledge Graph Diagram)."""
        clean_id = str(item_id).strip()
        url = f"{self.base_url}/{clean_id}/diagram"
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            logger.warning(f"API Lược đồ trả về status {response.status_code} cho văn bản ID {clean_id}")
            return None
        except requests.exceptions.RequestException as exc:
            logger.error(f"Lỗi mạng khi tải lược đồ cho văn bản {clean_id}: {exc}")
            return None

    def search_documents(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Gửi truy vấn tìm kiếm nâng cao để thu thập danh mục văn bản."""
        url = f"{self.base_url}/search"
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            logger.warning(f"API Tìm kiếm trả về status {response.status_code}")
            return None
        except requests.exceptions.RequestException as exc:
            logger.error(f"Lỗi mạng trong quá trình gọi API Tìm kiếm: {exc}")
            return None

    def close(self):
        """Giải phóng toàn bộ kết nối trong HTTP Connection Pool."""
        self.session.close()