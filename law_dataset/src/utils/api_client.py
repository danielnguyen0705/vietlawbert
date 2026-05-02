import requests
import json
import logging

class VBPLApiClient:
    def __init__(self):
        self.base_url = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        self.logger = logging.getLogger(__name__)

    def search_documents(self, payload):
        """Gửi Payload tìm kiếm thông minh để lấy danh sách ItemID"""
        url = f"{self.base_url}/search"
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Lỗi API Tìm kiếm: {e}")
            return None

    def get_diagram(self, item_id):
        """Lấy dữ liệu JSON Lược đồ (Quan hệ) của một văn bản"""
        url = f"{self.base_url}/{item_id}/diagram"
        try:
            response = requests.get(url, headers=self.headers, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Lỗi API Lược đồ cho ID {item_id}: {e}")
            return None

    def get_document_attributes(self, item_id):
        """Lấy dữ liệu JSON Thuộc tính (Metadata) của một văn bản"""
        # Lưu ý: Cần kiểm tra lại URL chính xác cho API thuộc tính
        url = f"{self.base_url}/{item_id}/attributes" 
        try:
            response = requests.get(url, headers=self.headers, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Lỗi API Thuộc tính cho ID {item_id}: {e}")
            return None

# ==========================================
# CÁCH SỬ DỤNG (Test độc lập):
# if __name__ == "__main__":
#     client = VBPLApiClient()
#     # Lấy thử lược đồ của ID 187505
#     data = client.get_diagram("187505")
#     print(json.dumps(data, indent=2, ensure_ascii=False))