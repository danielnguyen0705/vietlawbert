import os
from config import config

class ModelRouter:
    @staticmethod
    def get_model_for_task(task_type: str) -> str:
        """
        Routing logic để chọn model dựa trên tác vụ.
        - 'text': dùng qwen2.5 (bản 1.5b đã tải)
        - 'ocr': dự phòng cho model Vision (sẽ tích hợp sau)
        """
        if task_type == "text":
            return config.CONTEXTUALIZER_MODEL
        elif task_type == "ocr":
            # Trả về model vision nếu có, nếu chưa thì fallback về text
            return os.getenv("OCR_MODEL", "qwen2-vl:2b")
        return config.CONTEXTUALIZER_MODEL
