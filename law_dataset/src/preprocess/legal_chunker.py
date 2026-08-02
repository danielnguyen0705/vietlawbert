"""
legal_chunker.py - Structure-Aware Legal Chunking cho Văn bản Pháp luật Việt Nam

Nguyên tắc:
1. Atomic Unit: Khoản (Clause) là đơn vị cấu trúc nhỏ nhất
2. Không bao giờ cắt giữa chừng một Khoản
3. Tất cả Điểm (a, b, c, d) phải nằm trọn vẹn trong chunk
4. Metadata phải có hierarchy_path đầy đủ (phần > chương > mục > tiểu_mục > điều > khoản > điểm)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class LegalLevel(Enum):
    PHAN = "phần"
    CHUONG = "chương"
    MUC = "mục"
    TIEU_MUC = "tiểu_mục"
    DIEU = "điều"
    KHOAN = "khoản"
    DIEM = "điểm"


@dataclass
class HierarchyPath:
    """Đường dẫn phân cấp trong văn bản pháp luật"""
    phan: Optional[str] = None
    chuong: Optional[str] = None
    muc: Optional[str] = None
    tieu_muc: Optional[str] = None
    dieu: Optional[str] = None
    khoan: Optional[str] = None
    diem: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "phần": self.phan,
            "chương": self.chuong,
            "mục": self.muc,
            "tiểu_mục": self.tieu_muc,
            "điều": self.dieu,
            "khoản": self.khoan,
            "điểm": self.diem,
        }

    def copy(self) -> 'HierarchyPath':
        return HierarchyPath(
            phan=self.phan,
            chuong=self.chuong,
            muc=self.muc,
            tieu_muc=self.tieu_muc,
            dieu=self.dieu,
            khoan=self.khoan,
            diem=self.diem,
        )


@dataclass
class LegalChunk:
    """Một chunk văn bản pháp luật"""
    chunk_id: str
    text: str
    hierarchy: HierarchyPath
    level: LegalLevel
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# REGEX PATTERNS CHO CẤU TRÚC PHÁP LUẬT VN
# ============================================================

# Phần (Roman numerals: I, II, III, IV, V, ...)
PHAN_PATTERN = re.compile(
    r'^(?:#####)?\s*(?:Phần\s+)([IVXLCDM]+)\s*[:\.\s]*(.*)$',
    re.MULTILINE | re.IGNORECASE
)

# Chương (Roman numerals)
CHUONG_PATTERN = re.compile(
    r'^(?:##)?\s*(?:CHƯƠNG\s+)([IVXLCDM]+)\s*[:\.\s]*(.*)$',
    re.MULTILINE | re.IGNORECASE
)

# Mục
MUC_PATTERN = re.compile(
    r'^(?:###)?\s*(?:MỤC\s+)(\d+)\s*[:\.\s]*(.*)$',
    re.MULTILINE | re.IGNORECASE
)

# Tiểu mục
TIEU_MUC_PATTERN = re.compile(
    r'^(?:####)?\s*(?:TIỂU\s*MỤC\s+)(\d+)\s*[:\.\s]*(.*)$',
    re.MULTILINE | re.IGNORECASE
)

# Điều (Arabic numerals)
DIEU_PATTERN = re.compile(
    r'^(?:#####)?\s*(?:Điều\s+)(\d+[a-zA-Z]?)\s*[:\.\s]*(.*)$',
    re.MULTILINE | re.IGNORECASE
)

# Khoản (Arabic numerals: 1., 2., 3., ...)
# Pattern 1: "1. Nội dung..."
# Pattern 2: "1\. Nội dung..." (escaped dot từ HTML conversion)
KHOAN_PATTERN = re.compile(
    r'^(?:(\d+)[\.\)\\\.]+)\s+(.+)$',
    re.MULTILINE
)

# Điểm (lowercase letters: a., b., c., d., ...)
DIEM_PATTERN = re.compile(
    r'^([a-zđ])[\.\)\\\.]+\s+(.+)$',
    re.MULTILINE
)

# Tiêu đề Điều (ví dụ: "Điều 3. Nguyên tắc cử tuyển")
DIEU_TITLE_PATTERN = re.compile(
    r'(?:#####)?\s*Điều\s+(\d+[a-zA-Z]?)\s*[:\.\s]*(.*)$',
    re.MULTILINE | re.IGNORECASE
)


def find_all_matches(pattern: re.Pattern, text: str) -> List[re.Match]:
    """Tìm tất cả matches của pattern trong text"""
    return list(pattern.finditer(text))


def get_structure_positions(text: str) -> List[Dict[str, Any]]:
    """
    Phân tích và trả về danh sách các vị trí cấu trúc trong văn bản.
    Sắp xếp theo thứ tự xuất hiện trong text.
    """
    positions = []

    # Tìm tất cả các vị trí cấu trúc
    for match in find_all_matches(PHAN_PATTERN, text):
        positions.append({
            'level': LegalLevel.PHAN,
            'number': match.group(1),
            'title': match.group(2).strip(),
            'start': match.start(),
            'end': match.end(),
        })

    for match in find_all_matches(CHUONG_PATTERN, text):
        positions.append({
            'level': LegalLevel.CHUONG,
            'number': match.group(1),
            'title': match.group(2).strip(),
            'start': match.start(),
            'end': match.end(),
        })

    for match in find_all_matches(MUC_PATTERN, text):
        positions.append({
            'level': LegalLevel.MUC,
            'number': match.group(1),
            'title': match.group(2).strip(),
            'start': match.start(),
            'end': match.end(),
        })

    for match in find_all_matches(TIEU_MUC_PATTERN, text):
        positions.append({
            'level': LegalLevel.TIEU_MUC,
            'number': match.group(1),
            'title': match.group(2).strip(),
            'start': match.start(),
            'end': match.end(),
        })

    for match in find_all_matches(DIEU_PATTERN, text):
        positions.append({
            'level': LegalLevel.DIEU,
            'number': match.group(1),
            'title': match.group(2).strip(),
            'start': match.start(),
            'end': match.end(),
        })

    # Sắp xếp theo vị trí xuất hiện
    positions.sort(key=lambda x: x['start'])

    return positions


def build_hierarchy_at_position(
    positions: List[Dict[str, Any]],
    target_pos: int
) -> HierarchyPath:
    """
    Xây dựng hierarchy path tại một vị trí nhất định.
    """
    hierarchy = HierarchyPath()

    for pos in positions:
        if pos['start'] > target_pos:
            break

        level = pos['level']
        number = pos['number']
        title = pos['title']

        if level == LegalLevel.PHAN:
            hierarchy.phan = f"Phần {number}"
        elif level == LegalLevel.CHUONG:
            hierarchy.chuong = f"Chương {number}"
        elif level == LegalLevel.MUC:
            hierarchy.muc = f"Mục {number}"
        elif level == LegalLevel.TIEU_MUC:
            hierarchy.tieu_muc = f"Tiểu mục {number}"
        elif level == LegalLevel.DIEU:
            hierarchy.dieu = f"Điều {number}"

    return hierarchy


def chunk_by_dieu(text: str, doc_id: str) -> List[LegalChunk]:
    """
    Chunk văn bản theo Điều (Article).
    Mỗi Điều sẽ là một chunk riêng biệt.
    """
    chunks = []
    positions = get_structure_positions(text)

    # Tìm tất cả Điều positions
    dieu_positions = [p for p in positions if p['level'] == LegalLevel.DIEU]

    if not dieu_positions:
        # Nếu không tìm thấy Điều nào, trả về toàn bộ text
        chunks.append(LegalChunk(
            chunk_id=f"{doc_id}_full",
            text=text.strip(),
            hierarchy=HierarchyPath(),
            level=LegalLevel.DIEU,
            start_pos=0,
            end_pos=len(text),
        ))
        return chunks

    # Chunk cho phần trước Điều đầu tiên (nếu có)
    if dieu_positions[0]['start'] > 0:
        preamble = text[:dieu_positions[0]['start']].strip()
        if preamble:
            chunks.append(LegalChunk(
                chunk_id=f"{doc_id}_preamble",
                text=preamble,
                hierarchy=HierarchyPath(),
                level=LegalLevel.DIEU,
                start_pos=0,
                end_pos=dieu_positions[0]['start'],
                metadata={'type': 'preamble'}
            ))

    # Chunk cho từng Điều
    for i, dieu_pos in enumerate(dieu_positions):
        start = dieu_pos['start']

        # Kết thúc: Điều tiếp theo hoặc cuối văn bản
        if i + 1 < len(dieu_positions):
            end = dieu_positions[i + 1]['start']
        else:
            end = len(text)

        dieu_text = text[start:end].strip()
        dieu_number = dieu_pos['number']

        # Xây hierarchy
        hierarchy = build_hierarchy_at_position(positions, start)
        hierarchy.dieu = f"Điều {dieu_number}"

        chunks.append(LegalChunk(
            chunk_id=f"{doc_id}_D{dieu_number}",
            text=dieu_text,
            hierarchy=hierarchy,
            level=LegalLevel.DIEU,
            start_pos=start,
            end_pos=end,
        ))

    return chunks


def chunk_by_khoan(text: str, doc_id: str) -> List[LegalChunk]:
    """
    Chunk văn bản theo Khoản (Clause) trong mỗi Điều.
    Đây là atomic unit - không bao giờ cắt giữa chừng một Khoản.
    """
    all_chunks = []

    # Trước tiên, chunk theo Điều
    dieu_chunks = chunk_by_dieu(text, doc_id)

    for dieu_chunk in dieu_chunks:
        dieu_text = dieu_chunk.text
        dieu_number = dieu_chunk.hierarchy.dieu

        # Tìm tất cả Khoản trong Điều này
        khoan_matches = find_all_matches(KHOAN_PATTERN, dieu_text)

        if not khoan_matches:
            # Nếu không tìm thấy Khoản nào, giữ nguyên cả Điều
            all_chunks.append(dieu_chunk)
            continue

        # Chunk cho phần trước Khoản đầu tiên (tiêu đề Điều)
        if khoan_matches[0].start() > 0:
            dieu_header = dieu_text[:khoan_matches[0].start()].strip()
            if dieu_header:
                header_hierarchy = dieu_chunk.hierarchy.copy()
                header_hierarchy.khoan = None
                
                # Sửa lỗi: xử lý nếu dieu_number là None
                safe_dieu_id = dieu_number.replace(' ', '') if dieu_number else "header"

                all_chunks.append(LegalChunk(
                    chunk_id=f"{doc_id}_{safe_dieu_id}_header",
                    text=dieu_header,
                    hierarchy=header_hierarchy,
                    level=LegalLevel.DIEU,
                    start_pos=dieu_chunk.start_pos,
                    end_pos=dieu_chunk.start_pos + khoan_matches[0].start(),
                    metadata={'type': 'dieu_header'}
                ))

        # Chunk cho từng Khoản
        for j, khoan_match in enumerate(khoan_matches):
            khoan_number = khoan_match.group(1)
            khoan_start = khoan_match.start()

            # Kết thúc: Khoản tiếp theo hoặc cuối Điều
            if j + 1 < len(khoan_matches):
                khoan_end = khoan_matches[j + 1].start()
            else:
                khoan_end = len(dieu_text)

            khoan_text = dieu_text[khoan_start:khoan_end].strip()

            # Xây hierarchy
            khoan_hierarchy = dieu_chunk.hierarchy.copy()
            khoan_hierarchy.khoan = f"Khoản {khoan_number}"

            # Kiểm tra xem Khoản có chứa Điểm không
            diem_matches = find_all_matches(DIEM_PATTERN, khoan_text)

            if diem_matches and len(diem_matches) > 1:
                # Nếu Khoản có nhiều Điểm, tách từng Điểm
                # Nhưng vẫn giữ Khoản làm chunk chính nếu không quá dài
                safe_dieu_id = dieu_number.replace(' ', '') if dieu_number else "doc"
                if len(khoan_text) <= 2000:
                    # Khoản ngắn, giữ nguyên
                    all_chunks.append(LegalChunk(
                        chunk_id=f"{doc_id}_{safe_dieu_id}_K{khoan_number}",
                        text=khoan_text,
                        hierarchy=khoan_hierarchy,
                        level=LegalLevel.KHOAN,
                        start_pos=dieu_chunk.start_pos + khoan_start,
                        end_pos=dieu_chunk.start_pos + khoan_end,
                    ))
                else:
                    # Khoản dài, tách theo Điểm
                    # Phần trước Điểm đầu tiên
                    if diem_matches[0].start() > 0:
                        khoan_intro = khoan_text[:diem_matches[0].start()].strip()
                        if khoan_intro:
                            all_chunks.append(LegalChunk(
                                chunk_id=f"{doc_id}_{safe_dieu_id}_K{khoan_number}_intro",
                                text=khoan_intro,
                                hierarchy=khoan_hierarchy,
                                level=LegalLevel.KHOAN,
                                start_pos=dieu_chunk.start_pos + khoan_start,
                                end_pos=dieu_chunk.start_pos + khoan_start + diem_matches[0].start(),
                                metadata={'type': 'khoan_intro'}
                            ))

                    # Tách từng Điểm
                    for k, diem_match in enumerate(diem_matches):
                        diem_letter = diem_match.group(1)
                        diem_start = diem_match.start()

                        if k + 1 < len(diem_matches):
                            diem_end = diem_matches[k + 1].start()
                        else:
                            diem_end = len(khoan_text)

                        diem_text = khoan_text[diem_start:diem_end].strip()

                        diem_hierarchy = khoan_hierarchy.copy()
                        diem_hierarchy.diem = f"điểm {diem_letter}"

                        safe_dieu_id = dieu_number.replace(' ', '') if dieu_number else "doc"
                        all_chunks.append(LegalChunk(
                            chunk_id=f"{doc_id}_{safe_dieu_id}_K{khoan_number}_{diem_letter}",
                            text=diem_text,
                            hierarchy=diem_hierarchy,
                            level=LegalLevel.DIEM,
                            start_pos=dieu_chunk.start_pos + khoan_start + diem_start,
                            end_pos=dieu_chunk.start_pos + khoan_start + diem_end,
                        ))
            else:
                # Khoản không có Điểm hoặc chỉ có 1 Điểm, giữ nguyên
                safe_dieu_id = dieu_number.replace(' ', '') if dieu_number else "doc"
                all_chunks.append(LegalChunk(
                    chunk_id=f"{doc_id}_{safe_dieu_id}_K{khoan_number}",
                    text=khoan_text,
                    hierarchy=khoan_hierarchy,
                    level=LegalLevel.KHOAN,
                    start_pos=dieu_chunk.start_pos + khoan_start,
                    end_pos=dieu_chunk.start_pos + khoan_end,
                ))

    return all_chunks


def chunk_legal_document(
    text: str,
    doc_id: str,
    max_chunk_size: int = 2000
) -> List[LegalChunk]:
    """
    Main function: Chunk văn bản pháp luật theo cấu trúc.

    Args:
        text: Văn bản Markdown đã được làm sạch
        doc_id: ID của văn bản
        max_chunk_size: Kích thước tối đa của một chunk (bytes)

    Returns:
        Danh sách LegalChunk
    """
    # Chunk theo Khoản (atomic unit)
    chunks = chunk_by_khoan(text, doc_id)

    # Merge các chunk quá ngắn
    merged_chunks = merge_short_chunks(chunks, min_size=200)

    # Tạo chunk_id duy nhất
    for i, chunk in enumerate(merged_chunks):
        chunk.chunk_id = f"{doc_id}_chunk_{i+1}"

    return merged_chunks


def merge_short_chunks(
    chunks: List[LegalChunk],
    min_size: int = 200
) -> List[LegalChunk]:
    """
    Merge các chunk quá ngắn với chunk liền kề.
    """
    if not chunks:
        return []

    merged = []
    buffer = chunks[0]

    for i in range(1, len(chunks)):
        current = chunks[i]

        # Nếu buffer quá ngắn, merge với chunk tiếp theo
        if len(buffer.text) < min_size:
            # Chỉ merge nếu cùng cấp độ hoặc buffer là header/intro
            if (buffer.metadata.get('type') in ['dieu_header', 'khoan_intro', None] or
                buffer.level.value <= current.level.value):
                buffer = LegalChunk(
                    chunk_id=buffer.chunk_id,
                    text=buffer.text + "\n\n" + current.text,
                    hierarchy=buffer.hierarchy,
                    level=max(buffer.level, current.level, key=lambda x: list(LegalLevel).index(x)),
                    start_pos=buffer.start_pos,
                    end_pos=current.end_pos,
                    metadata=buffer.metadata,
                )
                continue

        merged.append(buffer)
        buffer = current

    merged.append(buffer)
    return merged


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def extract_cross_references(text: str) -> List[Dict[str, str]]:
    """Trích xuất các dẫn chiếu phức tạp trong văn bản."""
    refs = []

    # 1. Dẫn chiếu Điều/Khoản trong văn bản khác
    # VD: "Điều 25 của Luật Chuyển giao..." hoặc "Khoản 2 Điều 15 Nghị định 100..."
    complex_ref_pattern = re.compile(
        r'(?:tại\s+)?(?:[Kk]hoản\s+\d+\s+)?(?:[Đđ]iều\s+(\d+[a-zA-Z]?))\s+(?:của|trong|được\s+quy\s+định\s+tại)?\s*(Luật|Nghị\s+định|Thông\s+tư|Quyết\s+định)(?:\s+số)?\s*([\w\-/]+)?\s*([^.,;(\n]+)?',
        re.IGNORECASE
    )
    for match in complex_ref_pattern.finditer(text):
        target_doc = f"{match.group(2)} {match.group(3) or ''} {match.group(4) or ''}".strip()
        refs.append({
            'target_type': 'Điều',
            'target_number': match.group(1),
            'target_document': target_doc
        })

    # 2. Dẫn chiếu nội bộ ("Điều này", "Khoản này")
    internal_ref = re.compile(r'(?:tại|theo)\s+([Kk]hoản\s+\d+\s+)?([Đđ]iều\s+này)', re.IGNORECASE)
    for match in internal_ref.finditer(text):
        refs.append({
            'target_type': 'Nội bộ',
            'target_number': match.group(0).replace("tại ", "").replace("theo ", "").strip(),
            'target_document': 'Văn bản hiện tại'
        })

    # 3. Dẫn chiếu tổng quát đến Luật/Nghị định
    luat_ref_pattern = re.compile(
        r'(?:theo\s+quy\s+định\s+của\s+)?(Luật|Nghị\s+định|Quyết\s+định|Thông\s+tư)\s+(?:số\s+)?([\d\-/A-Z]+)\s+(?:ngày\s+\d+\s+tháng\s+\d+\s+năm\s+\d+)?\s*(?:của\s+([A-Z][^.,;(\n]+))?',
        re.IGNORECASE
    )
    for match in luat_ref_pattern.finditer(text):
        # Tránh trùng lặp với pattern 1
        doc_type = match.group(1)
        doc_num = match.group(2)
        target_doc = f"{doc_type} {doc_num} {match.group(3) or ''}".strip()
        if not any(r['target_document'] == target_doc for r in refs):
            refs.append({
                'target_type': 'Văn bản',
                'target_number': doc_num,
                'target_document': target_doc
            })

    return refs


# ============================================================
# TEST FUNCTION
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python legal_chunker.py <file.md>")
        sys.exit(1)

    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        text = f.read()

    doc_id = sys.argv[1].split('/')[-1].replace('.md', '')

    chunks = chunk_legal_document(text, doc_id)

    print(f"Tổng số chunks: {len(chunks)}")
    print("=" * 60)

    for chunk in chunks:
        print(f"\nChunk ID: {chunk.chunk_id}")
        print(f"Level: {chunk.level.value}")
        print(f"Hierarchy: {chunk.hierarchy.to_dict()}")
        print(f"Text length: {len(chunk.text)} chars")
        print(f"Text preview: {chunk.text[:200]}...")
        print("-" * 60)
