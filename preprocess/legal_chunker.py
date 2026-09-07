"""
legal_chunker.py - Structure-Aware Legal Chunking cho Văn bản Pháp luật Việt Nam.
Tối ưu hóa biểu diễn ngữ nghĩa phân cấp phục vụ huấn luyện mô hình VietLawBERT và GraphRAG.
"""

from __future__ import annotations

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

    def copy(self) -> HierarchyPath:
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
    """Một chunk văn bản pháp luật chứa toàn vẹn đơn vị ngữ nghĩa"""
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

MD_HEADING_PREFIX = r'^[ \t]*(?:#{1,6}[ \t]*)?(?:[*_]{1,3})?[“”"\'‘’(\[]*[ \t]*'

PHAN_PATTERN = re.compile(
    MD_HEADING_PREFIX + r'(?:Phần\s+)([IVXLCDM]+)\s*[:\.\s]*(.*)$',
    re.MULTILINE | re.IGNORECASE
)

CHUONG_PATTERN = re.compile(
    MD_HEADING_PREFIX + r'(?:CHƯƠNG\s+)([IVXLCDM]+)\s*[:\.\s]*(.*)$',
    re.MULTILINE | re.IGNORECASE
)

MUC_PATTERN = re.compile(
    MD_HEADING_PREFIX + r'(?:MỤC\s+)(\d+)\s*[:\.\s]*(.*)$',
    re.MULTILINE | re.IGNORECASE
)

TIEU_MUC_PATTERN = re.compile(
    MD_HEADING_PREFIX + r'(?:TIỂU\s*MỤC\s+)(\d+)\s*[:\.\s]*(.*)$',
    re.MULTILINE | re.IGNORECASE
)

DIEU_PATTERN = re.compile(
    MD_HEADING_PREFIX + r'(?:Điều\s+)(\d+[a-zA-Z]?)\s*[:\.\s]*(.*)$',
    re.MULTILINE | re.IGNORECASE
)

KHOAN_PATTERN = re.compile(
    r'^(?:(\d+)[\.\)\\\.]+)\s+(.+)$',
    re.MULTILINE
)

DIEM_PATTERN = re.compile(
    r'^([a-zđ])[\.\)\\\.]+\s+(.+)$',
    re.MULTILINE
)


def find_all_matches(pattern: re.Pattern, text: str) -> List[re.Match]:
    return list(pattern.finditer(text))


def get_structure_positions(text: str) -> List[Dict[str, Any]]:
    positions = []

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

    positions.sort(key=lambda x: x['start'])
    return positions


def build_hierarchy_at_position(
    positions: List[Dict[str, Any]],
    target_pos: int
) -> HierarchyPath:
    """
    Xây dựng hierarchy path chính xác. Tự động reset các cấp phân vị con
    khi chuyển sang Phần hoặc Chương mới để loại trừ rò rỉ ngữ cảnh.
    """
    hierarchy = HierarchyPath()

    for pos in positions:
        if pos['start'] > target_pos:
            break

        level = pos['level']
        number = pos['number']

        if level == LegalLevel.PHAN:
            hierarchy.phan = f"Phần {number}"
            hierarchy.chuong = None
            hierarchy.muc = None
            hierarchy.tieu_muc = None
            hierarchy.dieu = None
        elif level == LegalLevel.CHUONG:
            hierarchy.chuong = f"Chương {number}"
            hierarchy.muc = None
            hierarchy.tieu_muc = None
            hierarchy.dieu = None
        elif level == LegalLevel.MUC:
            hierarchy.muc = f"Mục {number}"
            hierarchy.tieu_muc = None
            hierarchy.dieu = None
        elif level == LegalLevel.TIEU_MUC:
            hierarchy.tieu_muc = f"Tiểu mục {number}"
            hierarchy.dieu = None
        elif level == LegalLevel.DIEU:
            hierarchy.dieu = f"Điều {number}"

    return hierarchy


def chunk_by_dieu(text: str, doc_id: str) -> List[LegalChunk]:
    chunks = []
    positions = get_structure_positions(text)
    dieu_positions = [p for p in positions if p['level'] == LegalLevel.DIEU]

    if not dieu_positions:
        chunks.append(LegalChunk(
            chunk_id=f"{doc_id}_full",
            text=text.strip(),
            hierarchy=HierarchyPath(phan="Phần Mở đầu"),
            level=LegalLevel.DIEU,
            start_pos=0,
            end_pos=len(text),
        ))
        return chunks

    if dieu_positions[0]['start'] > 0:
        preamble = text[:dieu_positions[0]['start']].strip()
        if preamble:
            chunks.append(LegalChunk(
                chunk_id=f"{doc_id}_preamble",
                text=preamble,
                hierarchy=HierarchyPath(phan="Phần Mở đầu"),
                level=LegalLevel.DIEU,
                start_pos=0,
                end_pos=dieu_positions[0]['start'],
                metadata={'type': 'preamble'}
            ))

    for i, dieu_pos in enumerate(dieu_positions):
        start = dieu_pos['start']
        end = dieu_positions[i + 1]['start'] if i + 1 < len(dieu_positions) else len(text)

        dieu_text = text[start:end].strip()
        dieu_number = dieu_pos['number']

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
    all_chunks = []
    dieu_chunks = chunk_by_dieu(text, doc_id)

    for dieu_chunk in dieu_chunks:
        dieu_text = dieu_chunk.text
        dieu_name = dieu_chunk.hierarchy.dieu or "Doc"
        safe_dieu_id = dieu_name.replace(' ', '')

        khoan_matches = find_all_matches(KHOAN_PATTERN, dieu_text)

        if not khoan_matches:
            all_chunks.append(dieu_chunk)
            continue

        if khoan_matches[0].start() > 0:
            dieu_header = dieu_text[:khoan_matches[0].start()].strip()
            if dieu_header:
                header_hierarchy = dieu_chunk.hierarchy.copy()
                header_hierarchy.khoan = None

                all_chunks.append(LegalChunk(
                    chunk_id=f"{doc_id}_{safe_dieu_id}_header",
                    text=dieu_header,
                    hierarchy=header_hierarchy,
                    level=LegalLevel.DIEU,
                    start_pos=dieu_chunk.start_pos,
                    end_pos=dieu_chunk.start_pos + khoan_matches[0].start(),
                    metadata={'type': 'dieu_header'}
                ))

        for j, khoan_match in enumerate(khoan_matches):
            khoan_number = khoan_match.group(1)
            khoan_start = khoan_match.start()
            khoan_end = khoan_matches[j + 1].start() if j + 1 < len(khoan_matches) else len(dieu_text)
            khoan_text = dieu_text[khoan_start:khoan_end].strip()

            khoan_hierarchy = dieu_chunk.hierarchy.copy()
            khoan_hierarchy.khoan = f"Khoản {khoan_number}"

            diem_matches = find_all_matches(DIEM_PATTERN, khoan_text)

            if diem_matches and len(diem_matches) > 1:
                if len(khoan_text) <= 2000:
                    all_chunks.append(LegalChunk(
                        chunk_id=f"{doc_id}_{safe_dieu_id}_K{khoan_number}",
                        text=khoan_text,
                        hierarchy=khoan_hierarchy,
                        level=LegalLevel.KHOAN,
                        start_pos=dieu_chunk.start_pos + khoan_start,
                        end_pos=dieu_chunk.start_pos + khoan_end,
                    ))
                else:
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

                    for k, diem_match in enumerate(diem_matches):
                        diem_letter = diem_match.group(1)
                        diem_start = diem_match.start()
                        diem_end = diem_matches[k + 1].start() if k + 1 < len(diem_matches) else len(khoan_text)
                        diem_text = khoan_text[diem_start:diem_end].strip()

                        diem_hierarchy = khoan_hierarchy.copy()
                        diem_hierarchy.diem = f"Điểm {diem_letter}"

                        all_chunks.append(LegalChunk(
                            chunk_id=f"{doc_id}_{safe_dieu_id}_K{khoan_number}_{diem_letter}",
                            text=diem_text,
                            hierarchy=diem_hierarchy,
                            level=LegalLevel.DIEM,
                            start_pos=dieu_chunk.start_pos + khoan_start + diem_start,
                            end_pos=dieu_chunk.start_pos + khoan_start + diem_end,
                        ))
            else:
                all_chunks.append(LegalChunk(
                    chunk_id=f"{doc_id}_{safe_dieu_id}_K{khoan_number}",
                    text=khoan_text,
                    hierarchy=khoan_hierarchy,
                    level=LegalLevel.KHOAN,
                    start_pos=dieu_chunk.start_pos + khoan_start,
                    end_pos=dieu_chunk.start_pos + khoan_end,
                ))

    return all_chunks


def split_oversized_chunk(chunk: LegalChunk, max_size: int) -> List[LegalChunk]:
    """Tách atomic unit quá dài ở biên câu/từ nhưng bảo toàn semantic chunk_id."""
    if max_size <= 0:
        raise ValueError("max_chunk_size phải lớn hơn 0")
    if len(chunk.text) <= max_size:
        return [chunk]

    text = chunk.text
    spans = []
    start = 0
    preferred_boundaries = ("\n\n", "\n", ". ", "; ", ": ", " ")
    while len(text) - start > max_size:
        window = text[start:start + max_size + 1]
        minimum_cut = max(1, max_size // 2)
        cut = -1
        for boundary in preferred_boundaries:
            candidate = window.rfind(boundary)
            if candidate >= minimum_cut:
                cut = candidate + len(boundary)
                break
        if cut <= 0:
            cut = max_size
        end = start + cut
        spans.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text):
        spans.append((start, len(text)))

    total = len(spans)
    output = []
    for index, (local_start, local_end) in enumerate(spans, 1):
        segment_text = text[local_start:local_end].strip()
        if not segment_text:
            continue
        metadata = dict(chunk.metadata)
        metadata.update({"oversized_segment": index, "oversized_segment_count": total})
        sub_chunk_id = f"{chunk.chunk_id}_p{index}" if total > 1 else chunk.chunk_id

        output.append(LegalChunk(
            chunk_id=sub_chunk_id,
            text=segment_text,
            hierarchy=chunk.hierarchy.copy(),
            level=chunk.level,
            start_pos=chunk.start_pos + local_start,
            end_pos=chunk.start_pos + local_end,
            metadata=metadata,
        ))
    return output


def merge_short_chunks(chunks: List[LegalChunk], min_size: int = 200) -> List[LegalChunk]:
    """
    Hợp nhất tiêu đề Điều (dieu_header) hoặc lời dẫn Khoản (khoan_intro) ngắn
    vào Khoản hoặc Điểm kế tiếp, bảo toàn định danh của nội dung chính.
    """
    if not chunks:
        return []

    merged = []
    buffer = chunks[0]

    for i in range(1, len(chunks)):
        current = chunks[i]

        if len(buffer.text) < min_size:
            is_header_or_intro = buffer.metadata.get('type') in {'dieu_header', 'khoan_intro'}
            same_article = buffer.hierarchy.dieu == current.hierarchy.dieu
            if is_header_or_intro and same_article:
                combined_metadata = dict(current.metadata)
                combined_metadata['prepended_header'] = buffer.text
                buffer = LegalChunk(
                    chunk_id=current.chunk_id,
                    text=buffer.text + "\n\n" + current.text,
                    hierarchy=current.hierarchy.copy(),
                    level=current.level,
                    start_pos=buffer.start_pos,
                    end_pos=current.end_pos,
                    metadata=combined_metadata,
                )
                continue

        merged.append(buffer)
        buffer = current

    merged.append(buffer)
    return merged


def chunk_legal_document(text: str, doc_id: str, max_chunk_size: int = 1600) -> List[LegalChunk]:
    """
    Pipeline phân mảnh cấu trúc hoàn chỉnh:
    Bảo toàn toàn bộ Anchor phân cấp pháp luật phục vụ Retrieval-Augmented Generation.
    """
    chunks = chunk_by_khoan(text, doc_id)
    merged_chunks = merge_short_chunks(chunks, min_size=200)

    bounded_chunks = []
    for chunk in merged_chunks:
        bounded_chunks.extend(split_oversized_chunk(chunk, max_chunk_size))

    # Gắn thứ tự tuyệt đối vào metadata, bảo tồn chunk_id ngữ nghĩa gốc
    for i, chunk in enumerate(bounded_chunks):
        chunk.metadata["chunk_index"] = i + 1

    return bounded_chunks


def extract_cross_references(text: str) -> List[Dict[str, str]]:
    """Trích xuất các dẫn chiếu chéo pháp luật chuẩn xác."""
    refs = []

    # 1. Dẫn chiếu Điều/Khoản trong văn bản khác
    complex_ref_pattern = re.compile(
        r'(?:tại\s+)?(?:[Kk]hoản\s+\d+\s+)?(?:[Đđ]iều\s+(\d+[a-zA-Z]?))\s+(?:của|trong|được\s+quy\s+định\s+tại)?\s*(Luật|Nghị\s+định|Thông\s+tư|Quyết\s+định)(?:\s+số)?\s*([0-9]+/[0-9]+/[A-Z0-9\-]+|[A-Z0-9\-]+)?',
        re.IGNORECASE
    )
    for match in complex_ref_pattern.finditer(text):
        target_doc = f"{match.group(2)} {match.group(3) or ''}".strip()
        refs.append({
            'target_type': 'Điều',
            'target_number': match.group(1),
            'target_document': target_doc
        })

    # 2. Dẫn chiếu nội bộ
    internal_ref = re.compile(r'(?:tại|theo)\s+([Kk]hoản\s+\d+\s+)?([Đđ]iều\s+này)', re.IGNORECASE)
    for match in internal_ref.finditer(text):
        refs.append({
            'target_type': 'Nội bộ',
            'target_number': match.group(0).replace("tại ", "").replace("theo ", "").strip(),
            'target_document': 'Văn bản hiện tại'
        })

    return refs
