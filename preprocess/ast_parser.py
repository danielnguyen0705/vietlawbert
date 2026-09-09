"""
ast_parser.py - Bộ phân tích cú pháp phân cấp pháp lý lai (Hybrid Legal AST Parser).
Bóc tách toàn diện: Phần -> Chương -> Mục -> Điều -> Khoản -> Điểm.
Tự động tiêm Metadata và cấu trúc phân cấp phục vụ Hierarchy-Aware MRL.
"""

from __future__ import annotations

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from lark import Lark

logger = logging.getLogger("VietLawBERT_ASTParser")

LEGAL_GRAMMAR = r"""
    start: document
    document: (part | chapter | section | article | text_line)+

    part: PART_KEYWORD ROMAN_NUM title?
    chapter: CHAPTER_KEYWORD ROMAN_NUM title?
    section: SECTION_KEYWORD INT title?
    article: ARTICLE_KEYWORD INT (DOT | COLON) title?

    title: /[^\n]+/
    text_line: /[^\n]+/

    PART_KEYWORD: "Phần" | "PHẦN"
    CHAPTER_KEYWORD: "Chương" | "CHƯƠNG"
    SECTION_KEYWORD: "Mục" | "MỤC"
    ARTICLE_KEYWORD: "Điều" | "ĐIỀU"

    ROMAN_NUM: /[IVXLCDM]+/
    INT: /\d+/
    DOT: "."
    COLON: ":"

    %import common.WS
    %ignore WS
"""


class LegalHierarchyTracker:
    """Theo dõi trạng thái phân cấp văn cảnh xuyên suốt tài liệu."""
    def __init__(self):
        self.current_part: str = ""
        self.current_chapter: str = ""
        self.current_section: str = ""
        self.current_article: str = ""
        self.current_article_title: str = ""

    def get_hierarchy_path(self, clause: Optional[str] = None, point: Optional[str] = None) -> str:
        nodes = []
        if self.current_part:
            nodes.append(self.current_part)
        if self.current_chapter:
            nodes.append(self.current_chapter)
        if self.current_section:
            nodes.append(self.current_section)
        if self.current_article:
            art_str = f"Điều {self.current_article}"
            if self.current_article_title:
                art_str += f" ({self.current_article_title})"
            nodes.append(art_str)
        if clause:
            nodes.append(f"Khoản {clause}")
        if point:
            nodes.append(f"Điểm {point}")
        return " > ".join(nodes) if nodes else "Toàn văn"

    def get_macro_label(self) -> str:
        """Trả về định danh vĩ mô cấp Chương/Luật phục vụ hàm mất mát phân cụm."""
        if self.current_chapter:
            return self.current_chapter
        if self.current_part:
            return self.current_part
        return "CHUNG"


class HybridASTParser:
    def __init__(self):
        try:
            self.lark = Lark(LEGAL_GRAMMAR, parser='lalr', maybe_placeholders=True)
            logger.info("Khởi tạo Lark EBNF Grammar thành công.")
        except Exception as exc:
            logger.warning("Lark Parser khởi tạo thất bại (%s), kích hoạt Full Regex State-machine.", exc)
            self.lark = None

        # Regex patterns phòng vệ
        self.re_part = re.compile(r'^(?:PHẦN|Phần)\s+([IVXLCDM]+)[\.:\s]*(.*)$', re.MULTILINE)
        self.re_chapter = re.compile(r'^(?:CHƯƠNG|Chương)\s+([IVXLCDM]+)[\.:\s]*(.*)$', re.MULTILINE)
        self.re_section = re.compile(r'^(?:MỤC|Mục)\s+(\d+)[\.:\s]*(.*)$', re.MULTILINE)
        self.re_article = re.compile(r'^(?:ĐIỀU|Điều)\s+(\d+)[\.:\s]*(.*)$', re.MULTILINE)
        self.re_clause = re.compile(r'^(\d+)[\.\)]\s+(.*)$')
        self.re_point = re.compile(r'^([a-zđ])[\.\)]\s+(.*)$')

    def parse_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Phân tích văn bản thành các khối tri thức nguyên tử mang nhãn phân cấp đầy đủ."""
        if not text or not text.strip():
            return []

        doc_id = str(metadata.get("doc_id") or metadata.get("item_id") or "UNKNOWN_DOC")
        doc_number = str(metadata.get("doc_number") or metadata.get("docNum") or "N/A")
        doc_title = str(metadata.get("title") or metadata.get("source_doc") or "Văn bản pháp luật")
        eff_date = str(metadata.get("effective_date") or metadata.get("effFrom") or "Chưa xác định")
        status = str(metadata.get("status") or "Còn hiệu lực")
        org = str(metadata.get("co_quan") or metadata.get("org") or "N/A")

        meta_header = (
            f"[META] Văn bản: {doc_number} | Tiêu đề: {doc_title} | "
            f"Cơ quan: {org} | Hiệu lực: {eff_date} | Tình trạng: {status}\n"
        )

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        tracker = LegalHierarchyTracker()
        chunks: List[Dict[str, Any]] = []

        current_article_lines: List[str] = []
        current_art_num: Optional[str] = None

        def flush_article_buffer():
            nonlocal current_article_lines, current_art_num
            if not current_art_num or not current_article_lines:
                current_article_lines = []
                return

            art_body = "\n".join(current_article_lines)
            sub_chunks = self._break_down_article(art_body, current_art_num, tracker, meta_header, metadata, doc_id)
            chunks.extend(sub_chunks)
            current_article_lines = []

        for line in lines:
            m_part = self.re_part.match(line)
            if m_part:
                flush_article_buffer()
                tracker.current_part = f"Phần {m_part.group(1)}"
                continue

            m_chap = self.re_chapter.match(line)
            if m_chap:
                flush_article_buffer()
                chap_title = m_chap.group(2).strip()
                tracker.current_chapter = f"Chương {m_chap.group(1)}" + (f" - {chap_title}" if chap_title else "")
                continue

            m_sec = self.re_section.match(line)
            if m_sec:
                flush_article_buffer()
                sec_title = m_sec.group(2).strip()
                tracker.current_section = f"Mục {m_sec.group(1)}" + (f" - {sec_title}" if sec_title else "")
                continue

            m_art = self.re_article.match(line)
            if m_art:
                flush_article_buffer()
                current_art_num = m_art.group(1)
                tracker.current_article = current_art_num
                tracker.current_article_title = m_art.group(2).strip()
                current_article_lines.append(line)
                continue

            if current_art_num:
                current_article_lines.append(line)
            else:
                # Dữ liệu thuộc phần mở đầu / căn cứ ban hành
                if len(line) > 40:
                    raw_text_chunk = f"{meta_header}[HIERARCHY] Căn cứ pháp lý\n[CONTENT] {line}"
                    chunks.append({
                        "chunk_id": f"{doc_id}_preamble_{len(chunks)}",
                        "doc_id": doc_id,
                        "macro_label": "CAN_CU",
                        "hierarchy_path": "Lời mở đầu / Căn cứ ban hành",
                        "text": raw_text_chunk,
                        "content": raw_text_chunk,
                        "metadata": metadata,
                    })

        flush_article_buffer()
        logger.info("Hybrid AST Parser trích xuất thành công %d chunks cho văn bản %s.", len(chunks), doc_id)
        return chunks

    def _break_down_article(
        self,
        art_body: str,
        art_num: str,
        tracker: LegalHierarchyTracker,
        meta_header: str,
        metadata: Dict[str, Any],
        doc_id: str,
    ) -> List[Dict[str, Any]]:
        """Phân rã một Điều thành các Khoản và Điểm để tránh vượt giới hạn ngữ cảnh."""
        sub_chunks: List[Dict[str, Any]] = []
        lines = art_body.splitlines()
        lead_content = lines[0] if lines else f"Điều {art_num}"

        clauses: List[Tuple[Optional[str], List[str]]] = []
        current_clause_num: Optional[str] = None
        current_clause_lines: List[str] = []

        for line in lines[1:]:
            m_cl = self.re_clause.match(line)
            if m_cl:
                if current_clause_lines or current_clause_num:
                    clauses.append((current_clause_num, current_clause_lines))
                current_clause_num = m_cl.group(1)
                current_clause_lines = [line]
            else:
                if current_clause_num:
                    current_clause_lines.append(line)
                else:
                    lead_content += "\n" + line

        if current_clause_lines or current_clause_num:
            clauses.append((current_clause_num, current_clause_lines))

        macro_label = tracker.get_macro_label()

        if not clauses:
            # Điều luật ngắn không phân Khoản
            h_path = tracker.get_hierarchy_path()
            raw_text_chunk = f"{meta_header}[HIERARCHY] {h_path}\n[CONTENT] {art_body.strip()}"
            sub_chunks.append({
                "chunk_id": f"{doc_id}_art_{art_num}",
                "doc_id": doc_id,
                "macro_label": macro_label,
                "hierarchy_path": h_path,
                "text": raw_text_chunk,
                "content": raw_text_chunk,
                "metadata": metadata,
            })
            return sub_chunks

        # Lưu bản tóm lược điều nếu phần mở đầu có nghĩa
        if len(lead_content.strip()) > 30:
            h_path = tracker.get_hierarchy_path()
            raw_text_chunk = f"{meta_header}[HIERARCHY] {h_path}\n[CONTENT] {lead_content.strip()}"
            sub_chunks.append({
                "chunk_id": f"{doc_id}_art_{art_num}_root",
                "doc_id": doc_id,
                "macro_label": macro_label,
                "hierarchy_path": h_path,
                "text": raw_text_chunk,
                "content": raw_text_chunk,
                "metadata": metadata,
            })

        for cl_num, cl_lines in clauses:
            cl_text = "\n".join(cl_lines).strip()
            h_path = tracker.get_hierarchy_path(clause=cl_num)
            raw_text_chunk = f"{meta_header}[HIERARCHY] {h_path}\n[CONTENT] {cl_text}"
            sub_chunks.append({
                "chunk_id": f"{doc_id}_art_{art_num}_cl_{cl_num}",
                "doc_id": doc_id,
                "macro_label": macro_label,
                "hierarchy_path": h_path,
                "text": raw_text_chunk,
                "content": raw_text_chunk,
                "metadata": metadata,
            })

        return sub_chunks