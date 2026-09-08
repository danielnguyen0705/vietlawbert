"""
pipelines.py - Pipeline bóc tách bản đồ tri thức quan hệ pháp luật và lưu trữ Shard Staging.
Chuẩn hóa Schema đồ thị (22 quan hệ HIN), trích xuất văn bản thô cho AST Parser và nén Shard cục bộ.
"""

from __future__ import annotations

import os
import json
import gzip
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from bs4 import BeautifulSoup

from configs.paths import RAW_SHARDS_DIR
from .items import VietLawItem


class LegalOntologyMappingPipeline:
    def __init__(self):
        self.ontology_path = Path(__file__).resolve().parent / "system_ontology_map.json"
        self.static_mapping, self.edge_templates, self.category_aliases = self._load_system_ontology()
        self.logger = logging.getLogger("VietLawBERT_Pipeline")

        # Cấu hình lưu trữ Shard Staging tự động khi không dùng cờ -O
        self.raw_shards_dir = Path(RAW_SHARDS_DIR)
        self.raw_shards_dir.mkdir(parents=True, exist_ok=True)
        self.shard_buffer: List[Dict[str, Any]] = []
        self.shard_counter = 1
        self.shard_size = int(os.getenv("STAGING_SHARD_SIZE", "1000"))
        self.dynamic_maps = {}

    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        pipeline.crawler = crawler
        # Kiểm tra xem Scrapy có đang xuất tệp qua Feed Exporter (-O) hay không
        feeds = getattr(crawler.settings, "get", lambda k, d=None: d)("FEEDS", {})
        pipeline.feed_export_active = bool(feeds)
        return pipeline

    def open_spider(self, spider):
        self.logger.info(
            "[PIPELINE KHỞI ĐỘNG] LegalOntologyMappingPipeline sẵn sàng. Tự động gom Shard đĩa: %s",
            "TẮT (Scrapy Feed -O đang kích hoạt)" if self.feed_export_active else "BẬT (Ghi trực tiếp raw_shards/)",
        )

    def close_spider(self, spider):
        # Xả nốt phần dữ liệu còn lại trong bộ đệm xuống đĩa
        if not self.feed_export_active and self.shard_buffer:
            self._flush_buffer_to_shard()

        if self.dynamic_maps:
            try:
                self.save_dynamic_mappings()
            except Exception as exc:
                self.logger.error("Lỗi khi lưu dynamic ontology cache: %s", exc)

        self.logger.info("[PIPELINE ĐÓNG] Hoàn tất xử lý và bảo toàn toàn bộ Shard dữ liệu.")

    def _flush_buffer_to_shard(self):
        """Nén Gzip và ghi Shard hoàn chỉnh xuống đĩa phục vụ Phase 2 Ingestion."""
        if not self.shard_buffer:
            return

        shard_path = self.raw_shards_dir / f"crawl_pages_{self.shard_counter:05d}.jsonl.gz"
        try:
            with gzip.open(shard_path, "wt", encoding="utf-8") as f:
                for record in self.shard_buffer:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.logger.info(
                "[PIPELINE SHARD] Đã ghi thành công %d bản ghi (đầy đủ 22 quan hệ) vào %s",
                len(self.shard_buffer),
                shard_path.name,
            )
            self.shard_buffer.clear()
            self.shard_counter += 1
        except Exception as exc:
            self.logger.error("Lỗi ghi Shard xuống đĩa: %s", exc, exc_info=True)

    def _load_system_ontology(self) -> Tuple[dict, dict, dict]:
        if not self.ontology_path.exists():
            return {}, {}, {}

        with open(self.ontology_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scoped_mapping = {}

        for group_name in ["documentNamesByType", "documentNamesBySource"]:
            for raw_key, edge_type in data.get(group_name, {}).items():
                scoped_mapping[(group_name, str(raw_key))] = edge_type

        for full_key, edge_type in data.get("dynamic_keys", {}).items():
            if ":" in full_key:
                group_name, raw_key = full_key.split(":", 1)
                scoped_mapping[(group_name, str(raw_key))] = edge_type

        return (
            scoped_mapping,
            data.get("relationship_templates", {}),
            data.get("category_normalization_aliases", {})
        )

    def save_dynamic_mappings(self):
        temp_path = self.ontology_path.with_suffix(".tmp")
        data = {}
        if self.ontology_path.exists():
            with open(self.ontology_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        data.setdefault("dynamic_keys", {}).update(self.dynamic_maps)

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path.replace(self.ontology_path)

    @staticmethod
    def _normalize_text(text: Optional[str]) -> str:
        text = unicodedata.normalize("NFKC", str(text or "")).lower()
        return re.sub(r"\s+", " ", text).strip()

    def _parse_html_groups(self, html_dom: Any) -> Dict[str, set]:
        if not html_dom:
            return {}
        if isinstance(html_dom, str):
            html_dom = BeautifulSoup(html_dom, "html.parser")

        groups = {}
        headers = html_dom.find_all(["h1", "h2", "h3", "h4", "h5", "strong", "b", "div", "p"])
        for header in headers:
            if header.name in {"div", "p"} and header.find(["h1", "h2", "h3", "h4", "h5", "strong", "b"]):
                continue
            header_text = header.get_text(" ", strip=True)
            header_norm = self._normalize_text(header_text)
            if not any(kw in header_norm for kw in ["văn bản", "van ban", "căn cứ", "can cu", "pháp lệnh", "nghị định"]):
                continue

            category = re.sub(r"\(\d+\)", "", header_text).strip()
            docs = []
            sibling = header.next_sibling
            while sibling:
                if getattr(sibling, "name", None) in ["h1", "h2", "h3", "h4", "h5"]:
                    break
                if getattr(sibling, "find_all", None):
                    links = sibling.find_all(["a", "li"])
                    if links:
                        docs.extend([x.get_text(" ", strip=True) for x in links if x.get_text(" ", strip=True)])
                sibling = sibling.next_sibling

            if not docs:
                direct_links = header.find_all_next("a", limit=20)
                docs = [x.get_text(" ", strip=True) for x in direct_links if x.get_text(" ", strip=True)]

            if docs:
                groups[category] = set(self._normalize_text(x) for x in docs)
        return groups

    @staticmethod
    def _extract_doc_number_only(text: str) -> str:
        text_lower = text.lower()
        match = re.search(r"(\d+/\d+/[a-z0-9\-đcphuqd]+)", text_lower)
        if match:
            return match.group(1).replace(" ", "")
        match_cp = re.search(r"(\d+/[a-z0-9\-đcphuqd]+)", text_lower)
        if match_cp:
            return match_cp.group(1).replace(" ", "")
        return text_lower.strip()

    def _jaccard_fallback(self, raw_key: str, html_dom: Any, json_docs_list: list) -> Tuple[str, float]:
        json_set = set(self._extract_doc_number_only(d.get("name") or d.get("title") or "") for d in json_docs_list)
        json_set = {x for x in json_set if x}
        if not json_set:
            return f"REL_TYPE_{raw_key}", 0.0

        html_groups = self._parse_html_groups(html_dom)
        best_label = f"REL_TYPE_{raw_key}"
        highest_score = 0.0

        for category_name, html_docs_set in html_groups.items():
            html_norm_set = set(self._extract_doc_number_only(x) for x in html_docs_set)
            html_norm_set = {x for x in html_norm_set if x}

            intersection = html_norm_set.intersection(json_set)
            union = html_norm_set.union(json_set)
            if not union:
                continue

            score = len(intersection) / len(union)
            if score > highest_score:
                highest_score = score
                best_label = category_name

        if highest_score >= 0.5:
            return self._normalize_category_name_to_edge_type(best_label), highest_score
        return f"REL_TYPE_{raw_key}", highest_score

    def _normalize_category_name_to_edge_type(self, category_name: str) -> str:
        normalized = self._normalize_text(category_name)
        if normalized in self.category_aliases:
            return self.category_aliases[normalized]

        ascii_text = unicodedata.normalize("NFD", category_name).encode("ascii", "ignore").decode("ascii")
        ascii_text = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_").upper()
        return ascii_text or "UNKNOWN_RELATION"

    def _to_standard_relation(self, doc: dict, edge_type: str, direction: str, method: str, source_item: Any) -> Optional[dict]:
        target_id = str(doc.get("id", "")).strip()
        if not target_id:
            return None

        tmpl = self.edge_templates.get(edge_type, {})
        graph_layer = tmpl.get("graph_layer", "Operational")
        if direction not in {"INCOMING", "OUTGOING"}:
            return None

        source_doc_id = str(source_item.get("item_id", "") if hasattr(source_item, "get") else "")
        source_doc_number = str(source_item.get("doc_number", "") if hasattr(source_item, "get") else "")

        return {
            "target_id": target_id,
            "target_name": doc.get("name") or doc.get("title") or "",
            "edge_type": edge_type,
            "direction": direction,
            "graph_layer": graph_layer,
            "extraction_method": method,
            "source_doc_id": source_doc_id,
            "source_doc_number": source_doc_number,
        }

    def process_diagram(self, item: Any, html_dom: Any = None) -> Any:
        diagram_json = item.get("diagram_json") or {}
        if html_dom is None and item.get("html_raw"):
            html_dom = BeautifulSoup(item.get("html_raw"), "html.parser")

        relationships = []
        unresolved = []

        for group_name in ["documentNamesByType", "documentNamesBySource"]:
            group_data = diagram_json.get(group_name) or {}
            direction = "OUTGOING" if group_name == "documentNamesByType" else "INCOMING"

            for raw_key, docs_list in group_data.items():
                raw_key = str(raw_key)
                edge_type = self.static_mapping.get((group_name, raw_key))
                method = "static"

                if not edge_type:
                    if html_dom:
                        edge_type, score = self._jaccard_fallback(raw_key, html_dom, docs_list)
                        method = "dynamic_jaccard"
                    else:
                        edge_type = f"REL_TYPE_{raw_key}"
                        score = 0.0
                        method = "fallback"

                    if edge_type.startswith("REL_TYPE_"):
                        unresolved.append({"group": group_name, "key": raw_key, "score": score})
                        continue

                    if method == "dynamic_jaccard":
                        self.dynamic_maps[f"{group_name}:{raw_key}"] = edge_type
                        self.static_mapping[(group_name, raw_key)] = edge_type

                for doc in docs_list or []:
                    rel = self._to_standard_relation(doc, edge_type, direction, method, item)
                    if rel:
                        relationships.append(rel)

        item["relationships"] = relationships
        if unresolved:
            item["diagram_status"] = "INCONSISTENT"
            item["diagram_unresolved_keys"] = unresolved
        elif relationships:
            item["diagram_status"] = "VALID"
        else:
            item["diagram_status"] = "EMPTY"

        return item

    def process_item(self, item: Any, spider: Any) -> Any:
        # 1. Bóc tách DOM có sẵn trong RAM để lấy văn bản thuần cho AST Parser
        html_dom = item.get("html_dom") if hasattr(item, "get") else None
        if html_dom:
            raw_text = html_dom.get_text("\n", strip=True)
            item["text"] = raw_text
            item["full_text"] = raw_text

        # 2. Bóc tách quan hệ đồ thị HIN
        item = self.process_diagram(item, html_dom)

        # 3. Chuyển đổi Item sang dạng dictionary sạch, loại bỏ DOM tránh lỗi tuần tự hóa
        if isinstance(item, VietLawItem):
            clean_record = item.to_clean_dict()
        else:
            clean_record = dict(item)
            clean_record.pop("html_dom", None)
            for k, v in list(clean_record.items()):
                if hasattr(v, "value"):
                    clean_record[k] = v.value

        # 4. Gom cụm và ghi Shard Staging tự động khi không dùng Feed Exporter (-O)
        if not self.feed_export_active:
            self.shard_buffer.append(clean_record)
            if len(self.shard_buffer) >= self.shard_size:
                self._flush_buffer_to_shard()

        return clean_record