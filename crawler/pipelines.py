"""
pipelines.py - Pipeline bóc tách bản đồ tri thức quan hệ pháp luật và nạp luồng Kafka.
Chuẩn hóa Schema đồ thị, tính điểm tương đồng Jaccard và đóng gói Kafka Envelope an toàn.
"""

from __future__ import annotations

import os
import json
import logging
import re
import socket
import unicodedata
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from bs4 import BeautifulSoup

from configs.config import config
from artifacts.canonical import encode_kafka_envelope
from .items import VietLawItem


class LegalOntologyMappingPipeline:
    def __init__(self):
        self.ontology_path = Path(__file__).resolve().parent / "system_ontology_map.json"
        self.static_mapping, self.edge_templates, self.category_aliases = self._load_system_ontology()
        self.logger = logging.getLogger("VietLawBERT_Pipeline")

        self.kafka_enabled = os.getenv("KAFKA_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        self.publish_empty = os.getenv("KAFKA_PUBLISH_EMPTY", "0").strip().lower() in {"1", "true", "yes"}

        self.kafka_producer = None
        self.kafka_topic = getattr(config, "KAFKA_TOPIC", "law-documents-v5")
        self.delivery_errors = []
        self.dynamic_maps = {}

        if self.kafka_enabled:
            try:
                from confluent_kafka import Producer
                self.kafka_producer = Producer({
                    "bootstrap.servers": getattr(config, "KAFKA_BROKER", "localhost:9092"),
                    "client.id": socket.gethostname(),
                    "acks": "all",
                    "linger.ms": 10,
                    "batch.size": 65536,
                    "compression.type": "zstd",
                    "enable.idempotence": True,
                })
            except ImportError:
                self.logger.warning("Thư viện confluent_kafka chưa được cài đặt. Tự động vô hiệu hóa đẩy Kafka.")
                self.kafka_enabled = False

    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        pipeline.crawler = crawler
        return pipeline

    def open_spider(self, spider):
        self.logger.info(
            "[PIPELINE KHỞI ĐỘNG] LegalOntologyMappingPipeline sẵn sàng. Đẩy Kafka: %s",
            "KÍCH HOẠT" if self.kafka_enabled else "TẮT",
        )

    def close_spider(self, spider):
        flush_error = None
        if self.kafka_producer is not None:
            remaining = self.kafka_producer.flush(30)
            if remaining or self.delivery_errors:
                flush_error = RuntimeError(
                    f"Thất thoát bản tin Kafka khi đóng spider: tồn đọng={remaining}, lỗi={len(self.delivery_errors)}"
                )

        # Đảm bảo lưu cache ontology ngoại trừ trường hợp dừng khẩn cấp
        if self.dynamic_maps:
            try:
                self.save_dynamic_mappings()
            except Exception as exc:
                self.logger.error("Lỗi khi lưu dynamic ontology cache: %s", exc)

        self.logger.info("[PIPELINE ĐÓNG] Hoàn tất xả bộ đệm an toàn.")
        if flush_error:
            raise flush_error

    def _delivery_report(self, err, msg):
        if err is not None:
            self.delivery_errors.append(str(err))
            self.logger.error("Giao dịch phát Kafka thất bại: %s", err)

    def _load_system_ontology(self) -> Tuple[dict, dict, dict]:
        if not self.ontology_path.exists():
            return {}, {}, {}

        with open(self.ontology_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scoped_mapping = {}

        # 1. Nạp mapping tĩnh theo nhóm
        for group_name in ["documentNamesByType", "documentNamesBySource"]:
            for raw_key, edge_type in data.get(group_name, {}).items():
                scoped_mapping[(group_name, str(raw_key))] = edge_type

        # 2. Nạp mapping động đã lưu từ các lượt crawl trước
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
        """Ghi nhận các khóa quan hệ mới theo cơ chế ghi tệp nguyên tử (.tmp -> replace)."""
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
        # 1. Bóc tách quan hệ đồ thị
        html_dom = item.get("html_dom") if hasattr(item, "get") else None
        item = self.process_diagram(item, html_dom)

        # 2. Kiểm tra chất lượng nội dung tối thiểu
        html_status = getattr(item.get("html_status"), "value", item.get("html_status"))
        if html_status != "VALID" and not self.publish_empty:
            self.logger.warning(
                "[PHÂN VÙNG CÁCH LY] Văn bản %s không có văn bản hợp lệ (cứu hộ=%s)",
                item.get("item_id"),
                item.get("rescue_status"),
            )
            if hasattr(self, "crawler") and self.crawler.stats:
                self.crawler.stats.inc_value("kafka/quarantined_empty")
            return item

        # 3. Chuẩn hóa Item thành dictionary sạch, loại bỏ các trường DOM không thể tuần tự hóa
        if isinstance(item, VietLawItem):
            clean_record = item.to_clean_dict()
        else:
            clean_record = dict(item)
            clean_record.pop("html_dom", None)
            for k, v in list(clean_record.items()):
                if hasattr(v, "value"):
                    clean_record[k] = v.value

        if not self.kafka_enabled or self.kafka_producer is None:
            return item

        # 4. Đóng gói chuẩn Kafka Envelope (Gzip base64 + SHA-256)
        try:
            kafka_envelope_bytes = encode_kafka_envelope(clean_record)
            msg_key = str(clean_record.get("item_id", "")).encode("utf-8")

            try:
                self.kafka_producer.produce(
                    self.kafka_topic,
                    key=msg_key,
                    value=kafka_envelope_bytes,
                    callback=self._delivery_report,
                )
            except BufferError:
                self.kafka_producer.poll(1.0)
                self.kafka_producer.produce(
                    self.kafka_topic,
                    key=msg_key,
                    value=kafka_envelope_bytes,
                    callback=self._delivery_report,
                )
            self.kafka_producer.poll(0)
        except Exception as exc:
            self.logger.error("[KAFKA LỖI] Không thể đóng gói hoặc đẩy Document %s: %s", clean_record.get("item_id"), exc)
            raise

        return item
