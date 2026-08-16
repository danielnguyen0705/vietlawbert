import os
import json
import logging
import re
import unicodedata
from bs4 import BeautifulSoup
from confluent_kafka import Producer
import socket


class LegalOntologyMappingPipeline:
    def __init__(self, dynamic_maps=None):
        self.dynamic_maps = dynamic_maps if dynamic_maps is not None else {}
        self.ontology_path = os.path.join(os.path.dirname(__file__), "system_ontology_map.json")
        self.static_mapping, self.edge_templates = self._load_system_ontology()
        self.logger = logging.getLogger("LegalOntologyMappingPipeline")
        self.kafka_enabled = os.getenv("KAFKA_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        self.publish_empty = os.getenv("KAFKA_PUBLISH_EMPTY", "0").strip().lower() in {"1", "true", "yes"}

        # Kafka Producer setup
        self.kafka_producer = Producer({
                'bootstrap.servers': os.getenv('KAFKA_BROKER', 'localhost:9092'),
                'client.id': socket.gethostname(),
                'acks': 'all',
                'linger.ms': 10,
                'batch.size': 65536,
                'compression.type': 'zstd',
                'enable.idempotence': True,
            }) if self.kafka_enabled else None
        self.kafka_topic = os.getenv("KAFKA_TOPIC", "law-documents-v5")
        self.delivery_errors = []

    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        pipeline.crawler = crawler
        return pipeline

    def open_spider(self):
        self.logger.info(
            "[PIPELINE] LegalOntologyMappingPipeline ready (Kafka Push %s)",
            "enabled" if self.kafka_enabled else "disabled",
        )

    def close_spider(self):
        if self.kafka_producer is not None:
            remaining = self.kafka_producer.flush(30)
            if remaining or self.delivery_errors:
                raise RuntimeError(
                    f"Kafka delivery incomplete: remaining={remaining}, errors={len(self.delivery_errors)}"
                )
        if self.dynamic_maps:
            self.save_dynamic_mappings()
        self.logger.info("[PIPELINE] closed%s", " after Kafka flush" if self.kafka_enabled else "")

    def _delivery_report(self, err, msg):
        if err is not None:
            self.delivery_errors.append(str(err))
            self.logger.error("Kafka delivery failed: %s", err)

    def _load_system_ontology(self):
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scoped_mapping = {}
        for group_name in ["documentNamesByType", "documentNamesBySource"]:
            for raw_key, edge_type in data.get(group_name, {}).items():
                scoped_mapping[(group_name, str(raw_key))] = edge_type
        return scoped_mapping, data.get("relationship_templates", {})

    def save_dynamic_mappings(self):
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("dynamic_keys", {}).update(self.dynamic_maps)
        with open(self.ontology_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _normalize_text(self, text):
        text = unicodedata.normalize("NFKC", str(text or "")).lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _parse_html_groups(self, html_dom):
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
            if not any(kw in header_norm for kw in ["văn bản", "van ban", "căn cứ", "can cu", "pháp lệnh", "phap lenh", "nghị định", "nghi dinh"]):
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

    def _extract_doc_number_only(self, text):
        match = re.search(r'(\d+/\d+/[a-z\-đdđcphuqd]+)', text.lower())
        if match:
            return match.group(1).replace(" ", "")
        match_cp = re.search(r'(\d+/[cphqd]+)', text.lower())
        if match_cp:
            return match_cp.group(1).replace(" ", "")
        return text

    def _jaccard_fallback(self, raw_key, html_dom, json_docs_list):
        json_set = set(self._extract_doc_number_only(d.get("name") or d.get("title") or "") for d in json_docs_list)
        json_set = {x for x in json_set if x}
        if not json_set:
            return f"REL_TYPE_{raw_key}", 0.0

        html_groups = self._parse_html_groups(html_dom)
        best_label = f"REL_TYPE_{raw_key}"
        highest_score = 0.0

        for category_name, html_docs_set in html_groups.items():
            html_normalized_set = set(self._extract_doc_number_only(x) for x in html_docs_set)
            html_normalized_set = {x for x in html_normalized_set if x}

            intersection = html_normalized_set.intersection(json_set)
            union = html_normalized_set.union(json_set)

            if not union:
                continue

            score = len(intersection) / len(union)
            if score > highest_score:
                highest_score = score
                best_label = category_name

        if highest_score >= 0.5:
            return self._normalize_category_name_to_edge_type(best_label), highest_score
        return f"REL_TYPE_{raw_key}", highest_score

    def _normalize_category_name_to_edge_type(self, category_name):
        normalized = self._normalize_text(category_name)
        direct = {
            "căn cứ ban hành": "CAN_CO_BAN_HANH",
            "can cu ban hanh": "CAN_CO_BAN_HANH",
            "văn bản được thay thế": "VAN_BAN_DUOC_THAY_THE",
            "van ban duoc thay the": "VAN_BAN_DUOC_THAY_THE",
            "văn bản bị bãi bỏ": "VAN_BAN_BI_BAI_BO",
            "van ban bi bai bo": "VAN_BAN_BI_BAI_BO",
            "văn bản áp dụng": "VAN_BAN_AP_DUNG",
            "van ban ap dung": "VAN_BAN_AP_DUNG",
            "văn bản dẫn chiếu": "VAN_BAN_DAN_CHIEU",
            "van ban dan chieu": "VAN_BAN_DAN_CHIEU",
            "văn bản quy định chi tiết hướng dẫn": "VAN_BAN_QUY_DINH_CHI_TIET_HUONG_DAN",
            "van ban quy dinh chi tiet huong dan": "VAN_BAN_QUY_DINH_CHI_TIET_HUONG_DAN",
            "văn bản sửa đổi bổ sung": "VAN_BAN_SUA_DOI_BO_SUNG",
            "van ban sua doi bo sung": "VAN_BAN_SUA_DOI_BO_SUNG",
            "văn bản thay thế": "VAN_BAN_THAY_THE",
            "van ban thay the": "VAN_BAN_THAY_THE"
        }
        if normalized in direct:
            return direct[normalized]
        ascii_text = unicodedata.normalize("NFD", category_name).encode("ascii", "ignore").decode("ascii")
        ascii_text = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_").upper()
        return ascii_text or "UNKNOWN_RELATION"

    def _to_standard_relation(self, doc, edge_type, direction, extraction_method, source_item):
        target_id = str(doc.get("id", "")).strip()
        if not target_id:
            return None
        tmpl = self.edge_templates.get(edge_type, {})
        graph_layer = tmpl.get("graph_layer", "Operational")
        if direction not in {"INCOMING", "OUTGOING"}:
            return None
        if graph_layer not in {"Hierarchical", "Temporal", "Operational"}:
            return None
        return {
            "target_id": target_id,
            "target_name": doc.get("name") or doc.get("title") or "",
            "edge_type": edge_type,
            "direction": direction,
            "graph_layer": graph_layer,
            "extraction_method": extraction_method,
            "source_doc_id": source_item.get("item_id"),
            "source_doc_number": source_item.get("doc_number", ""),
        }

    def process_diagram(self, item, html_dom=None):
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
                        # Giữ raw diagram trong artifact để phân loại sau; không đưa
                        # relationship type chưa biết vào knowledge graph production.
                        continue
                    if method == "dynamic_jaccard":
                        self.dynamic_maps[f"{group_name}:{raw_key}"] = edge_type
                for doc in docs_list or []:
                    rel = self._to_standard_relation(doc, edge_type, direction, method, item)
                    if rel:
                        relationships.append(rel)
        item["relationships"] = relationships
        if unresolved:
            item["diagram_status"] = "INCONSISTENT"
        elif relationships:
            item["diagram_status"] = "VALID"
        else:
            item["diagram_status"] = "EMPTY"
        if unresolved:
            item["diagram_unresolved_keys"] = unresolved
        return item

    def process_item(self, item):
        item = self.process_diagram(item, item.get("html_dom"))
        html_status = getattr(item.get("html_status"), "value", item.get("html_status"))
        if html_status != "VALID" and not self.publish_empty:
            self.logger.warning(
                "[CONTENT QUARANTINE] %s không có text hợp lệ (rescue=%s)",
                item.get("item_id"),
                item.get("rescue_status"),
            )
            if hasattr(self, "crawler"):
                self.crawler.stats.inc_value("kafka/quarantined_empty")
            return item
        if not self.kafka_enabled:
            return item
        try:
            payload = json.dumps(dict(item), ensure_ascii=False).encode('utf-8')
            try:
                self.kafka_producer.produce(
                    self.kafka_topic,
                    key=str(item.get('item_id', '')).encode('utf-8'),
                    value=payload,
                    callback=self._delivery_report,
                )
            except BufferError:
                self.kafka_producer.poll(1.0)
                self.kafka_producer.produce(
                    self.kafka_topic,
                    key=str(item.get('item_id', '')).encode('utf-8'),
                    value=payload,
                    callback=self._delivery_report,
                )
            self.kafka_producer.poll(0)
        except Exception as e:
            self.logger.error(f"[KAFKA ERROR] Failed to push item {item.get('item_id')}: {e}")
            raise
        return item
