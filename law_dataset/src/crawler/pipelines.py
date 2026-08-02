import os
import json
import logging
import re
import unicodedata
from bs4 import BeautifulSoup
from paths import (
    DATA_DIR,
    RAW_HTML_DIR,
    RAW_PDF_DIR,
    RAW_DIAGRAM_DIR,
)
from crawler.items import HTMLStatus, PDFStatus
from confluent_kafka import Producer
import socket


class LegalOntologyMappingPipeline:
    def __init__(self, dynamic_maps=None):
        self.dynamic_maps = dynamic_maps if dynamic_maps is not None else {}
        self.ontology_path = os.path.join(os.path.dirname(__file__), "system_ontology_map.json")
        self.static_mapping, self.edge_templates = self._load_system_ontology()
        self.logger = logging.getLogger("LegalOntologyMappingPipeline")

        # Kafka Producer setup
        self.kafka_producer = Producer({
            'bootstrap.servers': 'localhost:9092',
            'client.id': socket.gethostname(),
            'acks': 'all',
            'linger.ms': 10,
            'batch.size': 65536,
        })
        self.kafka_topic = "law-documents"

    def open_spider(self, spider):
        for d in [RAW_HTML_DIR, RAW_PDF_DIR, RAW_DIAGRAM_DIR]:
            os.makedirs(d, exist_ok=True)
        spider.logger.info("[PIPELINE] LegalOntologyMappingPipeline ready (Kafka Push enabled)")

    def close_spider(self, spider):
        self.kafka_producer.flush()
        spider.logger.info("[PIPELINE] Kafka Producer flushed and closed")

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

    def _make_filename(self, item_id):
        return str(item_id).strip().replace("/", "-").replace("\\", "-")

    def _classify_pdf_status(self, pdf_local_path):
        if not os.path.exists(pdf_local_path):
            return PDFStatus.NOT_FOUND
        try:
            import fitz
            doc = fitz.open(pdf_local_path)
            text = "".join([p.get_text() for p in doc])
            doc.close()
            return PDFStatus.DIGITAL_TEXT if len(text.strip()) > 500 else PDFStatus.SCANNED_OR_CORRUPTED
        except Exception:
            return PDFStatus.SCANNED_OR_CORRUPTED

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
            
            # Tìm thẻ chứa danh sách các văn bản trực tiếp dưới header
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
                # Fallback: Quét các thẻ <a> lân cận
                direct_links = header.find_all_next("a", limit=20)
                docs = [x.get_text(" ", strip=True) for x in direct_links if x.get_text(" ", strip=True)]
                
            if docs:
                groups[category] = set(self._normalize_text(x) for x in docs)
        return groups

    def _extract_doc_number_only(self, text):
        # Bóc tách dạng số hiệu: VD 46/2016/NĐ-CP hoặc 171/2013/NĐ-CP
        match = re.search(r'(\d+/\d+/[a-z\-đdđcphuqd]+)', text.lower())
        if match:
            return match.group(1).replace(" ", "")
        # Bóc tách dạng số hiệu thô: VD 49/CP
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
                
        if highest_score >= 0.5: # Giảm threshold xuống 0.5 vì so khớp dựa trên số hiệu cực kỳ chính xác
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

    def _to_standard_relation(self, doc, edge_type, extraction_method, source_item):
        target_id = str(doc.get("id", "")).strip()
        if not target_id:
            return None
        tmpl = self.edge_templates.get(edge_type, {})
        direction = tmpl.get("direction", "OUTGOING")
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
            for raw_key, docs_list in group_data.items():
                raw_key = str(raw_key)
                edge_type = self.static_mapping.get((group_name, raw_key))
                method = "static"
                if not edge_type:
                    if html_dom:
                        edge_type, score = self._jaccard_fallback(raw_key, html_dom, docs_list)
                        self.dynamic_maps[f"{group_name}:{raw_key}"] = edge_type
                        method = "dynamic_jaccard"
                    else:
                        edge_type = f"REL_TYPE_{raw_key}"
                        score = 0.0
                        method = "fallback"
                    if edge_type.startswith("REL_TYPE_"):
                        unresolved.append({"group": group_name, "key": raw_key, "score": score})
                for doc in docs_list or []:
                    rel = self._to_standard_relation(doc, edge_type, method, item)
                    if rel:
                        relationships.append(rel)
        item["relationships"] = relationships
        item["diagram_status"] = "VALID" if relationships and not unresolved else "INCONSISTENT"
        if unresolved:
            item["diagram_unresolved_keys"] = unresolved
        return item

    def process_item(self, item, spider):
        item = self.process_diagram(item, item.get("html_dom"))

        # PUSH TO KAFKA
        try:
            self.kafka_producer.produce(
                self.kafka_topic,
                value=json.dumps(item, ensure_ascii=False).encode('utf-8')
            )
            self.kafka_producer.poll(0)
        except Exception as e:
            spider.logger.error(f"[KAFKA ERROR] Failed to push item {item.get('item_id')}: {e}")

        return item
