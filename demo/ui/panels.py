"""
demo/ui/panels.py — HTML Renderer cho Evidence Inspector & Scientific Metrics
==============================================================================
Hai hàm render HTML thuần, không phụ thuộc Gradio state.
Được gọi từ handlers.py sau mỗi lượt chat.

FIX (2026-10): Evidence cards có thể expand/collapse để xem nội dung đầy đủ.
- Mỗi card có nút "▼ Xem đầy đủ" → toggle hiện toàn bộ nội dung
- Container card không bị overflow-hidden để nội dung có thể hiện ra
- Dùng <details>/<summary> thuần HTML5 — không cần JS phức tạp
"""

from __future__ import annotations

from typing import List, Dict, Any


# ─── Evidence Inspector ─────────────────────────────────────────────────────

def render_evidence(docs: List[Dict[str, Any]]) -> str:
    """Render danh sách chunks truy xuất thành HTML card có thể expand."""
    if not docs:
        return (
            "<p style='color:#888;padding:12px;font-style:italic'>"
            "Chưa có truy vấn nào. Nhập câu hỏi và nhấn Gửi.</p>"
        )

    cards = []
    for i, d in enumerate(docs, 1):
        doc_num  = d.get("doc_number", "N/A")
        h_path   = d.get("hierarchy_path", "")
        content  = d.get("content") or d.get("text") or ""
        eff_date = d.get("effective_date", "")
        rerank   = float(d.get("final_rerank_score") or d.get("score") or 0.0)
        graph_b  = float(d.get("graph_boost", 0.0))
        rrf      = float(d.get("rrf_score", 0.0))

        bar_pct   = min(int(rerank * 100), 100)
        bar_color = "#27ae60" if rerank >= 0.7 else "#f39c12" if rerank >= 0.4 else "#e74c3c"

        date_badge = (
            f"<span style='font-size:10px;color:#888;margin-left:6px'>• {eff_date}</span>"
            if eff_date else ""
        )

        # Nội dung preview (250 ký tự) và nội dung đầy đủ
        # Dùng <details>/<summary> — HTML5 native expand/collapse, không cần JS
        preview  = content[:250] + ("…" if len(content) > 250 else "")
        has_more = len(content) > 250

        if has_more:
            # Wrap bằng <details> để có thể expand xem toàn bộ nội dung
            # open_tag không có JS — chỉ cần click <summary>
            content_html = f"""
<details style="margin-bottom:4px">
  <summary style="cursor:pointer;font-size:12px;color:#5c6bc0;
                  user-select:none;list-style:none;outline:none;
                  padding:2px 0">
    <span class="ev-preview" style="font-size:12.5px;color:#333;line-height:1.5">
      {preview}
    </span>
    <br/>
    <span style="font-size:11px;color:#1565c0;text-decoration:underline">
      ▼ Xem đầy đủ ({len(content)} ký tự)
    </span>
  </summary>
  <div style="font-size:12.5px;color:#333;line-height:1.6;
              margin-top:6px;padding:8px;background:#f0f4ff;
              border-radius:6px;border-left:3px solid #5c6bc0;
              white-space:pre-wrap;word-break:break-word;
              max-height:400px;overflow-y:auto">
    {content}
  </div>
</details>"""
        else:
            content_html = (
                f"<div style='font-size:12.5px;color:#333;line-height:1.5;margin-bottom:8px'>"
                f"{content}</div>"
            )

        card = f"""
<div style="border:1px solid #dde;border-radius:8px;padding:10px 12px;
            margin-bottom:10px;background:#fafcff;
            box-shadow:0 1px 3px rgba(0,0,0,.06);
            overflow:visible">
  <div style="font-size:13px;font-weight:600;color:#1a237e;margin-bottom:2px">
    #{i}&nbsp; {doc_num}
    {date_badge}
  </div>
  <div style="font-size:11px;color:#5c6bc0;margin-bottom:6px">{h_path}</div>
  {content_html}
  <div style="display:flex;gap:14px;font-size:11px;color:#555;
              margin-top:6px;margin-bottom:4px;flex-wrap:wrap">
    <span>Rerank: <b style="color:#1565c0">{rerank:.3f}</b></span>
    <span>Graph+: <b style="color:#2e7d32">+{graph_b:.3f}</b></span>
    <span>RRF: <b style="color:#6a1b9a">{rrf:.4f}</b></span>
  </div>
  <div style="background:#e8eaf6;border-radius:4px;height:5px">
    <div style="background:{bar_color};width:{bar_pct}%;height:5px;
                border-radius:4px;transition:width .3s"></div>
  </div>
</div>"""
        cards.append(card)

    # Wrapper với scroll — toàn bộ panel cuộn, không cắt nội dung của từng card
    return (
        "<div style='overflow-y:auto;max-height:100%;padding-right:4px'>"
        + "".join(cards)
        + "</div>"
    )


# ─── Scientific Metrics ─────────────────────────────────────────────────────

def render_metrics(
    retrieval_latency_ms: float = 0.0,
    generation_latency_ms: float = 0.0,
    n_docs: int = 0,
    mode: str = "—",
    top_k: int = 5,
    attribution_score: float = 0.0,
    model_used: str = "—",
    is_mock: bool = False,
) -> str:
    """
    Render bảng metrics khoa học phục vụ RQ1–RQ4.

      - Retrieval latency (ms)   → SLA < 500ms (RQ1)
      - Generation latency (ms)  → tổng E2E
      - # Docs retrieved         → top_k thực tế
      - Attribution score (0–1)  → faithfulness (RQ4)
      - Model used               → cascade fallback (RQ3)
    """
    total_ms    = retrieval_latency_ms + generation_latency_ms
    retr_color  = "#27ae60" if retrieval_latency_ms < 500 else "#e74c3c"
    total_color = "#27ae60" if total_ms < 8000 else "#e74c3c"
    attr_color  = "#1565c0" if attribution_score >= 0.5 else "#e65100"
    mock_badge  = (
        "<div style='font-size:10px;color:#e65100;text-align:center;"
        "margin-bottom:6px'>⚠ MOCK backend</div>"
        if is_mock else ""
    )

    tiles = [
        ("🔍", f"{retrieval_latency_ms:.0f}", "ms", "Retrieval", retr_color, "#e8f5e9"),
        ("💬", f"{generation_latency_ms:.0f}", "ms", "Generation", total_color, "#fff8e1"),
        ("📄", str(n_docs), "", "Docs retrieved", "#37474f", "#f5f5f5"),
        ("🎯", f"{attribution_score:.2f}", "", "Attribution (RQ4)", attr_color, "#e3f2fd"),
    ]

    tiles_html = "".join(
        f"""<div style="text-align:center;background:{bg};border-radius:8px;
                        padding:8px 6px;min-width:80px">
          <div style="font-size:22px">{icon}</div>
          <div style="font-size:18px;font-weight:700;color:{color}">
            {val}{unit}
          </div>
          <div style="font-size:10px;color:#666;margin-top:2px">{label}</div>
        </div>"""
        for icon, val, unit, label, color, bg in tiles
    )

    return f"""
<div style="padding:6px">
  {mock_badge}
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
    {tiles_html}
  </div>
  <div style="display:flex;gap:16px;font-size:11.5px;color:#444;
              border-top:1px solid #e0e0e0;padding-top:6px;flex-wrap:wrap">
    <span>⚙ Mode: <b>{mode}</b></span>
    <span>📊 Top-K: <b>{top_k}</b></span>
    <span>🤖 Model: <b>{model_used}</b></span>
    <span>⏱ Total: <b style="color:{total_color}">{total_ms:.0f} ms</b></span>
  </div>
</div>"""
