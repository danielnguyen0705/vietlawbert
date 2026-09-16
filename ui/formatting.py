def shown(value):
    return "Chưa có" if value is None else str(value)


def format_answer(data, turn_id):
    demo = data["mode"] == "mock"
    elements = []
    for index, context in enumerate(data["contexts"], 1):
        name = f"Căn cứ {index} — lượt {turn_id}"
        content = (
            ("**DỮ LIỆU MINH HỌA — điểm số giả lập**\n\n" if demo else "")
            + f"**Số hiệu:** {shown(context.get('doc_number'))}\n\n"
            + f"**Điều/khoản:** {shown(context.get('hierarchy_path'))}\n\n"
            + context["content"]
            + f"\n\nRerank: {shown(context.get('final_rerank_score'))} | Graph boost: {shown(context.get('graph_boost'))}"
        )
        elements.append((name, content))
    score = data.get("attribution_score")
    attribution = "Chưa có" if score is None else f"{score * 100:.1f}%"
    answer = data["answer"]
    if demo and "Chế độ demo" not in answer:
        answer = "**Chế độ demo — dữ liệu minh họa**\n\n" + answer
    if elements:
        answer += "\n\n**Căn cứ:** " + " · ".join(name for name, _ in elements)
    else:
        answer += "\n\n_Chưa có căn cứ._"
    answer += f"\n\n⏱ {data['latency_ms']:.0f} ms | Attribution — đối sánh trích dẫn: {attribution}"
    answer += " (minh họa)" if demo else ""
    answer += "\n\n_Chỉ số đối sánh trích dẫn không phải độ chính xác pháp lý._"
    return answer, elements
