// retrieval_metrics.js — Metric truy xuất TẤT ĐỊNH cho VietLawBERT (promptfoo JS assertions)
//
// Triết lý (theo đề xuất testing của nhóm):
//   - Chỉ dùng so khớp đơn giản: contain-any (doc) / equal (điều, khoản) trên số hiệu + cấp bậc.
//   - KHÔNG dùng LLM-judge cho truy xuất (không tất định, phụ thuộc endpoint).
//   - KHÔNG để latency quyết định pass/fail (xem analyze_latency.py). Có hàm latencyReport
//     non-gating (weight:0) chỉ để hiển thị số đo từng dòng.
//   - Mỗi kết quả trả về {pass, score, reason} để FAIL luôn GIẢI THÍCH ĐƯỢC.
//
// Provider output kỳ vọng (chuỗi JSON):
//   { "retrieved_chunks": [...], "retrieved_docs": [...], "latency_ms": <num>, "answer": "..." }
// Test vars (từ vietlawbench_clean.jsonl): ground_truth_doc_number, ground_truth_chunk,
//   doc_norm, doc_core, article_id, clause_id, global_id, retrieval_level

function parseOutput(output) {
  if (typeof output === 'object' && output !== null) return output;
  try { return JSON.parse(output); } catch (e) { return null; }
}
function normDoc(s) { return String(s == null ? '' : s).replace(/\s+/g, '').toUpperCase(); }
function docCore(s) { const m = normDoc(s).match(/(\d+)\/(\d{4})/); return m ? `${m[1]}/${m[2]}` : normDoc(s); }
function artOf(chunkId) { const m = String(chunkId).match(/_art_(\d+)/); return m ? `art_${m[1]}` : null; }
function clOf(chunkId)  { const m = String(chunkId).match(/_cl_(\d+)/);  return m ? `cl_${m[1]}`  : null; }

// ---- doc_hit@k : contain-any trên số hiệu văn bản (một chiều, chuẩn hoá) ----
function docHitAtK(output, context) {
  const res = parseOutput(output);
  if (!res) return { pass: false, score: 0, reason: 'F4: output không parse được JSON' };
  const target = docCore(context.vars.doc_norm || context.vars.ground_truth_doc_number);
  const docs = (res.retrieved_docs || []).map(normDoc);
  const hit = docs.some(d => d.includes(target) || docCore(d) === target);
  return { pass: hit, score: hit ? 1 : 0,
    reason: hit ? `doc_hit: tìm thấy ${target}` : `MISS doc ${target}; trả về [${docs.slice(0,5).join(', ')}]` };
}

// ---- article_hit@k : equal trên (doc, điều) ----
function articleHitAtK(output, context) {
  const res = parseOutput(output);
  if (!res) return { pass: false, score: 0, reason: 'F4: output không parse được JSON' };
  const target = docCore(context.vars.doc_norm || context.vars.ground_truth_doc_number);
  const tgtArt = context.vars.article_id || artOf(context.vars.ground_truth_chunk);
  if (!tgtArt) return { pass: true, score: 1, weight: 0, reason: 'bỏ qua: GT không có cấp điều' };
  const docs = (res.retrieved_docs || []).map(docCore);
  const docOK = docs.includes(target);
  const chunks = res.retrieved_chunks || [];
  const artOK = chunks.some(c => artOf(c) === tgtArt);
  const pass = docOK && artOK;
  return { pass, score: pass ? 1 : 0,
    reason: pass ? `article_hit: ${target} ${tgtArt}`
                 : `MISS điều: doc ${docOK ? 'OK' : 'thiếu'}, ${tgtArt} ${artOK ? 'OK' : 'thiếu'}` };
}

// ---- clause_hit@k : equal trên global_id (doc#art#cl), chỉ áp khi GT có khoản ----
function clauseHitAtK(output, context) {
  const res = parseOutput(output);
  if (!res) return { pass: false, score: 0, reason: 'F4: output không parse được JSON' };
  const tgtCl = context.vars.clause_id || clOf(context.vars.ground_truth_chunk);
  if (!tgtCl) return { pass: true, score: 1, weight: 0, reason: 'bỏ qua: GT ở cấp điều, không có khoản' };
  const target = docCore(context.vars.doc_norm || context.vars.ground_truth_doc_number);
  const tgtArt = context.vars.article_id || artOf(context.vars.ground_truth_chunk);
  const docs = (res.retrieved_docs || []).map(docCore);
  const docOK = docs.includes(target);
  const chunks = res.retrieved_chunks || [];
  const clOK = chunks.some(c => artOf(c) === tgtArt && clOf(c) === tgtCl);
  const pass = docOK && clOK;
  return { pass, score: pass ? 1 : 0,
    reason: pass ? `clause_hit: ${target} ${tgtArt} ${tgtCl}`
                 : `MISS khoản ${tgtArt} ${tgtCl} (doc ${docOK ? 'OK' : 'thiếu'})` };
}

// ---- latency_report : KHÔNG gate (weight:0), chỉ hiển thị số đo ----
function latencyReport(output, context) {
  const res = parseOutput(output);
  const ms = res && typeof res.latency_ms === 'number' ? res.latency_ms : null;
  return { pass: true, score: 0, weight: 0,
    reason: ms == null ? 'latency: n/a' : `latency=${Math.round(ms)}ms (xem analyze_latency.py cho P50/P95/P99)` };
}

module.exports = { docHitAtK, articleHitAtK, clauseHitAtK, latencyReport };
