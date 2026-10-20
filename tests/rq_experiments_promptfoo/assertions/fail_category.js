// fail_category.js — Gắn nhãn NGUYÊN NHÂN cho mỗi dòng eval (non-gating, weight:0).
//
// Theo yêu cầu: "eval có thể FAIL nhưng khi có kết quả phải giải thích được vì sao FAIL".
// Phân loại:
//   F1 = GT nghi ngờ (lọt lưới làm sạch): chunk preamble / article != 'Điều N'
//   F2 = Retrieval miss: doc không nằm trong top-k
//   F3 = Metric khắt khe: doc/điều ĐÚNG nhưng sai ở cấp khoản (chunk-exact)
//   F4 = Hạ tầng: output rỗng / không parse / latency bất thường (timeout)
//   OK = trúng tới cấp sâu nhất mà GT yêu cầu
//
// Luôn pass:true để không ảnh hưởng điểm; nhãn nằm ở reason để lọc/thống kê về sau.

function parseOutput(o){ if(o&&typeof o==='object')return o; try{return JSON.parse(o);}catch(e){return null;} }
function normDoc(s){ return String(s==null?'':s).replace(/\s+/g,'').toUpperCase(); }
function docCore(s){ const m=normDoc(s).match(/(\d+)\/(\d{4})/); return m?`${m[1]}/${m[2]}`:normDoc(s); }
function artOf(c){ const m=String(c).match(/_art_(\d+)/); return m?`art_${m[1]}`:null; }
function clOf(c){ const m=String(c).match(/_cl_(\d+)/); return m?`cl_${m[1]}`:null; }

function classify(output, context) {
  const v = context.vars || {};
  // F1 — GT nghi ngờ lọt lưới
  const art = v.article_id || artOf(v.ground_truth_chunk);
  if (String(v.ground_truth_chunk).includes('_preamble') || !art) {
    return { pass: true, score: 0, weight: 0, reason: 'fail_category=F1 (GT nghi ngờ)' };
  }
  const res = parseOutput(output);
  if (!res || res.latency_ms == null || (res.retrieved_docs || []).length === 0) {
    return { pass: true, score: 0, weight: 0, reason: 'fail_category=F4 (hạ tầng/empty)' };
  }
  const target = docCore(v.doc_norm || v.ground_truth_doc_number);
  const docs = (res.retrieved_docs || []).map(docCore);
  if (!docs.includes(target)) {
    return { pass: true, score: 0, weight: 0, reason: 'fail_category=F2 (retrieval miss doc)' };
  }
  const cl = v.clause_id || clOf(v.ground_truth_chunk);
  const chunks = res.retrieved_chunks || [];
  const clOK = !cl || chunks.some(c => artOf(c) === art && clOf(c) === cl);
  if (!clOK) {
    return { pass: true, score: 0, weight: 0, reason: 'fail_category=F3 (đúng điều, lệch khoản)' };
  }
  return { pass: true, score: 0, weight: 0, reason: 'fail_category=OK' };
}

module.exports = { classify };
