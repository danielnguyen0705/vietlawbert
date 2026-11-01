// load_tests.js — Nạp testcase thành test của promptfoo.
// Dataset mặc định: vietlawbench_clean.jsonl. Đổi bộ test bằng biến môi trường:
//     VLB_DATASET=vietlawbench_multihop.jsonl  npx promptfoo eval -c ...
// Giới hạn số câu cho vòng CPU:  VLB_LIMIT=150
const fs = require('fs');
const path = require('path');

module.exports = async function () {
  const ds = process.env.VLB_DATASET || 'vietlawbench_clean.jsonl';
  const p = path.isAbsolute(ds) ? ds : path.join(__dirname, ds);
  const lines = fs.readFileSync(p, 'utf-8').split('\n').filter(Boolean);
  const limit = parseInt(process.env.VLB_LIMIT || '0', 10);
  const rows = limit > 0 ? lines.slice(0, limit) : lines;
  return rows.map((l) => {
    const r = JSON.parse(l);
    return {
      // retrieval_level vắng (bộ multi-hop) -> mặc định 'document' để metric không lỗi
      description: `${r.benchmark_id} [${r.retrieval_level || r.query_type || 'document'}]`,
      vars: {
        query: r.query,
        ground_truth_doc_number: r.ground_truth_doc_number,
        ground_truth_chunk: r.ground_truth_chunk,
        ground_truth_article: r.ground_truth_article,
        doc_norm: r.doc_norm,
        doc_core: r.doc_core,
        article_id: r.article_id,
        clause_id: r.clause_id,
        global_id: r.global_id,
        retrieval_level: r.retrieval_level || 'document',
        // Trường bổ sung của bộ multi-hop (phục vụ phân tích theo quan hệ):
        anchor_doc_number: r.anchor_doc_number,
        relation: r.relation,
        query_type: r.query_type,
      },
    };
  });
};
