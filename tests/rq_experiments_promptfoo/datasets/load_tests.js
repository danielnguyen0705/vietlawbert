// load_tests.js — Nạp testcase từ vietlawbench_clean.jsonl thành test của promptfoo.
// Dùng trong config:  tests: file://../datasets/load_tests.js
// Có thể giới hạn số câu cho vòng CPU bằng biến môi trường:  VLB_LIMIT=150 promptfoo eval ...
const fs = require('fs');
const path = require('path');

module.exports = async function () {
  const p = path.join(__dirname, 'vietlawbench_clean.jsonl');
  const lines = fs.readFileSync(p, 'utf-8').split('\n').filter(Boolean);
  const limit = parseInt(process.env.VLB_LIMIT || '0', 10);
  const rows = limit > 0 ? lines.slice(0, limit) : lines;
  return rows.map((l) => {
    const r = JSON.parse(l);
    return {
      description: `${r.benchmark_id} [${r.retrieval_level}]`,
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
        retrieval_level: r.retrieval_level,
      },
    };
  });
};
