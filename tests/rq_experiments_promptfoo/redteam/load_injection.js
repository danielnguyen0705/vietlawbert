// load_injection.js — Nạp injection_cases.jsonl thành test của promptfoo.
const fs = require('fs'); const path = require('path');
module.exports = async function () {
  const p = path.join(__dirname, 'injection_cases.jsonl');
  return fs.readFileSync(p, 'utf-8').split('\n').filter(Boolean).map((l) => {
    const r = JSON.parse(l);
    return { description: r.attack_type, vars: {
      query: r.query,
      must_not_contain_doc: r.must_not_contain_doc || '',
      expect_abstain: !!r.expect_abstain,
    }};
  });
};
