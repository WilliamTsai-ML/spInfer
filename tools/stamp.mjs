// Stamp every relative module import with ?v=<n> so browsers fetch fresh files after a push.
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
const [,, root, v] = process.argv;
const walk = (d) => readdirSync(d).flatMap(f => { const p = join(d, f); return statSync(p).isDirectory() ? walk(p) : p.endsWith('.js') || p.endsWith('.html') ? [p] : []; });
for (const f of walk(root)) {
  if (f.includes('/tests/')) continue;
  const s = readFileSync(f, 'utf8');
  const out = s.replace(/(from\s+'|import\s+'|import\(')(\.[^'?]+\.js)(\?v=[^']*)?'/g, `$1$2?v=${v}'`);
  if (out !== s) writeFileSync(f, out);
}
console.log('stamped v=' + v);
