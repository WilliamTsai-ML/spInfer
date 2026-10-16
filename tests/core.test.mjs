// Run: node tests/core.test.mjs
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { LearnerModel, thetaToCefr, probability } from '../js/core/LearnerModel.js';
import { VocabPlacement } from '../js/core/VocabPlacement.js';
import { SpeechAssessor, alignWords, normalizeWords } from '../js/core/SpeechAssessor.js';
import { VOCAB, PSEUDOWORDS, SPEAKING, BANDS } from '../js/data/pack-en-zh.js';

let passed = 0;
const test = (name, fn) => { try { fn(); passed++; } catch (e) { console.error('FAIL', name); throw e; } };

test('cefr mapping', () => {
  assert.equal(thetaToCefr(-2), 'A1'); assert.equal(thetaToCefr(0), 'B1'); assert.equal(thetaToCefr(1.2), 'B2'); assert.equal(thetaToCefr(3), 'C2');
  assert.ok(Math.abs(probability(0, 0) - 0.5) < 1e-9);
});

test('learner updates move theta the right way and tighten se', () => {
  const L = new LearnerModel();
  const v = L.skill('vocab');
  const se0 = v.se;
  for (let i = 0; i < 10; i++) v.update(0, true);
  assert.ok(v.theta > 0.8, 'ten correct answers at B1 should push above B1');
  assert.ok(v.se < se0);
  const json = JSON.parse(JSON.stringify(L));
  const L2 = new LearnerModel(json);
  assert.equal(L2.skill('vocab').n, 10);
});

// Simulate learners of known level and check the placement lands near them.
function simulate(trueTheta, seed) {
  const L = new LearnerModel();
  const p = new VocabPlacement({ vocab: VOCAB, pseudowords: PSEUDOWORDS, learner: L, seed });
  let rng = seed * 7919;
  const rand = () => { rng = (rng * 1103515245 + 12345) % 2147483648; return rng / 2147483648; };
  let item;
  while ((item = p.next())) {
    const known = item.real ? rand() < probability(trueTheta, item.d) : rand() < 0.05;
    p.answer(known);
  }
  return p.result();
}
test('placement converges for A1, B1 and C1 learners', () => {
  for (const [trueTheta, band] of [[-2, 'A1'], [0, 'B1'], [2, 'C1']]) {
    let errSum = 0, n = 0;
    for (let seed = 1; seed <= 20; seed++) { const r = simulate(trueTheta, seed); errSum += Math.abs(r.theta - trueTheta); n++; assert.ok(r.asked >= 18 && r.asked <= 36); }
    const mae = errSum / n;
    assert.ok(mae < 0.75, `${band}: mean abs error ${mae.toFixed(2)} too large`);
  }
});
test('guessers get flagged', () => {
  const L = new LearnerModel();
  const p = new VocabPlacement({ vocab: VOCAB, pseudowords: PSEUDOWORDS, learner: L, seed: 3 });
  let item; while ((item = p.next())) p.answer(true);
  const r = p.result();
  assert.ok(r.guessing); assert.ok(r.falseAlarms >= 3);
  assert.ok(r.theta < 3.2, 'saying yes to everything should not reach the top');
});

test('word alignment labels substitutions and omissions', () => {
  const e = normalizeWords('I think the three thieves are thirty.');
  const h = normalizeWords('I sink the tree thieves are thirty');
  const { status } = alignWords(e, h);
  assert.deepEqual(status, ['ok', 'wrong', 'ok', 'wrong', 'ok', 'ok', 'ok']);
  const { status: s2 } = alignWords(e, normalizeWords('I think three thieves thirty'));
  assert.equal(s2.filter(s => s === 'missing').length, 2);
});
test('assessor turns misses into phoneme evidence and updates speaking', () => {
  const L = new LearnerModel();
  const A = new SpeechAssessor({ learner: L });
  const r = A.assess(SPEAKING[0], 'I sink the tree thieves are thirty');
  assert.equal(r.phonemes['θ'].miss, 2); assert.equal(r.phonemes['θ'].hit, 2);
  assert.ok(L.skill('speaking').tested);
  A.assess(SPEAKING[0], 'I sink the tree sieves are dirty');
  assert.equal(L.weakPhonemes(2)[0].ph, 'θ');
});

test('data pack is consistent', () => {
  for (const band of Object.keys(VOCAB)) assert.ok(band in BANDS);
  const all = Object.values(VOCAB).flat();
  assert.equal(new Set(all).size, all.length, 'duplicate vocabulary word');
  for (const w of PSEUDOWORDS) assert.ok(!all.includes(w), `pseudoword ${w} is a real entry`);
  for (const s of SPEAKING) { const words = normalizeWords(s.text); for (const t of s.targets) for (const w of t.words) assert.ok(words.includes(w), `${s.id}: ${w} not in sentence`); }
});

test('layering: core imports only core/data; services never import ui; ui never imports web.js', () => {
  const src = (p) => readFileSync(p, 'utf8');
  for (const f of readdirSync('js/core')) { const s = src(`js/core/${f}`); assert.ok(!/from '\.\.\/(services|ui)\//.test(s), `core/${f} imports outside core/data`); }
  for (const f of readdirSync('js/services')) { const s = src(`js/services/${f}`); assert.ok(!/from '\.\.\/ui\//.test(s), `services/${f} imports ui`); }
  const ui = src('js/ui/App.js');
  assert.ok(!/services\/web\.js/.test(ui) && !/speechSynthesis|webkitSpeechRecognition|localStorage/.test(ui), 'ui touches a browser service API directly');
});

console.log(`${passed} core tests passed`);
