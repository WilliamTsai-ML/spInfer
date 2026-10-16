// WordDrill: single-word pronunciation on minimal pairs.
// Picks words weighted toward the learner's weak sounds and judges each attempt from
// the recognizer's transcript: the word itself, its foil (a known confusion), or neither.

import { normalizeWords } from './SpeechAssessor.js?v=9';
import { makeRng } from './VocabPlacement.js?v=9';

export class WordDrill {
  constructor({ pairs, learner, count = 10, seed = 1 }) {
    this.learner = learner;
    this.rng = makeRng(seed);
    this.items = this.pick(pairs, count);
    this.i = 0;
    this.log = [];
  }
  // Weak sounds get 3x weight, untested sounds 2x, strong sounds 1x; no sound twice in a row.
  pick(pairs, count) {
    const weak = new Set(this.learner.weakPhonemes(1).map(w => w.ph));
    const seen = this.learner.phonemes;
    const pool = pairs.map(p => ({ ...p, w: weak.has(p.ph) ? 3 : seen[p.ph] ? 1 : 2 }));
    const out = [];
    let lastPh = null;
    while (out.length < count && pool.length) {
      const cands = pool.filter(p => p.ph !== lastPh);
      const total = cands.reduce((s, p) => s + p.w, 0);
      let r = this.rng() * total, chosen = cands[cands.length - 1];
      for (const p of cands) { r -= p.w; if (r <= 0) { chosen = p; break; } }
      pool.splice(pool.indexOf(chosen), 1);
      out.push(chosen); lastPh = chosen.ph;
    }
    return out;
  }
  get current() { return this.items[this.i] ?? null; }
  get finished() { return this.i >= this.items.length; }
  get progress() { return this.i / this.items.length; }
  // Returns {verdict: 'hit'|'foil'|'other', heard}
  judge(transcript) {
    const item = this.current; if (!item) return null;
    const heard = normalizeWords(transcript);
    let verdict = 'other';
    if (heard.includes(item.word)) verdict = 'hit';
    else if (heard.includes(item.foil)) verdict = 'foil';
    if (verdict !== 'other') {
      this.learner.recordPhoneme(item.ph, verdict === 'hit');
      this.learner.update('speaking', -0.5, verdict === 'hit', 0.5);
    }
    this.log.push({ word: item.word, verdict, heard: heard.join(' ') });
    return { verdict, heard: heard.join(' '), item };
  }
  next() { this.i++; return this.current; }
  summary() {
    const hits = this.log.filter(l => l.verdict === 'hit').length;
    const judged = this.log.filter(l => l.verdict !== 'other').length;
    return { hits, judged, total: this.items.length };
  }
}
