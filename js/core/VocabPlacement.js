// VocabPlacement: adaptive yes/no vocabulary test.
// Picks the real word whose difficulty is closest to the current estimate,
// mixes in pseudowords to catch guessing, and stops when the estimate is tight enough.
// Deterministic for a given seed so it can be tested.

import { BANDS } from '../data/pack-en-zh.js';

export function makeRng(seed = 1) {
  let s = seed >>> 0 || 1;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

export class VocabPlacement {
  constructor({ vocab, pseudowords, learner, seed = 1, minItems = 12, maxItems = 24, targetSe = 0.55, pseudoEvery = 4 }) {
    this.learner = learner;
    this.skill = learner.skill('vocab');
    this.rng = makeRng(seed);
    this.minItems = minItems; this.maxItems = maxItems; this.targetSe = targetSe; this.pseudoEvery = pseudoEvery;
    this.pool = [];
    for (const [band, words] of Object.entries(vocab)) for (const w of words) this.pool.push({ word: w, band, d: BANDS[band], real: true });
    this.pseudo = pseudowords.map(w => ({ word: w, band: null, d: null, real: false }));
    this.shuffle(this.pool); this.shuffle(this.pseudo);
    this.asked = 0;
    this.falseAlarms = 0;
    this.pseudoShown = 0;
    this.current = null;
    this.log = [];
  }
  shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(this.rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } }
  get finished() {
    if (this.asked >= this.maxItems) return true;
    return this.asked >= this.minItems && this.skill.se <= this.targetSe;
  }
  get progress() { return Math.min(1, this.asked / this.maxItems); }
  next() {
    if (this.finished) return null;
    const wantPseudo = this.asked > 0 && this.asked % this.pseudoEvery === this.pseudoEvery - 1 && this.pseudo.length;
    if (wantPseudo) { this.current = this.pseudo.pop(); this.pseudoShown++; return this.current; }
    // Closest difficulty to theta, with a little jitter so the order is not fully predictable.
    const theta = this.skill.theta;
    let best = 0, bestScore = Infinity;
    for (let i = 0; i < this.pool.length; i++) {
      const score = Math.abs(this.pool[i].d - theta) + this.rng() * 0.6;
      if (score < bestScore) { bestScore = score; best = i; }
    }
    this.current = this.pool.splice(best, 1)[0];
    return this.current;
  }
  // known: the learner claims to know the word.
  answer(known) {
    const item = this.current; if (!item) return;
    this.asked++;
    if (item.real) this.skill.update(item.d, known);
    else if (known) {
      this.falseAlarms++;
      this.skill.nudge(-0.5);
      this.pseudoEvery = 2; // a guesser gets more pseudowords so the false-alarm rate is measured well
    }
    this.log.push({ word: item.word, real: item.real, band: item.band, known, theta: this.skill.theta, se: this.skill.se });
    this.current = null;
  }
  result() {
    const rate = this.pseudoShown > 0 ? this.falseAlarms / this.pseudoShown : 0;
    const guessing = rate >= 0.3;
    this.learner.flags.vocabGuessing = guessing;
    // Yes/no vocabulary tests are corrected for the false-alarm rate; a proportional discount here.
    if (rate > 0) this.skill.nudge(-1.5 * rate);
    return { theta: this.skill.theta, se: this.skill.se, cefr: this.skill.cefr, asked: this.asked, falseAlarms: this.falseAlarms, pseudoShown: this.pseudoShown, guessing };
  }
}
