// LearnerModel: one ability estimate per skill, on a logit scale where 0 = B1.
// Every answer anywhere in the app is also a measurement and flows through update().
// No DOM, no platform. Serializable with toJSON()/fromJSON().

export const SKILLS = ['vocab', 'listening', 'speaking', 'grammar'];

const CEFR_CUTS = [
  ['A1', -1.5], ['A2', -0.5], ['B1', 0.5], ['B2', 1.5], ['C1', 2.5], ['C2', Infinity],
];

export function probability(theta, difficulty) {
  return 1 / (1 + Math.exp(-(theta - difficulty)));
}

export function thetaToCefr(theta) {
  for (const [band, cut] of CEFR_CUTS) if (theta < cut) return band;
  return 'C2';
}

export class SkillEstimate {
  constructor({ theta = 0, info = 0, n = 0, history = [] } = {}) {
    this.theta = theta;   // ability
    this.info = info;     // Fisher information gathered so far
    this.n = n;           // number of observations
    this.history = history; // last few {d, y}
  }
  get se() { return this.info > 0 ? 1 / Math.sqrt(this.info) : Infinity; }
  get cefr() { return thetaToCefr(this.theta); }
  get tested() { return this.n > 0; }
  // Elo-style update with a step size that shrinks as evidence accumulates.
  update(difficulty, correct, weight = 1) {
    const p = probability(this.theta, difficulty);
    const k = (1.2 / Math.sqrt(1 + this.n * 0.5)) * weight;
    this.theta += k * ((correct ? 1 : 0) - p);
    this.theta = Math.max(-3.5, Math.min(3.5, this.theta));
    this.info += p * (1 - p) * weight;
    this.n += 1;
    this.history.push({ d: difficulty, y: correct ? 1 : 0 });
    if (this.history.length > 50) this.history.shift();
    return this;
  }
  // Direct nudge for events that are not pass/fail items (e.g. a false alarm).
  nudge(delta) { this.theta = Math.max(-3.5, Math.min(3.5, this.theta + delta)); return this; }
  toJSON() { return { theta: this.theta, info: this.info, n: this.n, history: this.history }; }
}

export class LearnerModel {
  constructor(json = null) {
    this.skills = {};
    for (const s of SKILLS) this.skills[s] = new SkillEstimate(json?.skills?.[s]);
    this.phonemes = json?.phonemes ?? {}; // ph -> {hit, miss}
    this.flags = json?.flags ?? {};
    this.updatedAt = json?.updatedAt ?? null;
  }
  skill(name) { return this.skills[name]; }
  update(skill, difficulty, correct, weight) {
    this.skills[skill].update(difficulty, correct, weight);
    return this;
  }
  recordPhoneme(ph, hit) {
    const p = (this.phonemes[ph] ??= { hit: 0, miss: 0 });
    hit ? p.hit++ : p.miss++;
  }
  // Sounds with enough misses to be worth drilling, worst first.
  weakPhonemes(minAttempts = 2) {
    return Object.entries(this.phonemes)
      .map(([ph, { hit, miss }]) => ({ ph, hit, miss, rate: miss / (hit + miss) }))
      .filter(x => x.hit + x.miss >= minAttempts && x.rate >= 0.4)
      .sort((a, b) => b.rate - a.rate);
  }
  // Overall level: the median of the tested skills, so one bad probe does not dominate.
  overall() {
    const t = SKILLS.map(s => this.skills[s]).filter(s => s.tested).map(s => s.theta).sort((a, b) => a - b);
    if (!t.length) return null;
    const mid = t.length / 2;
    return t.length % 2 ? t[Math.floor(mid)] : (t[mid - 1] + t[mid]) / 2;
  }
  toJSON() {
    const skills = {};
    for (const s of SKILLS) skills[s] = this.skills[s].toJSON();
    return { skills, phonemes: this.phonemes, flags: this.flags, updatedAt: this.updatedAt };
  }
}
