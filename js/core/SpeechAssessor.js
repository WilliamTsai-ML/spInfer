// SpeechAssessor: compares what the learner said (a transcript) with the target sentence,
// word by word, and turns misses into phoneme evidence.
// Word-level for now; a phoneme-level acoustic scorer plugs in behind the same result shape later.

import { BANDS } from '../data/pack-en-zh.js?v=9';

export function normalizeWords(text) {
  return (text || '').toLowerCase().replace(/[^a-z'\s]/g, ' ').split(/\s+/).filter(Boolean);
}

// Levenshtein alignment over word sequences. Returns per-expected-word status.
export function alignWords(expected, heard) {
  const n = expected.length, m = heard.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 0; i <= n; i++) dp[i][0] = i;
  for (let j = 0; j <= m; j++) dp[0][j] = j;
  for (let i = 1; i <= n; i++) for (let j = 1; j <= m; j++) {
    const sub = dp[i - 1][j - 1] + (expected[i - 1] === heard[j - 1] ? 0 : 1);
    dp[i][j] = Math.min(sub, dp[i - 1][j] + 1, dp[i][j - 1] + 1);
  }
  // Walk back to label each expected word.
  const status = new Array(n).fill('missing');
  const heardAs = new Array(n).fill(null);
  let i = n, j = m;
  while (i > 0 && j > 0) {
    const cost = expected[i - 1] === heard[j - 1] ? 0 : 1;
    if (dp[i][j] === dp[i - 1][j - 1] + cost) {
      status[i - 1] = cost ? 'wrong' : 'ok'; heardAs[i - 1] = heard[j - 1]; i--; j--;
    } else if (dp[i][j] === dp[i - 1][j] + 1) { status[i - 1] = 'missing'; i--; }
    else { j--; }
  }
  return { status, heardAs, distance: dp[n][m] };
}

export class SpeechAssessor {
  constructor({ learner }) { this.learner = learner; }
  // sentence: {text, band, targets:[{ph, words}]}; transcript: string.
  // Returns {score, words:[{word,status,heardAs}], phonemes:{ph:{hit,miss}}}
  assess(sentence, transcript) {
    const tokens = sentence.text.split(/\s+/).filter(t => normalizeWords(t).length);
    const expected = tokens.map(t => normalizeWords(t)[0]);
    const heard = normalizeWords(transcript);
    const { status, heardAs } = alignWords(expected, heard);
    const words = expected.map((w, i) => ({ word: w, display: tokens[i], status: status[i], heardAs: heardAs[i] }));
    const ok = status.filter(s => s === 'ok').length;
    const score = expected.length ? ok / expected.length : 0;
    const phonemes = {};
    for (const t of sentence.targets) {
      const rec = (phonemes[t.ph] ??= { hit: 0, miss: 0 });
      for (const tw of t.words) {
        const found = words.find(x => x.word === tw);
        if (!found) continue;
        found.target = t.ph;
        if (found.status === 'ok') rec.hit++; else rec.miss++;
        this.learner.recordPhoneme(t.ph, found.status === 'ok');
      }
    }
    const difficulty = BANDS[sentence.band] ?? 0;
    this.learner.update('speaking', difficulty, score >= 0.85, 1);
    if (score < 0.85 && score >= 0.6) this.learner.update('speaking', difficulty - 1, true, 0.5);
    return { score, words, phonemes, heard: heard.join(' ') };
  }
}
