// UI layer. Talks to core and to the service interfaces, never to browser APIs directly
// (DOM aside). Screens are small classes with a mount(root) method.

import { LearnerModel, SKILLS } from '../core/LearnerModel.js?v=8';
import { VocabPlacement } from '../core/VocabPlacement.js?v=8';
import { SpeechAssessor } from '../core/SpeechAssessor.js?v=8';
import { WordDrill } from '../core/WordDrill.js?v=8';
import { VOCAB, PSEUDOWORDS, SPEAKING, MINIMAL_PAIRS, PHONEME_NOTES, STRINGS } from '../data/pack-en-zh.js?v=8';

const h = (tag, attrs = {}, ...children) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c != null) el.append(c.nodeType ? c : String(c));
  return el;
};

export class App {
  constructor({ platform, root }) {
    this.platform = platform;
    this.root = root;
    this.lang = 'en';
    this.learner = new LearnerModel();
    this.placement = null;
    this.screen = null;
  }
  t(key) { return STRINGS[this.lang][key] ?? STRINGS.en[key] ?? key; }
  async boot() {
    const saved = await this.platform.store.get('profile');
    if (saved) { this.learner = new LearnerModel(saved.learner); this.lang = saved.lang ?? 'en'; }
    document.documentElement.lang = this.lang === 'zh' ? 'zh-Hant' : 'en';
    this.show(this.learner.overall() == null ? 'home' : 'profile');
  }
  async save() {
    this.learner.updatedAt = this.platform.clock.now();
    await this.platform.store.set('profile', { learner: this.learner.toJSON(), lang: this.lang });
  }
  async reset() {
    await this.platform.store.remove('profile');
    this.learner = new LearnerModel(); this.placement = null;
    this.show('home');
  }
  toggleLang() { this.lang = this.lang === 'en' ? 'zh' : 'en'; document.documentElement.lang = this.lang === 'zh' ? 'zh-Hant' : 'en'; this.save(); this.show(this.screen); }
  show(name) {
    this.screen = name;
    const Screen = { home: HomeScreen, vocab: VocabScreen, speak: SpeakScreen, words: WordScreen, profile: ProfileScreen }[name];
    this.root.replaceChildren();
    this.root.append(this.topBar());
    const body = h('main', { class: 'screen', 'data-screen': name });
    this.root.append(body);
    new Screen(this).mount(body);
    this.root.append(this.modelStrip());
  }
  topBar() {
    return h('header', { class: 'top' },
      h('div', { class: 'brand' }, h('span', { class: 'lamp', 'aria-hidden': 'true' }), this.t('appName')),
      h('button', { class: 'ghost', id: 'langToggle', onclick: () => this.toggleLang() }, this.t('language')));
  }
  // Tiny always-on readout of the learner model: the "continuous adjustment" made visible.
  modelStrip() {
    const cells = SKILLS.map(s => {
      const e = this.learner.skill(s);
      const txt = e.tested ? `${e.cefr} · θ ${e.theta.toFixed(2)} ±${isFinite(e.se) ? e.se.toFixed(2) : '∞'}` : '—';
      return h('div', { class: 'cell' }, h('span', { class: 'k' }, this.t(s)), h('span', { class: 'v' }, txt));
    });
    return h('footer', { class: 'model', 'aria-label': this.t('debug') }, cells);
  }
  refreshStrip() { const old = this.root.querySelector('footer.model'); if (old) old.replaceWith(this.modelStrip()); }
}

class HomeScreen {
  constructor(app) { this.app = app; }
  mount(root) {
    const a = this.app;
    root.append(
      h('section', { class: 'hero' },
        h('h1', {}, a.t('appName')),
        h('p', { class: 'lede' }, a.t('tagline')),
        h('p', { class: 'muted' }, a.lang === 'zh' ? '約 5 分鐘：字彙 2 分鐘，口說 2 分鐘。' : 'About 5 minutes: vocabulary, then speaking.'),
        h('div', { class: 'row' },
          h('button', { class: 'primary', id: 'start', onclick: () => a.show('vocab') }, a.t('start')),
          h('button', { class: 'secondary', id: 'speakOnly', onclick: () => a.show('speak') }, a.t('speakOnly'))),
        h('button', { class: 'secondary', id: 'wordsOnly', onclick: () => a.show('words') }, a.t('wordsOnly'))));
  }
}

class VocabScreen {
  constructor(app) { this.app = app; }
  mount(root) {
    const a = this.app;
    if (!a.placement) a.placement = new VocabPlacement({ vocab: VOCAB, pseudowords: PSEUDOWORDS, learner: a.learner, seed: (Date.now() % 100000) | 1 });
    this.p = a.placement;
    this.bar = h('div', { class: 'bar' }, h('i'));
    this.word = h('div', { class: 'word', id: 'word' });
    this.count = h('span', { class: 'muted small' });
    root.append(
      h('div', { class: 'progress' }, this.bar, this.count),
      h('h2', {}, a.t('vocabTitle')),
      h('p', { class: 'muted' }, a.t('vocabHint')),
      h('div', { class: 'card big' }, this.word),
      h('div', { class: 'row' },
        h('button', { class: 'secondary', id: 'no', onclick: () => this.answer(false) }, a.t('dontKnow')),
        h('button', { class: 'primary', id: 'yes', onclick: () => this.answer(true) }, a.t('know'))));
    this.ask();
  }
  ask() {
    const item = this.p.next();
    if (!item) return this.finish();
    this.word.textContent = item.word;
    this.word.classList.remove('pop'); void this.word.offsetWidth; this.word.classList.add('pop');
    this.bar.firstChild.style.width = `${Math.round(this.p.progress * 100)}%`;
    this.count.textContent = `${this.p.asked + 1}`;
  }
  answer(known) {
    this.p.answer(known);
    this.app.refreshStrip();
    this.ask();
  }
  async finish() {
    this.p.result();
    await this.app.save();
    this.app.placement = null;
    this.app.show('speak');
  }
}

// MicPanel: the recording control shared by every speaking screen.
// Owns the Record button, status line, event trace and the typed fallback box.
class MicPanel {
  constructor(app, { onText, placeholder }) {
    this.app = app; this.onText = onText;
    this.status = h('p', { class: 'muted small', id: 'status' });
    this.trace = h('p', { class: 'muted small trace', id: 'trace' });
    this.heard = h('div', { class: 'heard', id: 'heard' });
    this.input = h('textarea', { id: 'transcript', rows: 1, placeholder: placeholder ?? app.t('typeFallback'), autocapitalize: 'off', autocorrect: 'off' });
    this.recBtn = h('button', { class: 'primary mic', id: 'rec', onclick: () => this.toggleRecord() }, app.t('record'));
    this.input.addEventListener('input', () => { clearTimeout(this.typeTimer); if (this.input.value.trim()) this.typeTimer = setTimeout(() => this.deliver(this.input.value), 600); });
  }
  get elements() { return [this.status, this.input, this.heard, this.trace]; }
  reset() {
    const a = this.app, rec = a.platform.recognizer;
    this.heard.textContent = ''; this.input.value = ''; this.trace.textContent = ''; this.events = []; this.retries = 0;
    rec.abort(); this.stopUi();
    if (rec.name === 'keyboard') { this.recBtn.hidden = true; this.status.textContent = a.t('micDenied'); }
    else { this.recBtn.hidden = false; this.status.textContent = ''; }
  }
  deliver(text) { this.heard.textContent = this.app.t('heard') + ' ' + (text || '—'); this.onText(text); }
  async toggleRecord() {
    const a = this.app; const rec = a.platform.recognizer;
    if (this.recording) { rec.stop(); this.recBtn.textContent = '…'; return; }
    a.platform.speaker.stop(); // free the audio session before recording
    this.recording = true; this.recBtn.textContent = '■ ' + a.t('stop'); this.recBtn.classList.add('live');
    this.status.textContent = a.t('listening_'); this.heard.textContent = '';
    const seq = (this.seq = (this.seq || 0) + 1);
    try {
      await rec.start({
        lang: 'en-US',
        onEvent: (e) => { this.events.push(e); this.trace.textContent = this.events.join(' · '); },
        onInterim: (t) => { if (seq === this.seq) this.heard.textContent = a.t('heard') + ' ' + t; },
        onFinal: (t) => { if (seq !== this.seq) return; this.stopUi(); if (t) { this.retries = 0; this.deliver(t); } else if (!this.status.textContent) this.status.textContent = a.t('nothingHeard'); },
        onError: (err) => {
          if (seq !== this.seq) return;
          this.stopUi();
          if (!rec.available) { this.reset(); return; }
          if (err === 'no-speech') { this.status.textContent = a.t('nothingHeard'); return; }
          // iOS Safari: a later session can fail at once with not-allowed/aborted. One automatic retry.
          this.retries = (this.retries || 0) + 1;
          if (this.retries <= 3) { this.status.textContent = `${a.t('retrying')} ${this.retries}/3`; setTimeout(() => this.toggleRecord(), 300); return; }
          this.status.textContent = `${a.t('micError')} (${err}). ${a.t('tapAgain')}`;
        },
      });
    } catch (e) { this.stopUi(); this.status.textContent = `${a.t('micError')} (${e?.name || e}). ${a.t('tapAgain')}`; }
  }
  stopUi() { this.recording = false; this.recBtn.textContent = this.app.t('record'); this.recBtn.classList.remove('live'); if (this.status.textContent === this.app.t('listening_')) this.status.textContent = ''; }
}

class SpeakScreen {
  constructor(app) { this.app = app; this.i = 0; this.assessor = new SpeechAssessor({ learner: app.learner }); this.results = []; }
  // Four sentences per probe, spread over the sound targets; the rest stay for later sessions.
  get set() { return SPEAKING.slice(0, 4); }
  mount(root) {
    const a = this.app;
    this.bar = h('div', { class: 'bar' }, h('i'));
    this.sentence = h('div', { class: 'sentence', id: 'sentence' });
    this.mic = new MicPanel(a, { onText: (t) => this.apply(t) });
    this.nextBtn = h('button', { class: 'primary', id: 'next', disabled: true, onclick: () => this.next() }, a.t('next'));
    root.append(
      h('div', { class: 'progress' }, this.bar),
      h('h2', {}, a.t('speakTitle')),
      h('p', { class: 'muted' }, a.t('speakHint')),
      h('div', { class: 'card' }, this.sentence,
        h('div', { class: 'row left' },
          h('button', { class: 'ghost', id: 'listen', onclick: () => a.platform.speaker.speak(this.cur.text) }, '▶ ' + a.t('listen')))),
      h('div', { class: 'row' }, this.mic.recBtn, this.nextBtn),
      ...this.mic.elements);
    this.showSentence();
  }
  get cur() { return this.set[this.i]; }
  showSentence() {
    this.sentence.replaceChildren(...this.cur.text.split(' ').map(w => h('span', { class: 'w' }, w + ' ')));
    this.nextBtn.disabled = true; this.assessed = null;
    this.bar.firstChild.style.width = `${Math.round((this.i / this.set.length) * 100)}%`;
    this.mic.reset();
  }
  apply(transcript) {
    const r = this.assessor.assess(this.cur, transcript);
    this.assessed = r; this.results.push({ id: this.cur.id, ...r });
    this.sentence.replaceChildren(...r.words.map(x => h('span', { class: 'w ' + x.status, title: x.heardAs || '' }, x.display + ' ')));
    this.nextBtn.disabled = false;
    this.app.refreshStrip();
  }
  async next() {
    this.i++;
    await this.app.save();
    if (this.i >= this.set.length) return this.app.show('profile');
    this.showSentence();
  }
}

class WordScreen {
  constructor(app) { this.app = app; this.drill = new WordDrill({ pairs: MINIMAL_PAIRS, learner: app.learner, seed: (Date.now() % 100000) | 1 }); }
  mount(root) {
    const a = this.app;
    this.bar = h('div', { class: 'bar' }, h('i'));
    this.word = h('div', { class: 'word', id: 'word' });
    this.verdict = h('div', { class: 'verdict', id: 'verdict' });
    this.mic = new MicPanel(a, { onText: (t) => this.apply(t), placeholder: a.t('typeFallback') });
    this.nextBtn = h('button', { class: 'primary', id: 'next', disabled: true, onclick: () => this.next() }, a.t('next'));
    root.append(
      h('div', { class: 'progress' }, this.bar),
      h('h2', {}, a.t('wordTitle')),
      h('p', { class: 'muted' }, a.t('wordHint')),
      h('div', { class: 'card big' }, this.word, this.verdict,
        h('button', { class: 'ghost', id: 'listen', onclick: () => a.platform.speaker.speak(this.drill.current.word) }, '▶ ' + a.t('listen'))),
      h('div', { class: 'row' }, this.mic.recBtn, this.nextBtn),
      ...this.mic.elements);
    this.showWord();
  }
  showWord() {
    const item = this.drill.current;
    this.word.textContent = item.word; this.word.className = 'word';
    this.verdict.replaceChildren();
    this.nextBtn.disabled = true;
    this.bar.firstChild.style.width = `${Math.round(this.drill.progress * 100)}%`;
    this.mic.reset();
  }
  apply(transcript) {
    const a = this.app;
    const r = this.drill.judge(transcript); if (!r) return;
    const note = PHONEME_NOTES[r.item.ph]?.[a.lang] ?? r.item.ph;
    if (r.verdict === 'hit') { this.word.className = 'word ok'; this.verdict.replaceChildren(h('span', { class: 'good' }, '✓ ' + a.t('good'))); }
    else if (r.verdict === 'foil') { this.word.className = 'word bad'; this.verdict.replaceChildren(h('span', { class: 'bad' }, `${a.t('soundsLike')} “${r.item.foil}”`), h('span', { class: 'muted small' }, note)); }
    else { this.word.className = 'word'; this.verdict.replaceChildren(h('span', { class: 'muted' }, a.t('unclear'))); }
    this.nextBtn.disabled = r.verdict === 'other' && !r.heard;
    a.refreshStrip();
  }
  async next() {
    this.drill.next();
    await this.app.save();
    if (this.drill.finished) return this.app.show('profile');
    this.showWord();
  }
}

class ProfileScreen {
  constructor(app) { this.app = app; }
  mount(root) {
    const a = this.app, L = a.learner;
    const rows = SKILLS.map(s => {
      const e = L.skill(s);
      const pos = e.tested ? Math.round(((e.theta + 3) / 6) * 100) : null;
      return h('div', { class: 'skill' },
        h('div', { class: 'skill-head' }, h('span', {}, a.t(s)), h('strong', { class: 'level' }, e.tested ? e.cefr : a.t('notTested'))),
        h('div', { class: 'scale' },
          ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'].map(b => h('span', {}, b)),
          pos != null ? h('i', { style: `left:${pos}%` }) : null),
        e.tested ? h('div', { class: 'muted small' }, `${e.n} ${a.t('itemsAnswered')} · ${a.t('confidence')} ±${isFinite(e.se) ? e.se.toFixed(2) : '∞'}`) : null);
    });
    const weak = L.weakPhonemes(1);
    root.append(...[
      h('h2', {}, a.t('profileTitle')),
      L.flags.vocabGuessing ? h('p', { class: 'note' }, a.t('guessing')) : null,
      h('div', { class: 'skills' }, rows),
      h('h3', {}, a.t('weakSounds')),
      weak.length
        ? h('ul', { class: 'sounds' }, weak.map(w => h('li', {}, h('b', {}, '/' + w.ph + '/'), ' ', PHONEME_NOTES[w.ph]?.[a.lang] ?? '', h('span', { class: 'muted small' }, ` ${w.miss}/${w.hit + w.miss}`))))
        : h('p', { class: 'muted' }, a.t('noWeak')),
      h('p', { class: 'muted small' }, a.t('nextStep')),
      h('div', { class: 'row' },
        h('button', { class: 'secondary', id: 'reset', onclick: () => a.reset() }, a.t('startOver')),
        h('button', { class: 'secondary', id: 'words', onclick: () => a.show('words') }, a.t('words') + ' ↻'),
        h('button', { class: 'primary', id: 'again', onclick: () => a.show('speak') }, a.t('speaking') + ' ↻'))].filter(Boolean));
  }
}
