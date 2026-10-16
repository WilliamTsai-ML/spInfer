// Web implementations of the service interfaces. Browser APIs live here and nowhere else.

import { Recognizer, Speaker, Store, Clock, Monetization } from './interfaces.js?v=8';

export class WebSpeechRecognizer extends Recognizer {
  constructor() {
    super();
    const SR = typeof window !== 'undefined' && (window.SpeechRecognition || window.webkitSpeechRecognition);
    this.SR = SR || null;
    this.rec = null;
    this.failed = false;
  }
  get available() { return !!this.SR && !this.failed; }
  get name() { return 'web-speech'; }
  get listening() { return !!this.rec; }
  // Keep a microphone stream open across sessions. On iOS this keeps the audio session
  // alive so WebKit stops aborting every other recognition. Released if it ever causes trouble.
  async holdMic(onEvent) {
    if (this.stream || this.noHold) return;
    try {
      if (!navigator.mediaDevices?.getUserMedia) { this.noHold = true; return; }
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      onEvent?.('hold');
    } catch { this.noHold = true; onEvent?.('nohold'); }
  }
  releaseMic() {
    try { this.stream?.getTracks().forEach(t => t.stop()); } catch { /* ignore */ }
    this.stream = null;
  }
  // iOS Safari sometimes never fires onend for a second session, and often delivers
  // only interim results. So: abort any previous session, keep the best text seen so far,
  // and finish on a watchdog if the browser never says it ended.
  async start({ lang = 'en-US', onInterim, onFinal, onError, onEvent }) {
    if (!this.available) throw new Error('unavailable');
    this.abort();
    await this.holdMic(onEvent);
    // iOS Safari aborts a session that starts within ~1 s of the previous one ending.
    const gap = 1000 - (Date.now() - (this.endedAt || 0));
    if (gap > 0) { onEvent?.('wait'); await new Promise(r => setTimeout(r, gap)); }
    const rec = new this.SR();
    rec.lang = lang; rec.interimResults = true; rec.continuous = false; rec.maxAlternatives = 1;
    let finalText = '', lastText = '', done = false;
    const finish = (why) => {
      if (done) return; done = true;
      this.endedAt = Date.now();
      clearTimeout(this.watchdog); clearTimeout(this.maxTimer);
      if (this.rec === rec) this.rec = null;
      onEvent?.('finish:' + why);
      onFinal?.((finalText.trim() || lastText).trim());
    };
    rec.onstart = () => onEvent?.('start');
    rec.onaudiostart = () => onEvent?.('audio');
    rec.onspeechend = () => onEvent?.('speechend');
    rec.onresult = (e) => {
      this.everWorked = true;
      let interim = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) finalText += t + ' '; else interim += t;
      }
      lastText = (finalText + interim).trim() || lastText;
      onEvent?.('result');
      onInterim?.(lastText);
    };
    rec.onerror = (e) => {
      onEvent?.('error:' + e.error);
      // Only give up on the mic if it has never worked on this page. iOS Safari reports
      // 'not-allowed' or 'aborted' for later sessions even when the mic was granted.
      if (!this.everWorked && (e.error === 'not-allowed' || e.error === 'service-not-allowed' || e.error === 'audio-capture')) this.failed = true;
      if (lastText) return; // we already have text; let onend deliver it
      if (this.rec !== rec) return; // we aborted it ourselves
      // If holding the mic stream is what upsets WebKit, let it go and let the caller retry without it.
      if (this.stream && (e.error === 'audio-capture' || e.error === 'aborted')) { this.holdFailures = (this.holdFailures || 0) + 1; if (this.holdFailures >= 2) { this.releaseMic(); this.noHold = true; onEvent?.('release'); } }
      onError?.(e.error || 'error');
      if (e.error !== 'no-speech') finish('error');
    };
    rec.onend = () => { if (lastText) this.everWorked = true; this.endedAt = Date.now(); finish('end'); };
    this.rec = rec;
    // Hard cap so a stuck session cannot hang the screen.
    this.maxTimer = setTimeout(() => { try { rec.stop(); } catch {} ; setTimeout(() => finish('timeout'), 800); }, 15000);
    this.finishNow = finish;
    try { rec.start(); } catch (err) { finish('start-failed'); throw err; }
  }
  stop() {
    const rec = this.rec; if (!rec) return;
    try { rec.stop(); } catch { /* already stopped */ }
    // If iOS never reports the end, close it ourselves.
    clearTimeout(this.watchdog);
    this.watchdog = setTimeout(() => this.finishNow?.('watchdog'), 1500);
  }
  abort() {
    const rec = this.rec; if (!rec) return;
    this.rec = null;
    rec.onend = rec.onerror = rec.onresult = null;
    clearTimeout(this.watchdog); clearTimeout(this.maxTimer);
    try { rec.abort(); } catch { /* ignore */ }
  }
}

// Fallback that always works: the learner uses the keyboard's own dictation button.
// On iOS the keyboard mic is Apple's recognizer, so this is still real speech input.
export class KeyboardRecognizer extends Recognizer {
  get available() { return true; }
  get name() { return 'keyboard'; }
  async start() { /* the UI shows a text field; nothing to start */ }
}

export class WebSpeaker extends Speaker {
  get available() { return typeof window !== 'undefined' && 'speechSynthesis' in window; }
  pickVoice(lang) {
    const voices = window.speechSynthesis.getVoices().filter(v => v.lang && v.lang.toLowerCase().startsWith(lang.slice(0, 2)));
    const pref = ['Samantha', 'Ava', 'Allison', 'Daniel', 'Karen', 'Google US English'];
    return voices.find(v => pref.includes(v.name)) || voices.find(v => v.lang.toLowerCase() === lang.toLowerCase()) || voices[0] || null;
  }
  speak(text, { lang = 'en-US', rate = 0.92 } = {}) {
    if (!this.available) return Promise.resolve();
    return new Promise((resolve) => {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = lang; u.rate = rate;
      const v = this.pickVoice(lang); if (v) u.voice = v;
      u.onend = resolve; u.onerror = resolve;
      window.speechSynthesis.speak(u);
    });
  }
  stop() { if (this.available && (window.speechSynthesis.speaking || window.speechSynthesis.pending)) window.speechSynthesis.cancel(); }
}

export class LocalStore extends Store {
  constructor(prefix = 'lantern:') { super(); this.prefix = prefix; this.mem = new Map(); }
  async get(key) {
    try { const raw = localStorage.getItem(this.prefix + key); return raw ? JSON.parse(raw) : null; }
    catch { return this.mem.get(key) ?? null; }
  }
  async set(key, value) {
    this.mem.set(key, value);
    try { localStorage.setItem(this.prefix + key, JSON.stringify(value)); } catch { /* private mode etc. */ }
  }
  async remove(key) {
    this.mem.delete(key);
    try { localStorage.removeItem(this.prefix + key); } catch { /* ignore */ }
  }
}

export class MemoryStore extends Store {
  constructor() { super(); this.mem = new Map(); }
  async get(k) { return this.mem.get(k) ?? null; }
  async set(k, v) { this.mem.set(k, v); }
  async remove(k) { this.mem.delete(k); }
}

export class SystemClock extends Clock {}
export class NoneMonetization extends Monetization {}
