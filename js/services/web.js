// Web implementations of the service interfaces. Browser APIs live here and nowhere else.

import { Recognizer, Speaker, Store, Clock, Monetization } from './interfaces.js';

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
  async start({ lang = 'en-US', onInterim, onFinal, onError }) {
    if (!this.available) throw new Error('unavailable');
    const rec = new this.SR();
    rec.lang = lang; rec.interimResults = true; rec.continuous = false; rec.maxAlternatives = 1;
    let finalText = '';
    rec.onresult = (e) => {
      let interim = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) finalText += t + ' '; else interim += t;
      }
      onInterim?.((finalText + interim).trim());
    };
    rec.onerror = (e) => {
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed' || e.error === 'audio-capture') this.failed = true;
      onError?.(e.error || 'error');
    };
    rec.onend = () => { this.rec = null; onFinal?.(finalText.trim()); };
    this.rec = rec;
    rec.start();
  }
  stop() { try { this.rec?.stop(); } catch { /* already stopped */ } }
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
  stop() { if (this.available) window.speechSynthesis.cancel(); }
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
