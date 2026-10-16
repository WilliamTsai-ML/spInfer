// Platform factory: the only place that names concrete service classes.
// A native build swaps this file (or the recognizers/speakers it lists) and nothing else.

import { WebSpeechRecognizer, KeyboardRecognizer, WebSpeaker, LocalStore, MemoryStore, SystemClock, NoneMonetization } from './web.js?v=9';

export function createPlatform({ test = false } = {}) {
  const primary = new WebSpeechRecognizer();
  const fallback = new KeyboardRecognizer();
  return {
    // Recognizer chosen at call time so a mic refusal mid-session falls back cleanly.
    get recognizer() { return primary.available ? primary : fallback; },
    speaker: new WebSpeaker(),
    store: test ? new MemoryStore() : new LocalStore(),
    clock: new SystemClock(),
    monetization: new NoneMonetization(),
  };
}
