// Service interfaces. Core and UI depend on these, never on concrete classes.
// A native build (Capacitor plugin, Swift, Kotlin) provides its own implementations
// and only Platform.js changes.

export class Recognizer {
  // Returns true if this recognizer can be used in the current environment.
  get available() { return false; }
  get name() { return 'none'; }
  // start({lang, onInterim(text), onFinal(text), onError(err)}); resolves when listening has begun.
  async start(_opts) { throw new Error('not implemented'); }
  stop() {}
  abort() {}
  get listening() { return false; }
}

export class Speaker {
  get available() { return false; }
  // speak(text, {lang, rate}) resolves when finished.
  async speak(_text, _opts) {}
  stop() {}
}

export class Store {
  async get(_key) { return null; }
  async set(_key, _value) {}
  async remove(_key) {}
}

export class Clock {
  now() { return Date.now(); }
}

export class Monetization {
  // Nothing in the UI checks this today. It exists so passes or ads later are one class.
  get kind() { return 'none'; }
  can(_feature) { return true; }
}
