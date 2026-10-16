# Lantern English (prototype)

Offline English learning for Mandarin speakers. Web prototype; the iOS app will wrap the same code.

- `index.html` – the page (open from any static server, needs https or localhost for the microphone)
- `js/core/` – pure logic: learner model, adaptive placement, speech assessment
- `js/services/` – interfaces plus web implementations; `Platform.js` is the only file that names concrete classes
- `js/ui/` – screens
- `js/data/` – language packs (words, sentences, phoneme notes, UI strings)
- `tests/` – `node tests/core.test.mjs`

No server, no account, no tracking.
