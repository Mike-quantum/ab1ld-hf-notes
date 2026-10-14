# Attribution

Everything here builds on other people's work. Each report repeats the credits that apply to it.

| Source | Used for |
|---|---|
| Nils Schiffhauer, DK8OK: *STANAG 4285: Probing the Ionosphere* and the MATLAB app DK8OK_STANAG4285_ESSENTIAL (2026), shared via the Dopplergram group | Method ported to Python and compared (STANAG 4285 report); the single global PN clock with robust refit and fractional-lag reading adopted into our own sounder |
| Con Wassilieff, ZL2AFP: STANAG 4285 Receiver | Origin of DK8OK's work, as credited by DK8OK |
| Peter Martinez, G3PLX | Suggestions credited by DK8OK; G3PLX's CODAR oblique work (Dopplergram) |
| Codex (OpenAI), 2026-10-06 | Independent audit of our code, DK8OK's and the port; the probe-witness (held-out prediction) test idea (report Section 5, code/stanag4285-probe-witness); stress-test scenarios |
| John Seamons (ZL4VO/KF6VO, jks-prv) and contributors: KiwiSDR and kiwiclient (kiwirecorder.py) | All Kiwi recordings and their GPS time stamps; the file format read by code/kiwi-tools |
| NATO STANAG 4285 | Waveform structure (2400 Bd, 256-symbol frame, 80-symbol sync, 3 x 16 probe symbols); the data/probe scrambler (x^9 + x^4 + 1) used for the probe template |
| Nils Schiffhauer, DK8OK, 2026-10-07 (Dopplergram) | Request for the probe-witness code, template and an IQ excerpt; permission to use, change and share the Python port of his app |
| Nils Schiffhauer, DK8OK: MATLAB app DK8OK_STANAG4285_LIVE_PN_V1_4 (2026), Dopplergram group, 2026-10-11 | Ported to Python with defaults unchanged (code/stanag4285-livepn; part-by-part list in its CREDITS.md); his demo signal used as a known-answer test |
| GPT-6 Astra (OpenAI) and Claude Fable (Anthropic), 2026-10-13 | Independent reviews of code/stanag4285-livepn against DK8OK's MATLAB original. Fable measured the window-to-window variation that the comparison tolerances now use, and proposed the end-of-run start-time prior. Fixes made 2026-10-14. |
| Pierre Ynard: rx.linkfanel.net | Public KiwiSDR list and map, referred to for finding Kiwis (code/stanag4285-livepn) |
| Claude (Anthropic) | Analysis code and drafts, working with Mike Burns |
