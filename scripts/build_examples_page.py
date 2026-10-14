"""Build examples/index.html (GitHub Pages) from examples/bd-publico/*/metadata.json.

Usage, from the repo root:  python scripts/build_examples_page.py .
Edit ORDER below to change which examples appear and in what order.
"""
import difflib
import html
import json
import re
import sys
from pathlib import Path

repo = Path(sys.argv[1])
ex_dir = repo / "examples" / "bd-publico"

# Order and titles as in examples/bd-publico/README.md
ORDER = [
    ("b00107-u08744", "Person name"),
    ("b00108-u08816", "Locations (city and country)"),
    ("b00105-u08628", "Organizations and a city"),
]
LABELS = {
    "01": "Original recording",
    "02": "F5-TTS in-place editing <span class=\"pill\">default</span>",
    "03": "F5-TTS zero-shot infilling",
    "04": "Google TTS infilling",
    "05": "Voice conversion only (kNN-VC)",
    "06": "Content + voice (F5-TTS editing + kNN-VC)",
    "07": "Noisy input",
    "08": "Anonymized, no separation",
    "09": "Anonymized, background removed",
    "10": "Anonymized, background kept",
    "11": "Content + voice, background kept",
}
esc = html.escape


def words(t):
    return re.findall(r"\S+", t)


def norm(w):
    return re.sub(r"[^\w]", "", w.lower())


def diff_pair(orig, anon):
    """Return (orig_html, anon_html) with the changed words highlighted."""
    a, b = words(orig), words(anon)
    sm = difflib.SequenceMatcher(a=[norm(w) for w in a], b=[norm(w) for w in b], autojunk=False)
    oa, ob = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        # Only highlight entity replacements (all generalizations here: "... anónimo/a").
        # Other differences are spelling/punctuation normalization, not anonymization.
        if op != "equal" and "anónim" in " ".join(b[j1:j2]).lower():
            # leading words that only differ in spelling (arquitecto/arquiteto) stay plain
            while (i1 < i2 and j1 < j2 and
                   difflib.SequenceMatcher(a=norm(a[i1]), b=norm(b[j1])).ratio() >= 0.8):
                oa.append(esc(a[i1])); ob.append(esc(b[j1])); i1 += 1; j1 += 1
            sa, sb = esc(" ".join(a[i1:i2])), esc(" ".join(b[j1:j2]))
            if sa: oa.append(f"<del>{sa}</del>")
            if sb: ob.append(f"<ins>{sb}</ins>")
        else:
            sa, sb = esc(" ".join(a[i1:i2])), esc(" ".join(b[j1:j2]))
            if sa: oa.append(sa)
            if sb: ob.append(sb)
    return " ".join(oa), " ".join(ob)


def clip_row(doc_id, c):
    num = c["file"][:2]
    wer = c.get("wer_vs_intended_anon")
    wer_html = f"{wer * 100:.1f}%" if wer is not None else '<span class="na">—</span>'
    return f"""
          <tr>
            <td class="n">{int(num)}</td>
            <td class="sys">{LABELS[num]}</td>
            <td class="aud"><audio controls preload="none" src="bd-publico/{doc_id}/{esc(c['file'])}"></audio></td>
            <td class="asr">{esc(c.get('asr_hypothesis') or '')}</td>
            <td class="w">{wer_html}</td>
          </tr>"""


def clip_table(doc_id, clips):
    rows = "".join(clip_row(doc_id, c) for c in clips)
    return f"""
      <table>
        <thead><tr><th class="n">#</th><th>System</th><th>Audio</th><th>What Whisper hears</th><th class="w">WER</th></tr></thead>
        <tbody>{rows}
        </tbody>
      </table>"""


sections = []
for k, (doc_id, title) in enumerate(ORDER, 1):
    m = json.loads((ex_dir / doc_id / "metadata.json").read_text(encoding="utf-8"))
    clean = [c for c in m["clips"] if int(c["file"][:2]) <= 6]
    noisy = [c for c in m["clips"] if int(c["file"][:2]) >= 7]
    o_html, a_html = diff_pair(m["original_text"], m["intended_anonymized_text"])

    noisy_intended = {c["intended_anonymized_text"] for c in noisy if c.get("intended_anonymized_text")}
    noisy_note = ""
    same_words = lambda x, y: [norm(w) for w in words(x)] == [norm(w) for w in words(y)]
    if noisy_intended and not all(same_words(t, m["intended_anonymized_text"]) for t in noisy_intended):
        _, n_html = diff_pair(m["original_text"], noisy_intended.pop())
        noisy_note = (f'<p class="note">The noisy input is transcribed slightly differently, so '
                      f'detection and replacements differ too. Intended text for these clips:</p>'
                      f'<p class="quote">{n_html}</p>')
    noise = m.get("noise") or {}
    noise_desc = (f'MUSAN noise added at {noise.get("snr_db", 0):g}&nbsp;dB SNR. All anonymized clips use '
                  f'<strong>F5-TTS in-place editing</strong>, the default backend.') if noise else ""

    sections.append(f"""
  <section id="{doc_id}">
    <div class="ex-head">
      <span class="ex-num">{k}</span>
      <h2>{esc(title)}</h2>
    </div>
    <div class="texts">
      <div><div class="label">Original</div><p>{o_html}</p></div>
      <div><div class="label">Anonymized</div><p>{a_html}</p></div>
    </div>
    <h3>Clean recording</h3>{clip_table(doc_id, clean)}
    <h3>With background noise</h3>
    <p class="note">{noise_desc}</p>{noisy_note}{clip_table(doc_id, noisy)}
  </section>""")

nav = "".join(f'<li><a href="#{d}">{esc(t)}</a></li>' for d, t in ORDER)

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anonymization Examples</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  /* Font and colors follow the web app (static/index.html). */
  *, *::before, *::after {{ box-sizing: border-box; }}
  :root {{
    color-scheme: light;
    --primary: #1e40af; --primary-dark: #1e3a8a; --primary-light: #dbeafe;
    --bg: #ffffff; --panel: #f8fafc; --border: #e2e8f0; --muted: #64748b; --text: #0f172a;
  }}
  body {{ margin: 0; background: var(--bg); color: var(--text);
         font: 16px/1.6 'Inter', system-ui, -apple-system, sans-serif; }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 48px 16px 72px; }}
  a {{ color: var(--primary); text-underline-offset: 2px; }}
  h1 {{ font-size: 2rem; font-weight: 700; letter-spacing: -.02em; margin: 0 0 .3rem; }}
  .sub {{ color: var(--muted); margin: 0 0 1.2rem; }}
  .lead {{ margin: 0 0 1rem; }}
  .toc-title {{ font-weight: 600; margin: 0 0 .3rem; }}
  .toc {{ display: flex; flex-wrap: wrap; gap: .3rem 1.8rem; padding-left: 1.2rem; margin: 0; }}
  .toc li::marker {{ color: var(--muted); }}
  .toc a {{ font-weight: 500; }}

  section {{ border-top: 1px solid var(--border); margin-top: 2.8rem; padding-top: 2rem; }}
  .ex-head {{ display: flex; gap: .9rem; align-items: baseline; margin-bottom: 1.1rem; }}
  .ex-num {{ font-size: 2rem; font-weight: 700; line-height: 1; color: var(--primary-dark); }}
  h2 {{ font-size: 1.35rem; font-weight: 700; letter-spacing: -.01em; margin: 0; }}
  .src {{ margin: .1rem 0 0; color: var(--muted); font-size: .85rem; }}

  .texts {{ display: grid; grid-template-columns: 1fr 1fr; background: var(--panel);
           border: 1px solid var(--border); border-radius: .6rem; }}
  .texts > div {{ padding: .9rem 1.1rem; }}
  .texts > div + div {{ border-left: 1px solid var(--border); }}
  .label {{ font-size: .8rem; font-weight: 600; color: var(--muted); margin-bottom: .25rem; }}
  .texts p {{ margin: 0; }}
  del {{ color: var(--muted); text-decoration-color: #94a3b8; }}
  ins {{ text-decoration: none; font-weight: 600; color: var(--primary-dark);
         background: var(--primary-light); border-radius: .25rem; padding: 0 .25rem; }}

  h3 {{ font-size: 1.05rem; font-weight: 600; margin: 1.8rem 0 .4rem; }}
  .note {{ color: var(--muted); font-size: .9rem; margin: 0 0 .6rem; }}
  .quote {{ margin: 0 0 .9rem; padding-left: .9rem; border-left: 2px solid var(--border); }}
  code {{ font-size: .85em; }}

  table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
  th {{ text-align: left; font-weight: 600; font-size: .82rem; color: var(--muted);
        border-bottom: 1.5px solid var(--text); padding: .4rem .6rem; }}
  td {{ border-bottom: 1px solid var(--border); padding: .65rem .6rem; vertical-align: middle; }}
  tbody tr:hover {{ background: var(--panel); }}
  .n {{ width: 2rem; color: var(--muted); }}
  .sys {{ font-weight: 600; width: 23%; }}
  .pill {{ display: inline-block; font-size: .7rem; font-weight: 600; color: var(--primary);
          background: var(--primary-light); border-radius: 999px; padding: 0 .45rem; margin-left: .2rem; vertical-align: 1px; }}
  .aud {{ width: 260px; }}
  audio {{ width: 250px; height: 34px; display: block; }}
  .asr {{ color: var(--muted); font-size: .85rem; line-height: 1.45; }}
  .w {{ text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
  .na {{ color: #cbd5e1; }}

  @media (max-width: 760px) {{
    h1 {{ font-size: 1.6rem; }}
    .texts {{ grid-template-columns: 1fr; }}
    .texts > div + div {{ border-left: 0; border-top: 1px solid var(--border); }}
    thead {{ display: none; }}
    table, tbody, tr, td {{ display: block; width: 100%; }}
    tr {{ display: grid; grid-template-columns: 1.6rem minmax(0, 1fr) auto; gap: .35rem .5rem;
          grid-template-areas: "n sys w" ". aud aud" ". asr asr";
          border-bottom: 1px solid var(--border); padding: .8rem 0; }}
    td, td.sys, td.aud, td.n {{ border: 0; padding: 0; width: auto; }}
    td.n {{ grid-area: n; }} td.sys {{ grid-area: sys; }} td.w {{ grid-area: w; }}
    td.aud {{ grid-area: aud; }} td.asr {{ grid-area: asr; }}
    audio {{ width: 100%; }}
  }}
</style>
</head>
<body>
<main>
  <h1>Listening examples</h1>
  <p class="sub"><a href="https://github.com/cati-xf/anonymization-platform">GitHub repository</a></p>
  <p class="lead">Three utterances from the BD-PÚBLICO read-speech test set, each from a different
  speaker, taken from the thesis evaluation. Every entity is generalized: removed words are
  <del>struck through</del> and replacements are <ins>highlighted</ins>. Each clip shows what the
  Whisper model hears in it, and the word error rate (WER) against the intended
  anonymized text. Voice-conversion-only clips keep the original words, so they have no WER.</p>
  <p class="toc-title">The three examples:</p>
  <ol class="toc">{nav}</ol>
  {''.join(sections)}
</main>
</body>
</html>
"""
(repo / "examples" / "index.html").write_text(page, encoding="utf-8")
print("wrote", repo / "examples" / "index.html")
