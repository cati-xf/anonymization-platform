# Speech Anonymization Platform (European Portuguese)

A web platform that anonymizes European Portuguese text and speech. It transcribes audio,
detects sensitive information (names, places, organizations, phone numbers, IDs, ...), lets
you review and edit what was found, and then re-synthesizes the audio with the replacements.
It can keep the original speaker's voice, prosody and background noise.

Developed for the master's thesis *A Unified Platform for Multi-Modal Data Anonymization:
Privacy- and utility-preserving text and speech anonymization for European Portuguese*
(Instituto Superior Técnico, Universidade de Lisboa, 2026), supervised by Prof. Luísa Coheur
and Dr. Francisco Teixeira.

> [!IMPORTANT]
> **Before you start**
> - **Linux with conda and an NVIDIA GPU** for audio (developed on an RTX 6000 Ada,
>   48 GB). The Text tab runs on CPU.
> - **The F5-TTS model for European Portuguese is not public**: request it from INESC-ID
>   HLT (step 4). All audio options except kNN-VC voice conversion need it.
> - **kNN-VC voices are not in the repository**, so the two "Preserve natural speech
>   rhythm" options don't work until you add them (step 6).
> - **Internet** is only needed for the install, the first audio job (it downloads the
>   DiariZen and kNN-VC models), the optional Google TTS alternative, and live Wikidata
>   lookups, which are opt-in per term.

## Installation

Needs `git` and [Miniconda](https://github.com/conda-forge/miniforge). Tested with Python 3.13,
PyTorch 2.8.0 and CUDA 12.8 ([full snapshot](envs/SNAPSHOT.txt)).

### 1. Main environment

```bash
git clone https://github.com/cati-xf/anonymization-platform.git anon-platform
cd anon-platform
export ANON_DIR="$PWD"                            # used by the later steps
python3 -m venv venv
source venv/bin/activate
pip install --no-deps -r requirements.lock.txt    # exact versions used (incl. the spaCy model)
python -c "import fastapi, whisperx, spacy, torch; print('ok', torch.cuda.is_available())"
```

The last line should print `ok True`. This is enough for the Text tab.
(`pip check` will flag `gtts` wanting an older `click`: that's expected and harmless.)

### 2. Speech recognition model (Whisper CAMÕES)

[`inesc-id/WhisperLv3-FT-EP-CPP`](https://huggingface.co/inesc-id/WhisperLv3-FT-EP-CPP) has to be
converted for the app, and merged with the base Whisper tokenizer first because it ships
without one. Run this in the main venv, from the repo folder; the result goes in
`models/whisper_lv3_ft_ep_cpp_ct2`:

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
from transformers import WhisperTokenizer, WhisperFeatureExtractor
import os, shutil

src = snapshot_download("inesc-id/WhisperLv3-FT-EP-CPP")
dst = "models/whisper_merged"
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if os.path.isfile(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), dst)
WhisperTokenizer.from_pretrained("openai/whisper-large-v3").save_pretrained(dst)
WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3").save_pretrained(dst)
EOF

ct2-transformers-converter --model models/whisper_merged \
    --output_dir models/whisper_lv3_ft_ep_cpp_ct2 --quantization float16 --force
python -c "from transformers import WhisperTokenizer; \
WhisperTokenizer.from_pretrained('openai/whisper-large-v3').save_pretrained('models/whisper_lv3_ft_ep_cpp_ct2')"
rm -rf models/whisper_merged
```

Check: `models/whisper_lv3_ft_ep_cpp_ct2/vocabulary.json` should contain `<|pt|>` and
`<|transcribe|>`.

### 3. Forced aligner (MFA)

```bash
deactivate 2>/dev/null                             # leave the main venv first
conda env create -f "$ANON_DIR/envs/aligner.yml"   # env "aligner", MFA 3.3.9
conda run -n aligner mfa model download acoustic portuguese_mfa
conda run -n aligner mfa model download dictionary portuguese_mfa
conda run -n aligner mfa version                   # should print 3.3.9
```

### 4. Speech synthesis (F5-TTS, European Portuguese)

The F5-TTS model fine-tuned for European Portuguese at INESC-ID is **not public**. It comes
as a ready-to-use `F5-TTS.zip`, available on request from INESC-ID HLT.

```bash
conda env create -f "$ANON_DIR/envs/f5-tts.yml"    # env "f5-tts" (Python 3.11 + ffmpeg)
F5PY="$(conda info --base)/envs/f5-tts/bin/python"
unzip F5-TTS.zip -d ~/F5-TTS                       # gives ~/F5-TTS/F5-TTS
cd ~/F5-TTS/F5-TTS
$F5PY -m pip install --no-deps -r "$ANON_DIR/envs/f5-tts.pip-freeze.txt"
$F5PY -m pip install -e .                          # re-run if you move the folder
$F5PY -c "import f5_tts, torch; print('f5 ok', torch.cuda.is_available())"
cd "$ANON_DIR"
```

> [!NOTE]
> Steps 4 and 5 call each env's Python by its full path (`$F5PY`, `$DZPY`) on purpose:
> with a venv still active, a plain `pip` can install into the venv instead.

### 5. Background noise separation (TUSS + DiariZen)

Every audio job first separates speech from background noise. **TUSS** is already included
in `third_party/`. **DiariZen** needs its own env:

```bash
git clone https://github.com/BUTSpeechFIT/DiariZen.git ~/DiariZen
cd ~/DiariZen
git checkout 844f5555b0a98acd0931511fc641a8c5b8ba92c7     # version used in the thesis
git submodule update --init --recursive                   # its pyannote-audio fork
conda env create -f "$ANON_DIR/envs/diarizen.yml"         # env "diarizen"
DZPY="$(conda info --base)/envs/diarizen/bin/python"
TORCHIDX=https://download.pytorch.org/whl/cu121
$DZPY -m pip install --no-deps -r "$ANON_DIR/envs/diarizen.pip-freeze.txt" --extra-index-url $TORCHIDX
$DZPY -m pip install -e .
cd pyannote-audio && $DZPY -m pip install -e . && cd ..
$DZPY -c "import diarizen, torch; print('diarizen ok', torch.cuda.is_available())"
cd "$ANON_DIR"
```

### 6. kNN-VC voices (only for "Preserve natural speech rhythm")

Put the target speakers' clips in `ref/tmp/`. A speaker is all `.wav` files sharing a
prefix, e.g. `in_aca_*.wav` (about 10 min of 16 kHz mono speech each). The twelve voices
used in the thesis are curated clips of [Imprensa Nacional](https://imprensanacional.pt/digitais/audiolivros/)
audiobooks, available on request from INESC-ID HLT. Without them, these jobs are rejected
with "No kNN-VC reference clips for speaker …".

### 7. Local LLM (optional)

Only for the "Local LLM (Gemma)" replacement source:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:12b
```

## Running

```bash
cd anon-platform
source venv/bin/activate
python web_app.py --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. On a **remote GPU server**, open a tunnel from your own
machine with `ssh -L 8000:localhost:8000 user@server` and browse to the same address.

The app has no authentication, so don't expose it publicly. Jobs are kept in memory and
lost on restart.

<details>
<summary>Different paths or env names? Settings</summary>

If you followed the steps above, the defaults work. Otherwise copy `.env.example` to `.env`,
edit it, and load it before starting the app with `set -a; source .env; set +a`.

| If you… | set |
|---|---|
| put the Whisper model elsewhere | `WHISPER_MODEL_DIR` |
| named the conda envs differently | `MFA_CONDA_ENV`, `F5_TTS_CONDA_ENV`, `DIARIZEN_CONDA_ENV` |
| unzipped F5-TTS elsewhere than `~/F5-TTS/F5-TTS` | `F5_TTS_DIR` |
| want to block all live Wikidata lookups | `PRIVACY_MODE=1` |
| run Ollama elsewhere or with another model | `TEXT_LLM_OLLAMA_HOST`, `TEXT_LLM_MODEL` |
</details>

<details>
<summary>Local cache of looked-up terms</summary>

Replacement lookups are cached in `text_anon/anon_kb.db`, keyed by the entity text
from your documents. It never leaves your machine. To clear it:

```bash
python -c "import sqlite3; c = sqlite3.connect('text_anon/anon_kb.db'); \
c.execute('DELETE FROM kb_cache'); c.execute('DELETE FROM substitution_cache'); c.commit()"
```
</details>

## Project layout

```
web_app.py          web app (FastAPI back end)
static/             web app front end
text_anon/          text anonymization: detection (regex, spaCy, name lexicons) and replacement
  llm/              optional local-LLM replacement source (Ollama)
  anon_kb.db        local Wikidata index used for replacements
audio/              transcription, noise separation, speech synthesis and the audio pipelines
third_party/        bundled TUSS speech/noise separation (AGPL-3.0)
ref/                kNN-VC voices (ref/tmp/, not included) and UI voice previews
envs/               conda environment files and the tested-environment snapshot
scripts/            capture_env.sh (records the exact versions of a working install),
                    build_examples_page.py (rebuilds the examples listening page)
examples/           example outputs from the thesis evaluation
```

## Examples

**[▶ Listen to the examples](https://cati-xf.github.io/anonymization-platform/examples/)**: three BD-PÚBLICO utterances from the thesis
evaluation, run through every pipeline, with and without background noise, each with a
transcript and WER. The audio files are in [`examples/bd-publico/`](examples/bd-publico/).

## License

This code is under the [MIT License](LICENSE), except `third_party/standalone_inference/`
(TUSS, by MERL), which stays under AGPL-3.0-or-later. The F5-TTS and DiariZen model weights
are **non-commercial** (CC BY-NC 4.0).

<details>
<summary>Third-party components</summary>

| Component | Source | License |
|---|---|---|
| F5-TTS code / pretrained weights | SWivid/F5-TTS | MIT / CC-BY-NC-4.0 |
| F5-TTS European Portuguese fine-tune | INESC-ID (not public) | available on request |
| Whisper CAMÕES fine-tune | inesc-id/WhisperLv3-FT-EP-CPP | not stated (base whisper-large-v3: Apache-2.0) |
| kNN-VC | bshall/knn-vc | MIT |
| DiariZen code / model weights | BUTSpeechFIT/DiariZen | MIT / CC-BY-NC-4.0 |
| TUSS (bundled) | merlresearch/unified-source-separation | AGPL-3.0-or-later |
| Montreal Forced Aligner | MontrealCorpusTools | MIT |
| spaCy `pt_core_news_lg` | explosion | CC-BY-SA-4.0 |
| Voice clips | Imprensa Nacional audiobooks, curated at INESC-ID | research use; copyright with Imprensa Nacional |
</details>

## Citation

```bibtex
@mastersthesis{freire2026anonymization,
  author = {Catarina Xie Freire},
  title  = {A Unified Platform for Multi-Modal Data Anonymization: Privacy- and
            Utility-Preserving Text and Speech Anonymization for European Portuguese},
  school = {Instituto Superior T{\'e}cnico, Universidade de Lisboa},
  year   = {2026},
  month  = oct
}
```
