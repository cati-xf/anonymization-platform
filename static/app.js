
// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const name = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
  });
});

// "Pattern types to keep" — collect checked regex labels for a given panel prefix
function collectRegexSkip(pfx) {
  return [...document.querySelectorAll(`.${pfx}_regexskip:checked`)]
    .map(c => c.value).join(',');
}

// Every dropdown-like control on the page (the method <select>s and the
// Subcategories panel) opens as one of these floating panels, and only one
// is open at a time.
const DROPDOWN_SELECTOR = '.skip-labels.open, .custom-select.open';
function closeOtherDropdowns(except) {
  document.querySelectorAll(DROPDOWN_SELECTOR).forEach(el => { if (el !== except) el.classList.remove('open'); });
}

// Replace a plain <select> with a custom trigger + floating option panel,
// so it opens the same way as the Subcategories panel instead of relying on
// the browser's native popup. The original <select> stays in the DOM
// (hidden) as the real value holder, so every existing `.value` read and
// `change` listener elsewhere keeps working unchanged.
function initCustomSelect(select) {
  const wrap = document.createElement('div');
  wrap.className = 'custom-select';

  const trigger = document.createElement('button');
  trigger.type = 'button';
  trigger.className = 'custom-select-trigger';

  const panel = document.createElement('div');
  panel.className = 'custom-select-panel';

  const options = [...select.options];
  const syncSelected = () => {
    const current = options.find(o => o.value === select.value);
    trigger.textContent = current ? current.textContent : '';
    panel.querySelectorAll('.custom-select-option').forEach(row => {
      row.classList.toggle('selected', row.dataset.value === select.value);
    });
  };
  options.forEach(opt => {
    const row = document.createElement('div');
    row.className = 'custom-select-option';
    row.textContent = opt.textContent;
    row.dataset.value = opt.value;
    row.addEventListener('click', () => {
      select.value = opt.value;
      select.dispatchEvent(new Event('change', { bubbles: true }));
      syncSelected();
      wrap.classList.remove('open');
    });
    panel.appendChild(row);
  });
  syncSelected();

  trigger.addEventListener('click', () => {
    const willOpen = !wrap.classList.contains('open');
    closeOtherDropdowns(wrap);
    wrap.classList.toggle('open', willOpen);
  });

  wrap.appendChild(trigger);
  wrap.appendChild(panel);
  select.insertAdjacentElement('afterend', wrap);
  select.style.display = 'none';
}
document.querySelectorAll('select').forEach(initCustomSelect);

// The Exceptions panel (labeled outside it by its own entity-pill) is a
// custom dropdown: clicking its button opens a floating panel (like the
// <select> controls next to it), instead of an accordion that pushes the
// layout open.
document.querySelectorAll('.skip-labels').forEach(box => {
  const button = box.querySelector('.skip-labels-summary');

  // Keep the button label showing how many categories are selected
  const syncLabel = () => {
    const picked = [...box.querySelectorAll('input:checked')]
      .map(c => c.parentElement.textContent.trim());
    const isDefault = picked.length === 0;
    button.textContent = isDefault
      ? 'All subcategories will be anonymized'
      : 'Not anonymized: ' + picked.join(', ');
    button.classList.toggle('is-placeholder', isDefault);
  };
  box.addEventListener('change', syncLabel);
  syncLabel();

  // Toggle the floating panel open/closed
  button.addEventListener('click', () => {
    const willOpen = !box.classList.contains('open');
    closeOtherDropdowns(box);
    box.classList.toggle('open', willOpen);
  });
});

// Click outside, or Escape, closes any open dropdown/subcategories panel
document.addEventListener('click', e => {
  document.querySelectorAll(DROPDOWN_SELECTOR).forEach(el => {
    if (!el.contains(e.target)) el.classList.remove('open');
  });
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeOtherDropdowns(null);
});

// Hide the "subcategories that should not be anonymized" picker when the
// regex method itself is "none" — those checkboxes only matter when regex
// entities are otherwise being anonymized.
['t', 's', 'f'].forEach(pfx => {
  const select = document.getElementById(`${pfx}_regex`);
  if (!select) return;
  const skipBox = select.closest('.config-row').querySelector('.skip-labels');
  const sync = () => {
    const isNone = select.value === 'none';
    skipBox.classList.toggle('hidden', isNone);
    if (isNone) skipBox.classList.remove('open');
  };
  select.addEventListener('change', sync);
  sync();
});

// File upload — each tab holds an array of selected files. Selecting (or
// dropping) more than one file automatically switches that run into batch
// mode: no per-file review, see runAudioPipeline().
const selectedFiles = { spoken: [], voice: [], full: [] };

// { name, blob } entries for each successfully-finished file in the current
// batch run, per tab — used by the "Download All (.zip)" button.
const batchDownloads = { text: [], spoken: [], voice: [], full: [] };

['spoken', 'voice', 'full'].forEach(tab => {
  const zone  = document.getElementById(`uploadZone-${tab}`);
  const input = document.getElementById(`audioFile-${tab}`);
  zone.addEventListener('click', () => input.click());
  input.addEventListener('change', () => { if (input.files.length) applyFiles(tab, [...input.files]); });
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) applyFiles(tab, [...e.dataTransfer.files]);
  });
});

// Text tab: upload one or more .txt/.docx documents as an alternative to
// typing/pasting, as a drag-and-drop zone matching the audio tabs' upload
// zones. Selecting more than one file switches to batch mode (no review) —
// see runTextDetect()/runTextBatch().
let selectedTextFiles = [];

{
  const zone  = document.getElementById('uploadZone-text');
  const input = document.getElementById('docFile-text');
  zone.addEventListener('click', () => input.click());
  input.addEventListener('change', () => { if (input.files.length) applyTextFiles([...input.files]); input.value = ''; });
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) applyTextFiles([...e.dataTransfer.files]);
  });
}

function applyTextFiles(files) {
  selectedTextFiles = files;
  const runBtn = document.getElementById('runBtn-text');
  const warning = document.getElementById('batchWarning-text');

  // A fresh selection makes any previously-shown batch results stale.
  document.getElementById('batchResults-text').classList.add('hidden');

  if (files.length <= 1) {
    warning.classList.add('hidden');
    document.getElementById('textInputAlt').classList.remove('hidden');
    runBtn.textContent = '🔍 Detect Entities';
    if (files.length === 1) extractDocFile(files[0]);
    return;
  }

  const zone = document.getElementById('uploadZone-text');
  const name = document.getElementById('uploadName-text');
  zone.classList.add('has-file');
  name.style.display = 'block';
  name.textContent = `✓ ${files.length} files selected: ${files.map(f => f.name).join(', ')}`;
  reviewState.text.docId = null;

  // The typed/pasted-text box doesn't apply once multiple files are queued.
  document.getElementById('textInputAlt').classList.add('hidden');

  document.getElementById('batchCount-text').textContent = files.length;
  warning.classList.remove('hidden');
  runBtn.textContent = `⚡ Run Batch (${files.length} files)`;
  document.getElementById('reviewSection-text').classList.add('hidden');
  document.getElementById('textResult').classList.add('hidden');
}

async function extractDocFile(file) {
  const zone = document.getElementById('uploadZone-text');
  const main = zone.querySelector('.upload-main');
  const name = document.getElementById('uploadName-text');
  const originalMain = main.textContent;
  main.textContent = '⏳ Extracting…';
  name.style.display = 'none';

  const form = new FormData();
  form.append('document', file);

  try {
    const resp = await fetch('/extract-text', { method: 'POST', body: form });
    if (!resp.ok) { alert('Could not extract text: ' + await resp.text()); return; }
    const data = await resp.json();
    document.getElementById('textInput').value = data.text;
    reviewState.text.docId = data.doc_id || null;
    zone.classList.add('has-file');
    name.textContent = `✓ ${file.name}`;
    name.style.display = 'block';
    // A file's own content is now the input — hide the separate paste box.
    document.getElementById('textInputAlt').classList.add('hidden');
  } catch (e) {
    alert('Could not reach server: ' + e.message);
  } finally {
    main.textContent = originalMain;
  }
}

function applyFiles(tab, files) {
  selectedFiles[tab] = files;
  const zone = document.getElementById(`uploadZone-${tab}`);
  const name = document.getElementById(`uploadName-${tab}`);
  zone.classList.add('has-file');
  name.style.display = 'block';
  if (files.length === 1) {
    name.textContent = `✓ ${files[0].name}  (${(files[0].size/1024).toFixed(0)} KB)`;
  } else {
    name.textContent = `✓ ${files.length} files selected: ${files.map(f => f.name).join(', ')}`;
  }

  const warning = document.getElementById(`batchWarning-${tab}`);
  const countEl = document.getElementById(`batchCount-${tab}`);
  if (countEl) countEl.textContent = files.length;
  if (warning) warning.classList.toggle('hidden', files.length <= 1);

  // A fresh selection makes any previously-shown batch results stale.
  const batchCard = document.getElementById(`batchResults-${tab}`);
  if (batchCard) batchCard.classList.add('hidden');
}

function getEffectivePipeline(tab) {
  if (tab === 'spoken') return 'infilling';
  if (tab === 'voice')  return document.querySelector('input[name="vc_method"]:checked').value;
  if (tab === 'full')   return document.querySelector('input[name="full_method"]:checked').value;
}

// Show/hide reference speaker card for kNN pipelines
document.querySelectorAll('input[name="vc_method"]').forEach(r => {
  r.addEventListener('change', () => {
    const isKnn = r.value === 'vc_knn' && r.checked;
    document.getElementById('refSetCard-voice').classList.toggle('hidden', !isKnn);
    document.getElementById('f5VoiceCard-voice').classList.toggle('hidden', isKnn);
    document.getElementById('noiseCard-voice').classList.toggle('hidden', !isKnn);
    if (!isKnn) document.querySelector('input[name="noise_mode_voice"][value="remove"]').checked = true;
  });
});
document.querySelectorAll('input[name="full_method"]').forEach(r => {
  r.addEventListener('change', () => {
    const isInfilling = r.value === 'full_infilling_knn' && r.checked;
    document.getElementById('refSetCard-full').classList.toggle('hidden', r.value !== 'full_infilling_knn' || !r.checked);
    document.getElementById('f5VoiceCard-full').classList.toggle('hidden', r.value !== 'full_asr_tts' || !r.checked);
    document.getElementById('noiseCard-full').classList.toggle('hidden', !isInfilling);
    if (!isInfilling) document.querySelector('input[name="noise_mode_full"][value="remove"]').checked = true;
  });
});

// ── Entity review state ───────────────────────────────────────────────────────

// The three interchangeable infilling synthesis backends for anonymized
// spans. "Original" is always what the initial job used; the other two are
// only generated on demand (see switchVariant()) and then cached by job_id
// so navigating back to an already-heard one is free.
const SYNTH_VARIANTS = [
  { key: 'f5_tts', label: 'Original',      pipeline: null },
  { key: 'f5',     label: 'Alternative 1', pipeline: 'f5' },
  { key: 'google', label: 'Alternative 2', pipeline: 'google' },
];

// Per-tab review state
const reviewState = {
  text:   { text: '', entities: [], nextCustomId: 1, docId: null },
  spoken: { text: '', entities: [], nextCustomId: 1, jobId: null, variantJobs: null, currentVariant: null },
  voice:  { text: '', entities: [], nextCustomId: 1, jobId: null },
  full:   { text: '', entities: [], nextCustomId: 1, jobId: null, variantJobs: null, currentVariant: null, activePipeline: null },
};

// Manually editing the text invalidates the uploaded .docx's paragraph
// mapping used by downloadTextResult() — fall back to a plain .txt download.
document.getElementById('textInput').addEventListener('input', () => { reviewState.text.docId = null; });

// Maps CSS class suffix by label
function labelClass(label, source) {
  if (source === 'custom') return 'em-custom';
  const map = { PER: 'em-per', LOC: 'em-loc', ORG: 'em-org', MISC: 'em-misc' };
  return map[label] || 'em-regex';
}

function pillClass(label, source) {
  if (source === 'custom') return 'ep-per';
  const map = { PER: 'ep-per', LOC: 'ep-loc', ORG: 'ep-org', MISC: 'ep-misc' };
  return map[label] || 'ep-regex';
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Render the transcript with entity highlights
function renderTranscript(tab) {
  const state = reviewState[tab];
  const text = state.text;
  const entities = state.entities;

  // Only show non-removed entities as marks
  const active = entities
    .filter(e => !e.removed && e.start < e.end)
    .sort((a, b) => a.start - b.start);

  let html = '';
  let pos = 0;
  for (const e of active) {
    if (e.start > pos) html += escHtml(text.slice(pos, e.start));
    const cls = labelClass(e.label, e.source);
    const replLabel = (tab === 'spoken' || tab === 'full') && e.replacement === 'XXXXXX' ? '🔇 [BEEP]' : (e.replacement || '');
    html += `<mark class="entity-mark ${cls}" data-id="${e.id}" title="${escHtml(replLabel)}">${escHtml(e.original)}</mark>`;
    pos = e.end;
  }
  if (pos < text.length) html += escHtml(text.slice(pos));

  document.getElementById(`reviewTranscript-${tab}`).innerHTML = html;

  // Clicking a mark scrolls to and flashes the matching chip
  document.querySelectorAll(`#reviewTranscript-${tab} .entity-mark`).forEach(mark => {
    mark.addEventListener('click', () => {
      const chip = document.querySelector(`#entityList-${tab} .entity-chip[data-id="${mark.dataset.id}"]`);
      if (chip) {
        chip.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        chip.style.outline = '2px solid #3b82f6';
        setTimeout(() => { chip.style.outline = ''; }, 1200);
      }
    });
  });
}

// Render the entity list chips
function renderEntityList(tab) {
  const state = reviewState[tab];
  const list = document.getElementById(`entityList-${tab}`);

  if (!state.entities.length) {
    list.innerHTML = '<div class="review-empty">No entities detected. Use "Add Sensitive Word" to mark content manually.</div>';
    return;
  }

  const isAudio = tab === 'spoken' || tab === 'full';

  const hasUnresolved = state.entities.some(
    e => !e.removed && e.source !== 'custom' && !e.local_match && (e.method === 'gen' || e.method === 'rand')
  );
  const privacyNote = hasUnresolved ? `
    <div style="display:flex;align-items:flex-start;gap:.6rem;background:#fef3c7;border:1px solid #fcd34d;border-radius:.5rem;padding:.6rem .875rem;margin-bottom:.75rem;font-size:.825rem;color:#78350f;line-height:1.5">
      <span style="flex-shrink:0;font-size:1rem">⚠️</span>
      <span>Some entities were not found in the local database and show no local match. Clicking <strong>🌐 Search Wikidata</strong> will send that term to wikidata.org. Only use it if you are comfortable sending this data to an external service.</span>
    </div>` : '';

  list.innerHTML = privacyNote + state.entities.map(e => {
    const pc = pillClass(e.label, e.source);
    const labelText = e.source === 'custom' ? 'CUSTOM' : e.label;
    const removed = e.removed;
    const isBeep = isAudio && e.replacement === 'XXXXXX';
    const beepBtn = isAudio && !removed
      ? `<button class="ec-beep${isBeep ? ' is-active' : ''}" data-id="${e.id}" title="${isBeep ? 'Click to switch back to text replacement' : 'Replace with a redaction beep sound'}">🔇 Beep</button>`
      : '';
    const showLive = !removed && e.source !== 'custom' && !e.local_match
                     && (e.method === 'gen' || e.method === 'rand');
    const liveBtn = showLive
      ? `<button class="ec-live" data-id="${e.id}" data-term="${escHtml(e.original)}" data-label="${escHtml(e.label)}" data-method="${escHtml(e.method || '')}" title="Sends '${escHtml(e.original)}' to wikidata.org — external network request">🌐 Search Wikidata</button>`
      : '';
    const noLocalNote = showLive
      ? `<span class="ec-no-local">(no local match)</span>`
      : '';
    return `
    <div class="entity-chip${removed ? ' removed' : ''}" data-id="${e.id}">
      <span class="entity-pill ${pc} ec-label" style="font-size:.72rem">${escHtml(labelText)}</span>
      <span class="ec-original" title="${escHtml(e.original)}">${escHtml(e.original)}</span>
      <span class="ec-arrow">→</span>
      ${noLocalNote}
      <input class="ec-repl${isBeep ? ' is-beep' : ''}" data-id="${e.id}" value="${isBeep ? '🔇 [BEEP]' : escHtml(e.replacement || '')}"
             ${removed ? 'disabled' : ''} placeholder="replacement…"
             data-beep="${isBeep ? '1' : '0'}">
      ${liveBtn}
      ${beepBtn}
      <button class="ec-toggle ${removed ? 'inactive' : 'active'}" data-id="${e.id}"
              title="${removed ? 'Re-enable' : 'Remove'}">
        ${removed ? '↩ Restore' : '× Remove'}
      </button>
    </div>`;
  }).join('');

  // Replacement edits
  list.querySelectorAll('.ec-repl').forEach(input => {
    input.addEventListener('focus', () => {
      if (input.dataset.beep === '1') {
        input.value = '';
        input.dataset.beep = '0';
        input.classList.remove('is-beep');
        const ent = state.entities.find(e => e.id === input.dataset.id);
        if (ent) { ent.replacement = ''; renderTranscript(tab); }
      }
    });
    input.addEventListener('input', () => {
      const ent = state.entities.find(e => e.id === input.dataset.id);
      if (ent) { ent.replacement = input.value; renderTranscript(tab); }
    });
  });

  // Beep button — toggles beep on/off
  list.querySelectorAll('.ec-beep').forEach(btn => {
    btn.addEventListener('click', () => {
      const ent = state.entities.find(e => e.id === btn.dataset.id);
      if (ent) {
        ent.replacement = ent.replacement === 'XXXXXX' ? '' : 'XXXXXX';
        renderEntityList(tab);
        renderTranscript(tab);
      }
    });
  });

  // Live Wikidata lookup
  list.querySelectorAll('.ec-live').forEach(btn => {
    btn.addEventListener('click', async () => {
      const ent = state.entities.find(e => e.id === btn.dataset.id);
      if (!ent) return;
      btn.disabled = true;
      btn.textContent = '⏳ Searching…';
      const form = new FormData();
      form.append('term', btn.dataset.term);
      form.append('ner_label', btn.dataset.label);
      form.append('method', btn.dataset.method);
      try {
        const resp = await fetch('/lookup-live', { method: 'POST', body: form });
        const data = await resp.json();
        if (data.replacement) {
          ent.replacement = data.replacement;
          ent.local_match = true;
          renderEntityList(tab);
          renderTranscript(tab);
        } else {
          btn.replaceWith(Object.assign(document.createElement('span'), {
            className: 'ec-no-local',
            textContent: '(not found in Wikidata either)',
          }));
        }
      } catch {
        btn.disabled = false;
        btn.textContent = '🌐 Search Wikidata';
      }
    });
  });

  // Toggle remove/restore
  list.querySelectorAll('.ec-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const ent = state.entities.find(e => e.id === btn.dataset.id);
      if (ent) {
        ent.removed = !ent.removed;
        renderEntityList(tab);
        renderTranscript(tab);
      }
    });
  });
}

function showReviewUI(tab, text, entities) {
  const state = reviewState[tab];
  state.text = text;
  state.entities = entities.map(e => ({ ...e, removed: false }));
  state.nextCustomId = 1;
  renderTranscript(tab);
  renderEntityList(tab);
  document.getElementById(`reviewSection-${tab}`).classList.remove('hidden');
}

// Add custom entity
function showAddCustom(tab) {
  document.getElementById(`addCustomForm-${tab}`).classList.remove('hidden');
  document.getElementById(`customWord-${tab}`).focus();
}
function hideAddCustom(tab) {
  document.getElementById(`addCustomForm-${tab}`).classList.add('hidden');
  document.getElementById(`customWord-${tab}`).value = '';
  document.getElementById(`customRepl-${tab}`).value = '';
}

function addCustomEntity(tab) {
  const state = reviewState[tab];
  const wordInput = document.getElementById(`customWord-${tab}`);
  const replInput = document.getElementById(`customRepl-${tab}`);
  const phrase = wordInput.value.trim();
  if (!phrase) { wordInput.focus(); return; }

  const repl = replInput.value.trim() || `[CUSTOM_${state.nextCustomId}]`;
  const text = state.text;

  // Find all occurrences (case-insensitive)
  const lowerText = text.toLowerCase();
  const lowerPhrase = phrase.toLowerCase();
  let found = 0;
  let idx = 0;
  while ((idx = lowerText.indexOf(lowerPhrase, idx)) !== -1) {
    const end = idx + phrase.length;
    // Skip if overlaps existing entity
    const overlaps = state.entities.some(e => !e.removed && e.start < end && e.end > idx);
    if (!overlaps) {
      state.entities.push({
        id: `custom_${state.nextCustomId++}`,
        start: idx,
        end: end,
        original: text.slice(idx, end),
        replacement: repl,
        label: 'CUSTOM',
        source: 'custom',
        removed: false,
      });
      found++;
    }
    idx = end;
  }

  if (!found) {
    alert(`"${phrase}" not found in the transcript.`);
    return;
  }

  // Sort entities by position for consistent rendering
  state.entities.sort((a, b) => a.start - b.start);
  hideAddCustom(tab);
  renderEntityList(tab);
  renderTranscript(tab);
}

// ── Text tab two-phase flow ───────────────────────────────────────────────────

// More than one document selected → batch mode: each file is detected AND
// anonymized automatically, with no review step in between.
async function runTextDetect() {
  if (selectedTextFiles.length > 1) { await runTextBatch(); return; }

  const text = document.getElementById('textInput').value.trim();
  if (!text) { alert('Please enter some text to anonymize.'); return; }

  const btn = document.getElementById('runBtn-text');
  btn.disabled = true;
  btn.textContent = '⏳ Detecting…';
  document.getElementById('textResult').classList.add('hidden');
  document.getElementById('reviewSection-text').classList.add('hidden');

  const form = new FormData();
  form.append('text', text);
  form.append('per',          document.getElementById('t_per').value);
  form.append('loc',          document.getElementById('t_loc').value);
  form.append('org',          document.getElementById('t_org').value);
  form.append('misc',         document.getElementById('t_misc').value);
  form.append('regex_method', document.getElementById('t_regex').value);
  form.append('whitelist',    document.getElementById('t_whitelist').value);
  form.append('regex_skip',   collectRegexSkip('t'));
  form.append('replacement_source', document.querySelector('input[name="t_replacement_source"]:checked').value);

  try {
    const resp = await fetch('/detect-text', { method: 'POST', body: form });
    if (!resp.ok) { alert('Detection failed: ' + await resp.text()); return; }
    const data = await resp.json();
    showReviewUI('text', data.text, data.entities);
  } catch (e) {
    alert('Could not reach server: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '🔍 Detect Entities';
  }
}

async function runTextBatch() {
  const files = selectedTextFiles;
  const btn = document.getElementById('runBtn-text');
  btn.disabled = true;
  document.getElementById('reviewSection-text').classList.add('hidden');
  document.getElementById('textResult').classList.add('hidden');

  const warning = document.getElementById('batchWarning-text');
  document.getElementById('batchCount-text').textContent = files.length;
  warning.classList.remove('hidden');

  const list = document.getElementById('batchList-text');
  list.innerHTML = '';
  document.getElementById('batchResults-text').classList.remove('hidden');
  resetBatchDownloads('text');

  const cfg = {
    per:          document.getElementById('t_per').value,
    loc:          document.getElementById('t_loc').value,
    org:          document.getElementById('t_org').value,
    misc:         document.getElementById('t_misc').value,
    regex_method: document.getElementById('t_regex').value,
    whitelist:    document.getElementById('t_whitelist').value,
    regex_skip:   collectRegexSkip('t'),
    replacement_source: document.querySelector('input[name="t_replacement_source"]:checked').value,
  };

  for (const file of files) {
    const row = createBatchRow(file.name);
    list.appendChild(row.el);
    await runBatchTextFile(file, cfg, row);
  }
  btn.disabled = false;
  revealDownloadAllIfAny('text');
}

async function runBatchTextFile(file, cfg, row) {
  try {
    setBatchRowStatus(row, 'running', 'Extracting…');
    const extractForm = new FormData();
    extractForm.append('document', file);
    const extractResp = await fetch('/extract-text', { method: 'POST', body: extractForm });
    if (!extractResp.ok) throw new Error(await extractResp.text());
    const { text, doc_id } = await extractResp.json();

    setBatchRowStatus(row, 'running', 'Detecting…');
    const detectForm = new FormData();
    detectForm.append('text', text);
    Object.entries(cfg).forEach(([k, v]) => detectForm.append(k, v));
    const detectResp = await fetch('/detect-text', { method: 'POST', body: detectForm });
    if (!detectResp.ok) throw new Error(await detectResp.text());
    const { entities } = await detectResp.json();

    setBatchRowStatus(row, 'running', 'Anonymizing…');
    const activeEntities = entities.filter(e => e.replacement);
    const applyForm = new FormData();
    applyForm.append('text', text);
    applyForm.append('entities', JSON.stringify(activeEntities));
    const applyResp = await fetch('/apply-text', { method: 'POST', body: applyForm });
    if (!applyResp.ok) throw new Error(await applyResp.text());
    const { anonymized } = await applyResp.json();

    const baseName = file.name.replace(/\.[^.]+$/, '');
    setBatchRowStatus(row, 'running', 'Preparing download…');
    const { blob, filename } = await getTextResultBlob(doc_id, anonymized, baseName);

    // Batch mode intentionally shows no per-file preview or download — only
    // the aggregate "Download All (.zip)" button once the whole run finishes.
    setBatchRowStatus(row, 'done', '✓ Done');
    batchDownloads.text.push({ name: filename, blob });
  } catch (e) {
    setBatchRowStatus(row, 'error', '✗ Error');
    row.bodyEl.style.display = 'block';
    row.bodyEl.textContent = e.message;
  }
}

// Builds the downloadable file for one anonymized text result — a rebuilt
// .docx (via the server, keeping the original document's formatting) when a
// doc_id is available, otherwise a plain .txt blob.
async function getTextResultBlob(docId, anonymizedText, baseName) {
  if (!docId) {
    return { blob: new Blob([anonymizedText], { type: 'text/plain;charset=utf-8' }), filename: `${baseName}_anonymized.txt` };
  }
  const form = new FormData();
  form.append('doc_id', docId);
  form.append('text', anonymizedText);
  const resp = await fetch('/download-docx', { method: 'POST', body: form });
  if (!resp.ok) throw new Error(await resp.text());
  return { blob: await resp.blob(), filename: `${baseName}_anonymized.docx` };
}

async function applyTextAnonymization() {
  const state = reviewState['text'];
  const activeEntities = state.entities.filter(e => !e.removed && e.replacement);
  if (!activeEntities.length && !confirm('No active entities — the text will be returned unchanged. Continue?')) return;

  const btn = document.getElementById('applyBtn-text');
  btn.disabled = true;
  btn.textContent = '⏳ Applying…';

  const form = new FormData();
  form.append('text', state.text);
  form.append('entities', JSON.stringify(activeEntities));

  try {
    const resp = await fetch('/apply-text', { method: 'POST', body: form });
    if (!resp.ok) { alert('Apply failed: ' + await resp.text()); return; }
    const data = await resp.json();
    document.getElementById('textOriginalBox').textContent = data.original;
    document.getElementById('textResultBox').textContent = data.anonymized;
    document.getElementById('textResult').classList.remove('hidden');
    document.getElementById('textResult').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (e) {
    alert('Could not reach server: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '✓ Apply Anonymization';
  }
}

function triggerBlobDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function downloadTextResult() {
  const anonymizedText = document.getElementById('textResultBox').textContent;
  const docId = reviewState.text.docId;

  if (!docId) {
    triggerBlobDownload(new Blob([anonymizedText], { type: 'text/plain;charset=utf-8' }), 'anonymized.txt');
    return;
  }

  const btn = document.getElementById('downloadBtn-text');
  btn.disabled = true;
  btn.textContent = '⏳ Preparing…';

  const form = new FormData();
  form.append('doc_id', docId);
  form.append('text', anonymizedText);

  try {
    const resp = await fetch('/download-docx', { method: 'POST', body: form });
    if (!resp.ok) { alert('Could not generate .docx: ' + await resp.text()); return; }
    triggerBlobDownload(await resp.blob(), 'anonymized.docx');
  } catch (e) {
    alert('Could not reach server: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '⬇ Download';
  }
}

// ── Audio tab pipeline ────────────────────────────────────────────────────────

let activeEventSource = null;

// Builds the /jobs FormData shared by the single-file and batch paths.
// batchMode=true tells the backend to skip the human review/apply pause
// (only pipelines with a detect→review→synthesize flow honor it).
function buildAudioFormData(tab, pipeline, file, batchMode) {
  const pfx = tab === 'spoken' ? 's' : tab === 'full' ? 'f' : null;
  const form = new FormData();
  form.append('file', file);
  form.append('pipeline', pipeline);
  if (pfx) {
    form.append('per',          document.getElementById(`${pfx}_per`).value);
    form.append('loc',          document.getElementById(`${pfx}_loc`).value);
    form.append('org',          document.getElementById(`${pfx}_org`).value);
    form.append('misc',         document.getElementById(`${pfx}_misc`).value);
    form.append('regex_method', document.getElementById(`${pfx}_regex`).value);
    form.append('whitelist',    document.getElementById(`${pfx}_whitelist`).value);
    form.append('regex_skip',   collectRegexSkip(pfx));
    form.append('replacement_source', document.querySelector(`input[name="${pfx}_replacement_source"]:checked`).value);
  }

  if (pipeline === 'vc_tts') {
    const checked = document.querySelector('input[name="vc_f5_voice"]:checked');
    form.append('f5_voice', checked ? checked.value : 'in_aca');
  } else if (pipeline === 'vc_knn') {
    const checked = document.querySelector('input[name="vc_ref_set"]:checked');
    form.append('ref_set', checked ? checked.value : 'in_aca');
  } else if (pipeline === 'full_infilling_knn') {
    const checked = document.querySelector('input[name="full_ref_set"]:checked');
    form.append('ref_set', checked ? checked.value : 'in_aca');
  } else if (pipeline === 'full_asr_tts') {
    const checked = document.querySelector('input[name="full_f5_voice"]:checked');
    form.append('f5_voice', checked ? checked.value : 'in_aca');
  }

  const noiseModeEl = document.querySelector(`input[name="noise_mode_${tab}"]:checked`);
  form.append('noise_mode', noiseModeEl ? noiseModeEl.value : 'remove');
  form.append('batch_mode', batchMode ? 'true' : 'false');
  return form;
}

// One file selected → the normal single-job review flow (unchanged).
// More than one file selected → batch mode: no review, processed one after
// another, results shown in a per-file list.
async function runAudioPipeline(tab) {
  const files = selectedFiles[tab];
  if (!files || !files.length) { alert('Please select an audio file first.'); return; }
  if (files.length > 1) {
    await runAudioBatch(tab, files);
  } else {
    await runSingleAudioJob(tab, files[0]);
  }
}

async function runSingleAudioJob(tab, file) {
  resetAudioUI(tab);
  const batchCard = document.getElementById(`batchResults-${tab}`);
  if (batchCard) batchCard.classList.add('hidden');
  const warning = document.getElementById(`batchWarning-${tab}`);
  if (warning) warning.classList.add('hidden');

  const pipeline = getEffectivePipeline(tab);
  const form = buildAudioFormData(tab, pipeline, file, false);

  document.getElementById(`runBtn-${tab}`).disabled = true;
  document.getElementById(`jobStatus-${tab}`).classList.remove('hidden');

  let jobId;
  try {
    const resp = await fetch('/jobs', { method: 'POST', body: form });
    if (!resp.ok) { showJobError('Server error: ' + await resp.text(), tab); return; }
    jobId = (await resp.json()).job_id;
    reviewState[tab].jobId = jobId;
    if (reviewState[tab].activePipeline !== undefined) reviewState[tab].activePipeline = pipeline;
    if (reviewState[tab].variantJobs !== undefined) {
      reviewState[tab].variantJobs = { f5_tts: jobId, f5: null, google: null };
      reviewState[tab].currentVariant = 'f5_tts';
    }
  } catch (e) { showJobError('Could not reach server: ' + e.message, tab); return; }

  if (activeEventSource) activeEventSource.close();
  activeEventSource = new EventSource(`/jobs/${jobId}/stream`);
  activeEventSource.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'ping') return;
    if (msg.type === 'log')    { appendLog(msg.msg, tab); return; }
    if (msg.type === 'review') {
      // Don't close SSE — keep it open for synthesis logs
      onJobReview(msg.transcript, msg.entities, tab);
      return;
    }
    if (msg.type === 'done')  { activeEventSource.close(); onJobDone(jobId, tab); return; }
    if (msg.type === 'error') { activeEventSource.close(); showJobError(msg.msg, tab); return; }
  };
  activeEventSource.onerror = () => { activeEventSource.close(); showJobError('Lost connection to server.', tab); };
}

// ── Batch mode (multiple files, no review) ─────────────────────────────────

function createBatchRow(filename) {
  const el = document.createElement('div');
  el.className = 'batch-row';
  el.innerHTML = `
    <div class="batch-row-head">
      <span class="batch-row-name">${escHtml(filename)}</span>
      <span class="status-badge badge-running"><span class="spinner"></span> Queued</span>
    </div>
    <div class="batch-row-body" style="display:none"></div>
  `;
  return { el, statusEl: el.querySelector('.status-badge'), bodyEl: el.querySelector('.batch-row-body') };
}

function setBatchRowStatus(row, kind, text) {
  const cls = kind === 'done' ? 'badge-done' : kind === 'error' ? 'badge-error' : 'badge-running';
  const spinner = kind === 'running' ? '<span class="spinner"></span> ' : '';
  row.statusEl.className = `status-badge ${cls}`;
  row.statusEl.innerHTML = spinner + text;
}

// Resets a tab's batch-download list and hides the "Download All" button —
// called at the start of every batch run.
function resetBatchDownloads(tab) {
  batchDownloads[tab] = [];
  const btn = document.getElementById(`downloadAllBtn-${tab}`);
  if (btn) { btn.classList.add('hidden'); btn.disabled = false; btn.textContent = '⬇ Download All (.zip)'; }
}

// Shows the "Download All" button once a batch run has finished, if at
// least one file produced a downloadable result.
function revealDownloadAllIfAny(tab) {
  const btn = document.getElementById(`downloadAllBtn-${tab}`);
  if (btn) btn.classList.toggle('hidden', batchDownloads[tab].length === 0);
}

async function downloadAllBatch(tab) {
  const entries = batchDownloads[tab];
  if (!entries || !entries.length) return;

  const btn = document.getElementById(`downloadAllBtn-${tab}`);
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = '⏳ Zipping…';

  try {
    const zip = new JSZip();
    const usedNames = new Set();
    entries.forEach(({ name, blob }) => zip.file(uniqueZipName(usedNames, name), blob));
    const zipBlob = await zip.generateAsync({ type: 'blob' });
    triggerBlobDownload(zipBlob, `batch_results_${tab}.zip`);
  } catch (e) {
    alert('Could not create zip: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
}

function uniqueZipName(usedNames, name) {
  if (!usedNames.has(name)) { usedNames.add(name); return name; }
  const dot = name.lastIndexOf('.');
  const base = dot === -1 ? name : name.slice(0, dot);
  const ext = dot === -1 ? '' : name.slice(dot);
  let i = 2;
  let candidate = `${base}_${i}${ext}`;
  while (usedNames.has(candidate)) { i++; candidate = `${base}_${i}${ext}`; }
  usedNames.add(candidate);
  return candidate;
}

async function runAudioBatch(tab, files) {
  resetAudioUI(tab);
  document.getElementById(`jobStatus-${tab}`).classList.add('hidden');
  const reviewSection = document.getElementById(`reviewSection-${tab}`);
  if (reviewSection) reviewSection.classList.add('hidden');
  document.getElementById(`audioResult-${tab}`).classList.add('hidden');

  const warning = document.getElementById(`batchWarning-${tab}`);
  if (warning) {
    document.getElementById(`batchCount-${tab}`).textContent = files.length;
    warning.classList.remove('hidden');
  }

  const batchCard = document.getElementById(`batchResults-${tab}`);
  const list = document.getElementById(`batchList-${tab}`);
  list.innerHTML = '';
  batchCard.classList.remove('hidden');
  resetBatchDownloads(tab);

  const pipeline = getEffectivePipeline(tab);
  const rows = files.map(f => {
    const row = createBatchRow(f.name);
    list.appendChild(row.el);
    return row;
  });

  document.getElementById(`runBtn-${tab}`).disabled = true;
  for (let i = 0; i < files.length; i++) {
    await runBatchAudioFile(tab, files[i], pipeline, rows[i]);
  }
  document.getElementById(`runBtn-${tab}`).disabled = false;
  revealDownloadAllIfAny(tab);
}

function runBatchAudioFile(tab, file, pipeline, row) {
  return new Promise(resolve => {
    setBatchRowStatus(row, 'running', 'Running…');
    const form = buildAudioFormData(tab, pipeline, file, true);

    fetch('/jobs', { method: 'POST', body: form }).then(resp => {
      if (!resp.ok) return resp.text().then(t => { throw new Error(t); });
      return resp.json();
    }).then(data => {
      const jobId = data.job_id;
      const es = new EventSource(`/jobs/${jobId}/stream`);
      es.onmessage = e => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'ping' || msg.type === 'log' || msg.type === 'review') return;
        if (msg.type === 'done') {
          es.close();
          setBatchRowStatus(row, 'done', '✓ Done');
          renderBatchAudioResult(tab, row, jobId, file.name).then(resolve).catch(resolve);
        }
        if (msg.type === 'error') {
          es.close();
          setBatchRowStatus(row, 'error', '✗ Error');
          row.bodyEl.style.display = 'block';
          row.bodyEl.textContent = msg.msg;
          resolve();
        }
      };
      es.onerror = () => {
        es.close();
        setBatchRowStatus(row, 'error', '✗ Connection lost');
        resolve();
      };
    }).catch(err => {
      setBatchRowStatus(row, 'error', '✗ Error');
      row.bodyEl.style.display = 'block';
      row.bodyEl.textContent = err.message;
      resolve();
    });
  });
}

// Batch mode intentionally shows no per-file preview or download — only the
// aggregate "Download All (.zip)" button once the whole run finishes.
async function renderBatchAudioResult(tab, row, jobId, filename) {
  const data = await (await fetch(`/jobs/${jobId}/result`)).json();
  if (data.result && data.result.output_audio) {
    const baseName = filename.replace(/\.[^.]+$/, '');
    const audioUrl = `/jobs/${jobId}/audio`;
    try {
      const blob = await (await fetch(audioUrl)).blob();
      batchDownloads[tab].push({ name: `${baseName}_anonymized.wav`, blob });
    } catch (e) {
      // Zip just won't include this file.
    }
  }
}

function onJobReview(transcript, entities, tab) {
  // Update status badge to show waiting state
  document.getElementById(`jobTitle-${tab}`).textContent = '📋 Review required';
  const badge = document.getElementById(`statusBadge-${tab}`);
  badge.className = 'status-badge';
  badge.style.background = '#fef3c7';
  badge.style.color = '#92400e';
  badge.textContent = '⏸ Awaiting review';
  setProgressState(tab, 'paused', 'Waiting for your review');

  showReviewUI(tab, transcript, entities);
  document.getElementById(`reviewSection-${tab}`).scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function applyAudioAnonymization(tab) {
  const state = reviewState[tab];
  const jobId = state.jobId;
  if (!jobId) return;

  const activeEntities = state.entities.filter(e => !e.removed);
  if (!activeEntities.length && !confirm('No active entities — audio will be synthesized unchanged. Continue?')) return;

  const btn = document.getElementById(`applyBtn-${tab}`);
  btn.disabled = true;
  btn.textContent = '⏳ Submitting…';

  // Update badge back to running
  document.getElementById(`jobTitle-${tab}`).textContent = '⚙️ Running…';
  const badge = document.getElementById(`statusBadge-${tab}`);
  badge.className = 'status-badge badge-running';
  badge.style.background = '';
  badge.style.color = '';
  badge.innerHTML = '<span class="spinner"></span> Running';
  setProgressState(tab, 'running', 'Synthesizing…');
  document.getElementById(`reviewSection-${tab}`).classList.add('hidden');

  const form = new FormData();
  form.append('entities', JSON.stringify(activeEntities));

  try {
    const resp = await fetch(`/jobs/${jobId}/apply`, { method: 'POST', body: form });
    if (!resp.ok) {
      showJobError('Apply failed: ' + await resp.text(), tab);
      btn.disabled = false;
      btn.textContent = '✓ Apply & Synthesize';
    }
    // SSE is still open, synthesis logs will flow through
  } catch (e) {
    showJobError('Could not reach server: ' + e.message, tab);
    btn.disabled = false;
    btn.textContent = '✓ Apply & Synthesize';
  }
}

// Only step markers ("▶ ...") are surfaced as the progress label; raw terminal output is dropped.
function appendLog(text, tab) {
  if (!text.startsWith('▶')) return;
  const panel = document.getElementById(`logPanel-${tab}`);
  panel.textContent = text.replace(/^▶\s*/, '');
  panel.dataset.state = 'running';
}

function setProgressState(tab, state, label) {
  const panel = document.getElementById(`logPanel-${tab}`);
  if (!panel) return;
  panel.dataset.state = state;
  if (label !== undefined) panel.textContent = label;
}

async function onJobDone(jobId, tab) {
  const data = await (await fetch(`/jobs/${jobId}/result`)).json();
  document.getElementById(`jobTitle-${tab}`).textContent = '✅ Complete';
  const badge = document.getElementById(`statusBadge-${tab}`);
  badge.className = 'status-badge badge-done';
  badge.style.background = '';
  badge.style.color = '';
  badge.textContent = '✓ Done';
  setProgressState(tab, 'done', 'Done');

  if (data.result) {
    const r = data.result;
    const diffGrid      = document.getElementById(`diffGrid-${tab}`);
    const transcriptSec = document.getElementById(`transcriptSection-${tab}`);
    const audioSection  = document.getElementById(`audioSection-${tab}`);
    const resultDivider = document.getElementById(`resultDivider-${tab}`);

    if (r.transcript && r.anonymized) {
      document.getElementById(`transcriptBox-${tab}`).textContent = r.transcript;
      document.getElementById(`anonymizedBox-${tab}`).textContent = r.anonymized;
      diffGrid.classList.remove('hidden');
      transcriptSec.classList.add('hidden');
    } else if (r.transcript) {
      document.getElementById(`transcriptOnlyBox-${tab}`).textContent = r.transcript;
      transcriptSec.classList.remove('hidden');
      diffGrid.classList.add('hidden');
    } else {
      diffGrid.classList.add('hidden');
      transcriptSec.classList.add('hidden');
    }

    const hasText  = r.transcript || r.anonymized;
    const hasAudio = !!r.output_audio;
    resultDivider.classList.toggle('hidden', !hasText || !hasAudio);

    if (hasAudio) {
      const audioUrl = `/jobs/${jobId}/audio`;
      document.getElementById(`audioPlayer-${tab}`).src   = audioUrl;
      document.getElementById(`downloadLink-${tab}`).href = audioUrl;
      audioSection.classList.remove('hidden');
    } else {
      audioSection.classList.add('hidden');
    }

    document.getElementById(`audioResult-${tab}`).classList.remove('hidden');
  }

  renderVariantSwitcher(tab);

  document.getElementById(`runBtn-${tab}`).disabled = false;
}

// Shows exactly the two synthesis variants NOT currently playing, as buttons.
// A variant already generated this session shows "↩ Back to X" (free, no
// recompute — see switchVariant()); one never generated shows "🔄 Try X".
function renderVariantSwitcher(tab) {
  const container = document.getElementById(`variantSwitcher-${tab}`);
  const hint = document.getElementById(`tryAgainHint-${tab}`);
  if (!container) return;

  const state = reviewState[tab];
  const activePipeline = state.activePipeline;
  const infillingSynth = activePipeline == null || activePipeline === 'infilling' || activePipeline === 'full_infilling_knn';
  if (!infillingSynth || !state.variantJobs) {
    container.classList.add('hidden');
    if (hint) hint.hidden = true;
    return;
  }

  container.innerHTML = '';
  SYNTH_VARIANTS.filter(v => v.key !== state.currentVariant).forEach(v => {
    const cached = state.variantJobs[v.key];
    const btn = document.createElement('button');
    btn.className = 'btn-try-again';
    btn.dataset.variant = v.key;
    btn.textContent = cached ? `↩ Back to ${v.label}` : `🔄 Try ${v.label}`;
    btn.onclick = () => switchVariant(tab, v.key);
    container.appendChild(btn);
  });
  container.classList.remove('hidden');
  if (hint) hint.hidden = false;
}

async function switchVariant(tab, variantKey) {
  const state = reviewState[tab];
  const cachedId = state.variantJobs[variantKey];

  if (cachedId) {
    state.currentVariant = variantKey;
    await onJobDone(cachedId, tab); // pure re-render — no new synthesis
    return;
  }

  const variant = SYNTH_VARIANTS.find(v => v.key === variantKey);
  const baseJobId = state.jobId; // any job in this chain carries the same _audio_path/_words/_entities forward
  const container = document.getElementById(`variantSwitcher-${tab}`);
  container.querySelectorAll('button').forEach(b => b.disabled = true);
  const clickedBtn = container.querySelector(`[data-variant="${variantKey}"]`);
  if (clickedBtn) clickedBtn.innerHTML = '<span class="spinner"></span> Synthesizing…';

  setProgressState(tab, 'running', 'Starting…');
  document.getElementById(`jobTitle-${tab}`).textContent = '🔄 Synthesizing…';
  const badge = document.getElementById(`statusBadge-${tab}`);
  badge.className = 'status-badge badge-running';
  badge.style.background = '';
  badge.style.color = '';
  badge.innerHTML = '<span class="spinner"></span> Running';
  document.getElementById(`jobStatus-${tab}`).classList.remove('hidden');

  try {
    const resp = await fetch(`/jobs/${baseJobId}/resynth?pipeline=${variant.pipeline}`, { method: 'POST' });
    if (!resp.ok) {
      showJobError('Re-synthesis failed: ' + await resp.text(), tab);
      renderVariantSwitcher(tab);
      return;
    }
    const { job_id: newJobId } = await resp.json();
    state.variantJobs[variantKey] = newJobId;
    state.jobId = newJobId;
    state.currentVariant = variantKey;

    if (activeEventSource) activeEventSource.close();
    activeEventSource = new EventSource(`/jobs/${newJobId}/stream`);
    activeEventSource.onmessage = e => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'ping') return;
      if (msg.type === 'log')   { appendLog(msg.msg, tab); return; }
      if (msg.type === 'done')  { activeEventSource.close(); onJobDone(newJobId, tab); return; }
      if (msg.type === 'error') { activeEventSource.close(); showJobError(msg.msg, tab); return; }
    };
    activeEventSource.onerror = () => { activeEventSource.close(); showJobError('Lost connection.', tab); };
  } catch (e) {
    showJobError('Error: ' + e.message, tab);
    renderVariantSwitcher(tab);
  }
}

function showJobError(msg, tab) {
  document.getElementById(`jobTitle-${tab}`).textContent = '❌ Failed';
  const badge = document.getElementById(`statusBadge-${tab}`);
  badge.className = 'status-badge badge-error';
  badge.style.background = '';
  badge.style.color = '';
  badge.textContent = '✗ Error';
  setProgressState(tab, 'error', msg);
  document.getElementById(`runBtn-${tab}`).disabled = false;
}

function resetAudioUI(tab) {
  document.getElementById(`audioResult-${tab}`).classList.add('hidden');
  const reviewSection = document.getElementById(`reviewSection-${tab}`);
  if (reviewSection) reviewSection.classList.add('hidden');
  setProgressState(tab, 'running', 'Starting…');
  document.getElementById(`jobTitle-${tab}`).textContent = '⚙️ Running…';
  const badge = document.getElementById(`statusBadge-${tab}`);
  badge.className = 'status-badge badge-running';
  badge.style.background = '';
  badge.style.color = '';
  badge.innerHTML = '<span class="spinner"></span> Running';
  const applyBtn = document.getElementById(`applyBtn-${tab}`);
  if (applyBtn) { applyBtn.disabled = false; applyBtn.textContent = '✓ Apply & Synthesize'; }
  const variantSwitcher = document.getElementById(`variantSwitcher-${tab}`);
  if (variantSwitcher) variantSwitcher.classList.add('hidden');
  const tryAgainHint = document.getElementById(`tryAgainHint-${tab}`);
  if (tryAgainHint) tryAgainHint.hidden = true;
  if (reviewState[tab].variantJobs !== undefined) {
    reviewState[tab].variantJobs = null;
    reviewState[tab].currentVariant = null;
  }
}

// ── Speaker preview playback ──────────────────────────────────────────────────

let _activeSampleAudio = null;
let _activePlayBtn = null;

function _stopSample() {
  if (_activeSampleAudio) {
    _activeSampleAudio.pause();
    _activeSampleAudio.currentTime = 0;
    _activeSampleAudio = null;
  }
  if (_activePlayBtn) {
    _activePlayBtn.textContent = '▶';
    _activePlayBtn.classList.remove('playing');
    _activePlayBtn = null;
  }
}

document.addEventListener('click', e => {
  const btn = e.target.closest('.speaker-play-btn');
  if (!btn) return;
  e.preventDefault();
  e.stopPropagation();

  const src = btn.dataset.src;
  const isPlaying = _activePlayBtn === btn;
  _stopSample();
  if (isPlaying) return;

  const audio = new Audio(src);
  audio.play();
  btn.textContent = '◼';
  btn.classList.add('playing');
  _activeSampleAudio = audio;
  _activePlayBtn = btn;
  audio.onended = _stopSample;
});
