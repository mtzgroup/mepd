// RunInputs TOML profiles: the compute settings a job runs with.
import { html, useEffect, useRef, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { prefs, set, state, toast, useStore } from '../store.js';

export function ProfilesView() {
  const profiles = useStore((s) => s.profiles);
  const summaries = useStore((s) => s.pathSummaries);
  const readOnly = !!useStore((s) => s.demo);
  // Come back to the profile you were editing (not the first in the list).
  const [name, setNameState] = useState(() => {
    const last = prefs.get('profileName', null);
    return profiles.includes(last) ? last : (profiles[0] || null);
  });
  const setName = (n) => { setNameState(n); prefs.set('profileName', n); };
  const [text, setText] = useState('');
  const [saved, setSaved] = useState('');
  const [check, setCheck] = useState(null);
  const [checking, setChecking] = useState(false);
  const [mode, setMode] = useState('form');   // 'form' (choices + Advanced) or 'toml' (the raw file)

  useEffect(() => { if (!name && profiles.length) setName(profiles[0]); }, [profiles]);
  useEffect(() => {
    setCheck(null);
    if (!name) { setText(''); setSaved(''); return; }
    api.get(`/api/profiles/${encodeURIComponent(name)}`).then((t) => { setText(t); setSaved(t); }).catch(() => {});
  }, [name]);

  const dirty = text !== saved;
  useEffect(() => { setCheck(null); }, [text]);   // a Validate result describes the text it checked
  const save = async () => {
    const ok = await attempt(() => api.put(`/api/profiles/${encodeURIComponent(name)}`, { text }), `Saved ${name}`);
    if (ok) setSaved(text);
    return !!ok;
  };
  // Publish the edit state so leaving the tab (or starting a job with this
  // profile) can warn instead of silently using the last saved version.
  useEffect(() => {
    set({ profileEditor: name ? { name, dirty, save } : null });
  }, [name, dirty, text]);
  useEffect(() => () => set({ profileEditor: null }), []);
  useEffect(() => {
    const onUnload = (e) => { if (state.profileEditor?.dirty) { e.preventDefault(); e.returnValue = ''; } };
    window.addEventListener('beforeunload', onUnload);
    return () => window.removeEventListener('beforeunload', onUnload);
  }, []);
  const pick = async (p) => {
    if (p === name) return;
    if (dirty && confirm(`Save your changes to "${name}" before switching?\n\nOK saves. Cancel discards them.`)) {
      if (!(await save())) return;
    }
    setName(p);
  };
  const saveAs = async () => {
    const n = prompt('New profile name (letters, digits, - and _):', `${name || 'profile'}-copy`);
    if (!n) return;
    const ok = await attempt(() => api.put(`/api/profiles/${encodeURIComponent(n)}`, { text }), `Created ${n}`);
    if (ok) { setSaved(text); setName(n); }
  };
  const del = async () => {
    if (!confirm(`Delete profile ${name}?`)) return;
    await attempt(() => api.del(`/api/profiles/${encodeURIComponent(name)}`), 'Deleted');
    setName(null);
  };
  const validate = async () => {
    setChecking(true);
    setCheck(await attempt(() => api.post('/api/profiles/validate', { text })));
    setChecking(false);
  };

  return html`
    <div class="profiles-view">
      <div class="view-head">
        <h2>Compute profiles</h2>
        <p class="small muted">Each profile is a <code>RunInputs</code> TOML (the file <code>mepd … --inputs</code> takes): engine, program and level of theory, path minimizer, optimizer, thresholds. A job copies its profile into its own folder, so editing one later never changes past results.</p>
      </div>
      <div class="profiles-body">
        <ul class="profile-list">
          ${profiles.map((p) => html`<li class=${p === name ? 'on' : ''} onClick=${() => pick(p)}>${p}</li>`)}
        </ul>
        ${name ? html`
          <div class="profile-editor">
            <div class="editor-bar">
              <strong class="mono">${name}.toml</strong>${dirty && html` <span class="badge">unsaved</span>`}
              <span class="spacer"></span>
              ${readOnly ? html`<span class="small muted">read-only in the demo</span>` : html`
              <button class="btn" onClick=${validate} disabled=${checking} title="Builds RunInputs (engine included) from this text in a scratch process">${checking ? 'Checking…' : 'Validate'}</button>
              <button class="btn" onClick=${saveAs}>Save as…</button>
              <button class="btn primary" onClick=${save} disabled=${!dirty}>Save</button>
              <button class="btn danger-outline" onClick=${del} title="Delete this profile">Delete</button>`}
            </div>
            ${summaries[name] && html`<div class="path-summary small">
              <span>Saved version: <b>${summaries[name].text}</b>${dirty ? ' · unsaved edits are not used by jobs until you Save' : ''}</span>
              ${summaries[name].warnings.map((w) => html`<div class="level-note">${w}</div>`)}
            </div>`}
            ${check && html`<pre class=${check.ok ? 'ok-box' : 'error-box'}>${check.ok ? '✓ ' : '✗ '}${check.message}</pre>`}
            <div class="segmented profile-mode">
              <button class=${mode === 'form' ? 'on' : ''} onClick=${() => setMode('form')}>Settings</button>
              <button class=${mode === 'toml' ? 'on' : ''} onClick=${() => setMode('toml')}>TOML</button>
            </div>
            ${mode === 'form'
              ? html`<${ProfileForm} key=${name} text=${text} onText=${setText} readOnly=${readOnly} />`
              : html`<textarea class="toml" spellcheck="false" value=${text} readOnly=${readOnly} onInput=${(e) => setText(e.target.value)}
              onKeyDown=${(e) => { if ((e.metaKey || e.ctrlKey) && e.key === 's') { e.preventDefault(); save(); } }} />`}
          </div>`
        : html`<div class="empty-hint"><p>No profiles yet.</p></div>`}
      </div>
    </div>`;
}

// ------------------------------------------------------------ form
// The profile as a few big choices plus an Advanced section showing only
// the settings those choices use. Every change goes to the server, which
// edits the TOML (keeping everything the form doesn't cover) and returns
// the new text and form; nothing is saved until Save.
function ProfileForm({ text, onText, readOnly }) {
  const [form, setForm] = useState(null);
  const [error, setError] = useState(null);
  const [advanced, setAdvancedState] = useState(() => prefs.get('profileAdvanced', false));
  const setAdvanced = (open) => { setAdvancedState(open); prefs.set('profileAdvanced', open); };
  const applied = useRef(null);   // text we just got back from an apply (its form is already here)

  useEffect(() => {
    if (text === applied.current) return undefined;
    let live = true;
    api.post('/api/profiles/form', { text })
      .then((f) => { if (live) { setForm(f); setError(null); } })
      .catch((e) => { if (live) setError(e.message); });
    return () => { live = false; };
  }, [text]);

  // Changes are applied one at a time, each to the newest text: a quick
  // second change must build on the first, not on the text both started from.
  const latest = useRef(text);
  useEffect(() => { latest.current = text; }, [text]);
  const queue = useRef(Promise.resolve());
  const change = (path, value) => {
    queue.current = queue.current.then(async () => {
      const out = await attempt(() => api.post('/api/profiles/form/apply', { text: latest.current, path, value }));
      if (!out) return false;
      latest.current = out.text;
      applied.current = out.text;
      setForm(out.form);
      onText(out.text);
      return true;
    });
    return queue.current;
  };

  if (error) return html`<p class="error-box">${error} · fix it on the TOML tab.</p>`;
  if (!form) return html`<p class="muted small">Reading the profile…</p>`;
  const method = form.basic.find((c) => c.key === 'path_method')?.value;
  return html`
    <div class="pform">
      ${form.issues.map((i) => html`<p class="level-note small">${i}</p>`)}
      ${(form.notes || []).map((n) => html`<p class="small muted">${n}</p>`)}
      ${form.has_comments && html`<p class="small muted">This profile has # comments: a change made here removes them (edit on the TOML tab to keep them).</p>`}
      <div class="pform-basic">
        ${form.basic.map((c) => html`<${ChoiceField} key=${c.key} choice=${c} readOnly=${readOnly} onChange=${(v) => change(c.key, v)} />`)}
        ${form.level.length > 0 && html`<div class="pfield pform-level">
          <span class="field-label">Level of theory</span>
          <div class="pform-pair">${form.level.map((f) => html`<${Field} key=${f.path} f=${f} bare readOnly=${readOnly} onChange=${change} />`)}</div>
        </div>`}
        <${Field} f=${form.images} readOnly=${readOnly} onChange=${change} />
      </div>
      ${form.unused.length > 0 && html`<p class="level-note small">
        ${form.unused.length} setting${form.unused.length > 1 ? 's' : ''} in [path_min_inputs] ${form.unused.length > 1 ? 'are' : 'is'} not used by ${method}:
        <span class="mono">${form.unused.slice(0, 8).join(', ')}${form.unused.length > 8 ? ', …' : ''}</span>
        ${!readOnly && html` <button class="btn-link small" onClick=${() => change('remove_unused', true)}>Remove ${form.unused.length > 1 ? 'them' : 'it'}</button>`}</p>`}
      <details class="advanced pform-adv" open=${advanced} onToggle=${(e) => setAdvanced(e.target.open)}>
        <summary>Advanced</summary>
        ${form.groups.map((g) => html`
          <fieldset class="pform-group" key=${g.id}>
            <legend>${g.title}</legend>
            ${g.note && html`<p class="small muted">${g.note}</p>`}
            <div class="pform-grid">
              ${g.fields.map((f) => html`<${Field} key=${f.path} f=${f} readOnly=${readOnly} onChange=${change} />`)}
            </div>
          </fieldset>`)}
      </details>
    </div>`;
}

function ChoiceField({ choice, onChange, readOnly }) {
  const current = choice.options.find((o) => o.value === choice.value);
  return html`<label class="pfield">
    <span class="field-label">${choice.label}</span>
    <select value=${choice.value} disabled=${readOnly}
      onChange=${(e) => { const el = e.target; Promise.resolve(onChange(el.value)).then((ok) => { if (ok === false) el.value = choice.value; }); }}>
      ${choice.options.map((o) => html`<option value=${o.value} disabled=${!!o.disabled && o.value !== choice.value}>
        ${o.label}${o.disabled ? ` (${o.disabled})` : ''}</option>`)}
    </select>
    <span class="field-help">${current?.help || choice.help}</span>
  </label>`;
}

function fmtDefault(v) {
  if (v === null || v === undefined || v === '') return 'mepd default';
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  return String(v);
}

function Field({ f, onChange, readOnly, bare = false }) {
  const set = f.value !== null && f.value !== undefined && f.value !== '';
  // mepd writes every default into a profile: only a real change gets a reset link.
  const changed = set && String(f.value) !== String(f.default ?? '');
  const reset = changed && !readOnly
    ? html`<button class="btn-link small pfield-reset" title=${`Back to the default (${fmtDefault(f.default)})`}
        onClick=${(e) => { e.preventDefault(); onChange(f.path, null); }}>↺ default</button>` : null;
  let input;
  if (f.type === 'bool') {
    const on = set ? !!f.value : !!f.default;
    return html`<label class=${`pfield pfield-bool ${f.key ? 'key' : ''}`} title=${f.help}>
      <input type="checkbox" class="switch" checked=${on} disabled=${readOnly} onChange=${(e) => onChange(f.path, e.target.checked)} />
      <span class="field-label">${f.label}</span>${reset}
      ${f.help && html`<span class="field-help">${f.help}</span>`}
    </label>`;
  }
  if (f.type === 'select') {
    const shown = String(set ? f.value : (f.default ?? ''));
    input = html`<select value=${shown} disabled=${readOnly}
      onChange=${(e) => { const el = e.target; Promise.resolve(onChange(f.path, el.value)).then((ok) => { if (ok === false) el.value = shown; }); }}>
      ${f.options.map((o) => html`<option value=${String(o.value)}>${o.label}</option>`)}</select>`;
  } else {
    const numeric = f.type === 'int' || f.type === 'float';
    input = html`<input type=${numeric ? 'number' : 'text'} step=${f.type === 'int' ? '1' : 'any'} value=${set ? f.value : ''}
      placeholder=${f.placeholder || fmtDefault(f.default)} readOnly=${readOnly}
      onChange=${(e) => {
        // A half-typed number reads as '' in the browser: refuse it rather
        // than silently clearing the setting.
        if (numeric && e.target.validity.badInput) {
          toast(`${f.label}: not a valid number`, 'error');
          e.target.value = set ? f.value : '';
          return;
        }
        const v = e.target.value === '' ? null : e.target.value;
        // Rejected (e.g. 1.5 in a whole-number field): show the kept value again.
        Promise.resolve(onChange(f.path, v)).then((ok) => { if (ok === false) e.target.value = set ? f.value : ''; });
      }} />`;
  }
  if (bare) return html`<span class="pform-bare" title=${f.help}>${input}<span class="field-help">${f.label}</span></span>`;
  return html`<label class=${`pfield ${f.key ? 'key' : ''}`} title=${f.help}>
    <span class="field-label">${f.label}${reset}</span>
    ${input}
    ${f.help && html`<span class="field-help">${f.help}</span>`}
  </label>`;
}
