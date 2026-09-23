// RunInputs TOML profiles: the compute settings a job runs with.
import { html, useEffect, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { set, state, useStore } from '../store.js';

export function ProfilesView() {
  const profiles = useStore((s) => s.profiles);
  const summaries = useStore((s) => s.pathSummaries);
  const readOnly = !!useStore((s) => s.demo);
  const [name, setName] = useState(profiles[0] || null);
  const [text, setText] = useState('');
  const [saved, setSaved] = useState('');
  const [check, setCheck] = useState(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => { if (!name && profiles.length) setName(profiles[0]); }, [profiles]);
  useEffect(() => {
    setCheck(null);
    if (!name) { setText(''); setSaved(''); return; }
    api.get(`/api/profiles/${encodeURIComponent(name)}`).then((t) => { setText(t); setSaved(t); }).catch(() => {});
  }, [name]);

  const dirty = text !== saved;
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
              <button class="btn ghost" onClick=${del} title="Delete profile">✕</button>`}
            </div>
            ${summaries[name] && html`<div class="path-summary small">
              <span>Saved version: <b>${summaries[name].text}</b>${dirty ? ' · unsaved edits are not used by jobs until you Save' : ''}</span>
              ${summaries[name].warnings.map((w) => html`<div class="level-note">${w}</div>`)}
            </div>`}
            ${check && html`<pre class=${check.ok ? 'ok-box' : 'error-box'}>${check.ok ? '✓ ' : '✗ '}${check.message}</pre>`}
            <textarea class="toml" spellcheck="false" value=${text} readOnly=${readOnly} onInput=${(e) => setText(e.target.value)}
              onKeyDown=${(e) => { if ((e.metaKey || e.ctrlKey) && e.key === 's') { e.preventDefault(); save(); } }} />
          </div>`
        : html`<div class="empty-hint"><p>No profiles yet.</p></div>`}
      </div>
    </div>`;
}
