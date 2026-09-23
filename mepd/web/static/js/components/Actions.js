// "What can I run on what I selected?" -- lists every operation whose
// target fits the current selection, grouped by intent, each expandable
// into its parameter form. Batch runs fall out naturally: N edges selected
// => one job per edge.
import { html, useEffect, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, prefs, useStore, state } from '../store.js';
import { ParamForm, clampToSchema, defaultsFor } from './ParamForm.js';

// How many jobs an operation would create for this selection (0 = not applicable).
export function applicability(op, sel) {
  const nS = sel.structures.length, nE = sel.edges.length;
  if (op.target === 'pair') {
    if (nE > 0 && nS === 0) return { n: nE, what: nE === 1 ? 'this edge' : `${nE} edges` };
    if (nE === 0 && nS === 2) return { n: 1, what: 'these two structures' };
    return null;
  }
  if (op.target === 'structure') {
    if (nS > 0 && nE === 0) return { n: nS, what: nS === 1 ? 'this structure' : `${nS} structures` };
    return null;
  }
  if (op.target === 'set') {
    if (nS >= op.min_structures && nE === 0) return { n: 1, what: `${nS} structures` };
    return null;
  }
  return null;
}

function ProfilePicker({ value, onChange }) {
  const profiles = useStore((s) => s.profiles);
  const levels = useStore((s) => s.levels);
  const levelProfile = useStore((s) => s.levelProfile);
  const summary = useStore((s) => s.pathSummaries[value ?? '']);
  return html`
    <div class="field">
      <label class="field-label">Compute profile</label>
      <select value=${value || ''} onChange=${(e) => onChange(e.target.value || null)}>
        <option value="">mepd built-in defaults · ${levels['']?.label ?? ''}</option>
        ${profiles.map((p) => html`<option value=${p}>${p} · ${levels[p]?.label ?? ''}${p === levelProfile ? ' (workspace level)' : ''}</option>`)}
      </select>
      ${summary && html`<span class="field-help">Path search: <b>${summary.text}</b></span>`}
      ${summary?.warnings.map((w) => html`<p class="level-note small">${w}</p>`)}
      <span class="field-help">RunInputs TOML: engine, level of theory, path-minimizer settings. Edit under Profiles.</span>
    </div>`;
}

function OperationCard({ op, sel, fit, open, onToggle }) {
  const [values, setValues] = useState(() => clampToSchema(op.schema, { ...defaultsFor(op.schema), ...prefs.get(`params:${op.key}`, {}) }));
  const [profile, setProfile] = useState(() => prefs.get('profile', state.profiles.includes('default') ? 'default' : null));
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState(null);

  useEffect(() => { setPreview(null); }, [sel, values, profile]);

  // Drop remembered keys the schema no longer has.
  const clean = (v) => Object.fromEntries(Object.entries(v).filter(([k]) => k in (op.schema?.properties || {})));
  const body = (dry) => ({
    op: op.key, structures: sel.structures, edges: sel.edges,
    params: clean(values), profile, dry_run: dry,
  });

  const run = async () => {
    setBusy(true);
    prefs.set(`params:${op.key}`, clean(values));
    prefs.set('profile', profile);
    const jobs = await attempt(() => api.post('/api/jobs', body(false)),
      (js) => (js.length === 1 ? `Queued: ${js[0].title}` : `Queued ${js.length} jobs`));
    setBusy(false);
    if (jobs?.length === 1) openJob(jobs[0].id);
  };
  const showCommand = async () => {
    const out = await attempt(() => api.post('/api/jobs', body(true)));
    if (out) setPreview(out.map((j) => j.command));
  };

  // Level-of-theory consistency between the chosen profile and the targets.
  const levels = useStore((s) => s.levels);
  const structures = useStore((s) => s.workspace.structures);
  const edges = useStore((s) => s.workspace.edges);
  const targetIds = sel.edges.length
    ? sel.edges.flatMap((id) => (edges[id] ? [edges[id].source, edges[id].target] : []))
    : sel.structures;
  const jobKey = levels[profile ?? '']?.key;
  const offLevel = targetIds.map((id) => structures[id]).filter((r) => r && !(r.optimized && r.level?.key === jobKey));
  const levelNote = !offLevel.length || op.key === 'optimize' || op.key === 'channels' ? null
    : op.target === 'pair'
      ? `${offLevel.length} endpoint(s) are not minima at ${levels[profile ?? '']?.label ?? 'this level'}; they will be re-minimized at this profile first, so the whole path is on one surface.`
      : op.key === 'tsopt' ? null
        : `${offLevel.map((r) => r.name).join(', ')} ${offLevel.length > 1 ? 'are' : 'is'} not a minimum at ${levels[profile ?? '']?.label ?? 'this level'}. Optimize at this profile first, or results will mix levels of theory.`;

  if (!op.available) {
    return html`<div class="op-card disabled" title=${op.unavailable_reason}>
      <div class="op-head"><span class="op-title">${op.title}</span><span class="badge muted">not available</span></div>
      <p class="small muted">${op.unavailable_reason}</p>
    </div>`;
  }
  return html`
    <div class=${`op-card ${open ? 'open' : ''}`}>
      <button type="button" class="op-head" onClick=${onToggle} aria-expanded=${open}>
        <span class="op-title">${op.title}</span>
        <span class="op-chevron">${open ? '▾' : '▸'}</span>
      </button>
      <p class="op-summary small">${op.summary}</p>
      ${open && html`
        <div class="op-body">
          <${ParamForm} schema=${op.schema} values=${values} onChange=${setValues} />
          <${ProfilePicker} value=${profile} onChange=${setProfile} />
          ${levelNote && html`<p class="level-note small">${levelNote}</p>`}
          <div class="op-run">
            <button class="btn primary" disabled=${busy} onClick=${run}>
              ${busy ? 'Queuing…' : fit.n > 1 ? `Run on ${fit.what} (${fit.n} jobs)` : `Run on ${fit.what}`}
            </button>
            <button class="btn-link small" onClick=${showCommand}>Show command</button>
          </div>
          ${preview && html`<pre class="cmd-preview">${preview.join('\n\n')}</pre>`}
        </div>`}
    </div>`;
}

export function ActionPanel({ sel }) {
  const operations = useStore((s) => s.operations);
  const [openKey, setOpenKey] = useState(null);
  const fits = operations.map((op) => ({ op, fit: applicability(op, sel) })).filter((x) => x.fit);
  if (!fits.length) return null;
  const byCat = {};
  for (const x of fits) (byCat[x.op.category] ||= []).push(x);
  return html`
    <section class="actions">
      ${Object.entries(byCat).map(([cat, items]) => html`
        <div class="action-group">
          <h3 class="section-title">${cat}</h3>
          ${items.map(({ op, fit }) => html`<${OperationCard} key=${op.key} op=${op} sel=${sel} fit=${fit}
              open=${openKey === op.key} onToggle=${() => setOpenKey(openKey === op.key ? null : op.key)} />`)}
        </div>`)}
    </section>`;
}
