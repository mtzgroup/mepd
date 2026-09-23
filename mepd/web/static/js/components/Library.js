// Left sidebar: add structures (SMILES / xyz text / dropped files) and
// browse the library. Library entries and graph nodes are the same thing.
import { html, useState } from '../lib.js';
import { api, attempt, refreshState } from '../api.js';
import { prefs, select, useStore } from '../store.js';
import { cls, depictUrl, levelStatus } from '../util.js';

export async function uploadFiles(files, { charge = null, multiplicity = null } = {}) {
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  fd.append('optimize', prefs.get('optimizeOnAdd', true) ? 'true' : 'false');
  if (charge !== null && charge !== '') fd.append('charge', charge);
  if (multiplicity !== null && multiplicity !== '') fd.append('multiplicity', multiplicity);
  return attempt(() => api.post('/api/structures/upload', fd),
    (added) => `Added ${added.length} structure${added.length === 1 ? '' : 's'}`);
}

function AddBox() {
  const [text, setText] = useState('');
  const [charge, setCharge] = useState('');
  const [mult, setMult] = useState('');
  const [busy, setBusy] = useState(false);
  const [optimize, setOptimize] = useState(() => prefs.get('optimizeOnAdd', true));
  const validate = useStore((s) => s.validateMinima);
  const level = useStore((s) => s.levels[s.levelProfile ?? '']);
  const isXyz = /^\s*\d+\s*\n/.test(text);
  const n = isXyz ? null : text.split('\n').filter((l) => l.trim() && !l.trim().startsWith('#')).length;

  const add = async () => {
    if (!text.trim()) return;
    setBusy(true);
    const added = await attempt(() => api.post('/api/structures', {
      text, charge: charge === '' ? null : +charge, multiplicity: mult === '' ? null : +mult, optimize,
    }), (a) => `Added ${a.length} structure${a.length === 1 ? '' : 's'}${optimize ? ` · optimizing at ${level?.label ?? 'the workspace level'}` : ''}`);
    setBusy(false);
    if (added) {
      setText('');
      select({ structures: added.map((a) => a.id) });
    }
  };
  return html`
    <div class="add-box">
      <textarea rows="3" value=${text} spellcheck="false"
        placeholder=${'SMILES (one per line, optional name after a space)\nor paste XYZ — or drop .xyz/.smi files here'}
        onInput=${(e) => setText(e.target.value)}
        onKeyDown=${(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) add(); }} />
      <div class="add-row">
        <input type="number" class="tiny" placeholder="charge" value=${charge} onInput=${(e) => setCharge(e.target.value)} title="Charge (default: from SMILES / xyz)" />
        <input type="number" class="tiny" placeholder="mult." min="1" value=${mult} onInput=${(e) => setMult(e.target.value)} title="Spin multiplicity" />
        <button class="btn primary" disabled=${busy || !text.trim()} onClick=${add}>
          ${busy ? 'Embedding…' : isXyz ? 'Add XYZ' : n > 1 ? `Add ${n}` : 'Add'}
        </button>
      </div>
      <label class="small opt-toggle" title="Minimize at the workspace level of theory so every structure in the graph lives on the same potential energy surface">
        <input type="checkbox" checked=${optimize} onChange=${(e) => { setOptimize(e.target.checked); prefs.set('optimizeOnAdd', e.target.checked); }} />
        <span>Optimize at <b>${level?.label ?? 'workspace level'}</b> on add${validate ? ' + Hessian check' : ''}</span>
      </label>
    </div>`;
}

export function LevelChip({ rec }) {
  useStore((s) => [s.levelProfile, s.levels]);
  const st = levelStatus(rec);
  return html`<span class=${`level-chip level-${st.kind}`} title=${st.title}>${st.text}</span>`;
}

function LevelBar() {
  const levelProfile = useStore((s) => s.levelProfile);
  const profiles = useStore((s) => s.profiles);
  const levels = useStore((s) => s.levels);
  const validate = useStore((s) => s.validateMinima);
  const structures = useStore((s) => s.workspace.structures);
  // TS structures (kind 'ts') are never counted: minimizing them would destroy them.
  const off = Object.values(structures).filter((r) => ['other', 'none', 'failed'].includes(levelStatus(r, levels[levelProfile ?? '']?.key).kind));
  const busy = Object.values(structures).filter((r) => r.status === 'optimizing').length;
  const setLevel = (profile) => attempt(() => api.put('/api/level', { profile: profile || null }));
  const reopt = () => attempt(() => api.post('/api/structures/reoptimize', { structures: off.map((r) => r.id) }),
    (js) => `Re-optimizing ${off.length} structure${off.length > 1 ? 's' : ''} (${js.length} job${js.length > 1 ? 's' : ''})`);
  return html`
    <div class="level-bar">
      <label class="small" title="The level of theory every structure in this workspace is minimized at. Change the profile's engine/method under Profiles.">
        Level of theory
        <select value=${levelProfile ?? ''} onChange=${(e) => setLevel(e.target.value)}>
          ${profiles.map((p) => html`<option value=${p}>${p} · ${levels[p]?.label ?? ''}</option>`)}
          <option value="">mepd defaults · ${levels['']?.label ?? ''}</option>
        </select>
      </label>
      <label class="small hess-toggle" title="After optimizing a structure, compute its Hessian and require no imaginary frequency. A structure stuck on a saddle point is pushed along its unstable mode and re-optimized; if that fails it is flagged 'not a minimum'.">
        <input type="checkbox" checked=${validate}
          onChange=${(e) => attempt(() => api.put('/api/level', { validate_minima: e.target.checked }))
            .then((r) => r && refreshState())} />
        Verify minima (Hessian)
      </label>
      ${busy > 0 && html`<span class="small muted">${busy} optimizing…</span>`}
      ${off.length > 0 && html`<button class="btn small warn-outline" onClick=${reopt}
        title="Minimize every structure that is not a minimum at this level (force-field embeddings, other levels, failed optimizations)">
        Re-optimize ${off.length} off-level</button>`}
    </div>`;
}

function Card({ rec, selected, order }) {
  const url = depictUrl(rec.smiles, 180, 110);
  const origin = rec.origin?.kind;
  return html`
    <li class=${cls('card', selected && 'selected')} draggable="false"
      onClick=${(e) => select({ structures: [rec.id] }, e.shiftKey || e.metaKey || e.ctrlKey)}
      title=${rec.smiles || rec.formula}>
      <div class="card-img">
        ${url ? html`<img src=${url} alt="" loading="lazy" />` : html`<span class="muted small">${rec.formula}</span>`}
        ${order > 0 && html`<span class="order-badge" title="Selection order (1 = start)">${order}</span>`}
      </div>
      <div class="card-body">
        <div class="card-name">${rec.name}</div>
        <div class="card-meta">
          <span>${rec.formula}</span>
          ${(rec.charge !== 0 || rec.multiplicity !== 1) && html`<span class="badge">${rec.charge >= 0 ? '+' : ''}${rec.charge} / ${rec.multiplicity}</span>`}
          <span class=${`origin origin-${origin}`} title=${origin === 'job' ? `From ${rec.origin.label}` : `Entered as ${origin}`}>
            ${origin === 'job' ? 'result' : origin}</span>
          <${LevelChip} rec=${rec} />
        </div>
      </div>
    </li>`;
}

export function Library() {
  const structures = useStore((s) => s.workspace.structures);
  const selected = useStore((s) => s.selection.structures);
  const [q, setQ] = useState('');
  const [drag, setDrag] = useState(false);
  const list = Object.values(structures)
    .filter((r) => !q || `${r.name} ${r.smiles} ${r.formula}`.toLowerCase().includes(q.toLowerCase()))
    .sort((a, b) => b.created - a.created);

  const onDrop = (e) => {
    e.preventDefault();
    setDrag(false);
    if (e.dataTransfer.files.length) uploadFiles([...e.dataTransfer.files]);
  };
  return html`
    <aside class=${cls('library', drag && 'drag')}
      onDragOver=${(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave=${(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setDrag(false); }}
      onDrop=${onDrop}>
      <div class="pane-head">
        <h2>Structures <span class="count">${Object.keys(structures).length}</span></h2>
      </div>
      <${LevelBar} />
      <${AddBox} />
      ${Object.keys(structures).length > 6 && html`
        <input class="search" type="search" placeholder="Filter by name, SMILES, formula" value=${q} onInput=${(e) => setQ(e.target.value)} />`}
      ${list.length === 0 && !q
        ? html`<div class="empty-hint">
            <p><strong>Start here.</strong> Add reactants, products or any structure you want to explore.</p>
            <p class="small muted">Select one structure to explore around it, two (or an edge) to connect them, or several to batch.</p>
          </div>`
        : html`<ul class="cards">
            ${list.map((r) => html`<${Card} key=${r.id} rec=${r} selected=${selected.includes(r.id)}
              order=${selected.length > 1 ? selected.indexOf(r.id) + 1 : 0} />`)}
          </ul>`}
      <div class="drop-overlay">Drop .xyz / .smi files to add</div>
    </aside>`;
}
