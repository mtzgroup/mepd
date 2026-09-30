// Analyze: the workspace's reactions as a list to compare, filter and act on
// (Explore is the map; this is where barriers are looked at and asked for).
//  * reactions are grouped by what they do (net reactants -> net products),
//    so parallel routes -- direct, water-shuttled, ... -- sit together with
//    their barriers;
//  * the selected reaction's card: species, energy ladder, complexes, MD
//    event, TS;
//  * New reaction: pick species from the graph as reactants (and products,
//    or let the bond rules propose them) and turn them into reactions.
import { html, useMemo, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { set, state, useStore } from '../store.js';
import { depictUrl, edgeStatus, fmtKcal } from '../util.js';
import { NO_TS_SHORT, ReactionCard, findTs, noTsWhy } from './Reactions.js';

const count = (ids) => ids.reduce((m, id) => ({ ...m, [id]: (m[id] || 0) + 1 }), {});

// What a reaction does, shuttles taken out: the key its parallel routes share.
function netKey(r) {
  const sh = count(r.shuttles || []);
  const side = (ids) => {
    const c = count(ids);
    return Object.keys(c).sort().flatMap((id) => Array(Math.max(0, c[id] - (sh[id] || 0))).fill(id)).join('+');
  };
  return `${side(r.reactants)}>${side(r.products)}`;
}

function netLabel(r, structures) {
  const sh = count(r.shuttles || []);
  const side = (ids) => {
    const c = count(ids);
    return Object.entries(c).filter(([id, n]) => n - (sh[id] || 0) > 0)
      .map(([id, n]) => `${n - (sh[id] || 0) > 1 ? `${n - (sh[id] || 0)} ` : ''}${structures[id]?.name || '?'}`).join(' + ');
  };
  return `${side(r.reactants)} → ${side(r.products)}`;
}

function routeLabel(r, structures) {
  const sh = [...new Set(r.shuttles || [])].map((id) => structures[id]?.name || '?');
  return sh.length ? `via ${sh.join(', ')}` : 'direct';
}

function tsState(r, edges, jobs) {
  const edge = r.edge ? edges[r.edge] : null;
  if (!edge) return { kind: 'none', text: NO_TS_SHORT[r.complex_reason] || 'no TS endpoints', title: noTsWhy(r) };
  const st = edgeStatus(edge, jobs);
  if (st.status === 'running' || st.status === 'queued') return { kind: 'busy', text: `${st.status}…` };
  if (st.barrier != null) return { kind: 'ts', text: `ΔE‡ ${fmtKcal(st.barrier)}`, v: st.barrier };
  if (st.barrierUnverified != null) return { kind: 'unverified', text: `≈${fmtKcal(st.barrierUnverified)}?`, v: st.barrierUnverified, title: 'path maximum; no TS/IRC confirmed it' };
  return { kind: 'todo', text: 'no TS yet' };
}

// ------------------------------------------------------------ new reaction
function SpeciesPicker({ label, ids, onChange, species }) {
  const [q, setQ] = useState('');
  const [openList, setOpenList] = useState(false);
  const ql = q.trim().toLowerCase();
  // Exact name/SMILES first, then prefixes, then the rest (shortest first).
  const rank = (s) => {
    const names = [s.name, s.smiles].map((x) => (x || '').toLowerCase());
    return names.includes(ql) ? 0 : names.some((x) => x.startsWith(ql)) ? 1 : 2;
  };
  const hits = ql && openList ? species.filter((s) => `${s.name} ${s.smiles} ${s.formula}`.toLowerCase().includes(ql))
    .sort((a, b) => rank(a) - rank(b) || (a.name || '').length - (b.name || '').length).slice(0, 8) : [];
  const c = count(ids);
  return html`<div class="cp-side">
    <div class="section-title">${label}</div>
    <div class="cp-chips">
      ${Object.entries(c).map(([id, n]) => html`<span class="cp-chip">
        ${state.workspace.structures[id]?.smiles && html`<img src=${depictUrl(state.workspace.structures[id].smiles, 60, 45)} alt="" />`}
        <span>${n > 1 ? `${n} × ` : ''}${state.workspace.structures[id]?.name || '?'}</span>
        <button class="btn-icon" title="One more" onClick=${() => onChange([...ids, id])}>+</button>
        <button class="btn-icon" title="One fewer" onClick=${() => { const k = ids.indexOf(id); onChange(ids.filter((_, i) => i !== k)); }}>−</button>
      </span>`)}
      <div class="cp-search">
        <input type="search" placeholder="add a species…" value=${q}
          onInput=${(e) => { setQ(e.target.value); setOpenList(true); }} onFocus=${() => setOpenList(true)}
          onBlur=${() => setTimeout(() => setOpenList(false), 150)}
          onKeyDown=${(e) => { if (e.key === 'Escape') setOpenList(false); if (e.key === 'Enter' && hits[0]) { onChange([...ids, hits[0].id]); setQ(''); } }} />
        ${hits.length > 0 && html`<ul class="cp-hits">${hits.map((s) => html`<li><button onMouseDown=${(e) => e.preventDefault()} onClick=${() => { onChange([...ids, s.id]); setQ(''); setOpenList(false); }}>
          ${s.smiles && html`<img src=${depictUrl(s.smiles, 60, 45)} alt="" />`}<span>${s.name}</span></button></li>`)}</ul>`}
      </div>
    </div>
  </div>`;
}

function Composer({ initial, onDone }) {
  const structures = useStore((s) => s.workspace.structures);
  const species = useMemo(() => Object.values(structures).filter((s) => s.role !== 'complex' && s.role !== 'ts')
    .sort((a, b) => a.name.localeCompare(b.name)), [structures]);
  const [reactants, setReactants] = useState(initial || []);
  const [products, setProducts] = useState([]);
  const [proposals, setProposals] = useState(null);
  const [picked, setPicked] = useState(new Set());
  const [busy, setBusy] = useState(false);
  const propose = async () => {
    setBusy(true);
    const out = await attempt(() => api.post('/api/reactions/propose', { reactants }));
    setBusy(false);
    if (out) { setProposals(out); setPicked(new Set()); }
  };
  const create = async () => {
    setBusy(true);
    const made = [];
    if (products.length) {
      const r = await attempt(() => api.post('/api/reactions/compose', { reactants, products }), 'Reaction added');
      if (r) made.push(r);
    } else {
      for (const k of picked) {
        const p = proposals.proposals[k];
        const r = await attempt(() => api.post('/api/reactions/compose', { reactants,
          proposal: { ...p, complex_xyz: proposals.complex_xyz, charge: proposals.charge, multiplicity: proposals.multiplicity } }));
        if (r) made.push(r);
      }
    }
    setBusy(false);
    if (made.length) onDone(made[made.length - 1].id);
  };
  const shuttleNote = reactants.some((id) => products.includes(id));
  return html`<div class="composer">
    <p class="small muted">Put species from your graph together. Add a partner on both sides to use it as a shuttle
      (e.g. water relaying a proton). Leave the products empty to have them proposed.</p>
    <div class="cp-row">
      <${SpeciesPicker} label="Reactants" ids=${reactants} onChange=${(v) => { setReactants(v); setProposals(null); }} species=${species} />
      <span class="cp-arrow">→</span>
      <${SpeciesPicker} label="Products (optional)" ids=${products} onChange=${(v) => { setProducts(v); setProposals(null); }} species=${species} />
    </div>
    ${shuttleNote && html`<p class="small muted">Species on both sides act as shuttles.</p>`}
    <div class="cp-actions">
      ${products.length
        ? html`<button class="btn primary" disabled=${busy || !reactants.length} onClick=${create}>Create reaction</button>`
        : html`<button class="btn primary" disabled=${busy || !reactants.length} onClick=${propose}>${busy ? 'Proposing…' : 'Propose products'}</button>`}
      <button class="btn-link small" onClick=${() => onDone(null)}>Cancel</button>
    </div>
    ${proposals && !products.length && html`<div class="cp-proposals">
      <div class="section-title">${proposals.proposals.length} possible products (bond rules: up to 2 bonds broken and 2 formed)</div>
      ${!proposals.proposals.length && html`<p class="small muted">None.</p>`}
      <ul>${proposals.proposals.map((p, k) => html`<li>
        <label class=${picked.has(k) ? 'on' : ''}>
          <input type="checkbox" checked=${picked.has(k)} onChange=${(e) => { const n = new Set(picked); e.target.checked ? n.add(k) : n.delete(k); setPicked(n); }} />
          ${p.fragments.map((f, i) => html`${i ? html`<span class="rc-op">+</span>` : ''}<img src=${depictUrl(f, 90, 68)} alt=${f} title=${f} />`)}
          <span class="mono small">${p.smiles}</span>
        </label></li>`)}</ul>
      <button class="btn primary" disabled=${busy || !picked.size} onClick=${create}>
        ${busy ? 'Creating…' : `Create ${picked.size || ''} reaction${picked.size === 1 ? '' : 's'}`}</button>
    </div>`}
  </div>`;
}

// ------------------------------------------------------------ the view
export function AnalyzeView() {
  const ws = useStore((s) => s.workspace);
  const jobs = useStore((s) => s.jobs);
  const nav = useStore((s) => s.analyze) || {};
  const [q, setQ] = useState('');
  const [only, setOnly] = useState('all');   // all | todo | ts | shuttle
  const structures = ws.structures, edges = ws.edges;
  const all = Object.values(ws.reactions || {});
  const filterIds = nav.species || null;      // from Explore: reactions of these species
  const shown = all.filter((r) => {
    const ids = [...r.reactants, ...r.products];
    if (filterIds && !filterIds.every((id) => ids.includes(id))) return false;
    if (q) {
      const text = ids.map((id) => `${structures[id]?.name} ${structures[id]?.smiles}`).join(' ').toLowerCase();
      if (!text.includes(q.toLowerCase())) return false;
    }
    const t = tsState(r, edges, jobs);
    if (only === 'todo' && t.kind !== 'todo') return false;
    if (only === 'ts' && !['ts', 'unverified'].includes(t.kind)) return false;
    if (only === 'shuttle' && !(r.shuttles || []).length) return false;
    return true;
  });
  const groups = {};
  for (const r of shown) (groups[netKey(r)] ||= []).push(r);
  const ordered = Object.values(groups).map((rs) => rs.sort((a, b) => (tsState(a, edges, jobs).v ?? 1e9) - (tsState(b, edges, jobs).v ?? 1e9)))
    .sort((a, b) => b.length - a.length || (tsState(a[0], edges, jobs).v ?? 1e9) - (tsState(b[0], edges, jobs).v ?? 1e9));
  const selected = nav.reaction && ws.reactions?.[nav.reaction] ? ws.reactions[nav.reaction] : null;
  const pick = (rid) => set({ analyze: { ...nav, reaction: rid, compose: false } });
  const todo = shown.filter((r) => tsState(r, edges, jobs).kind === 'todo');
  const findAll = async () => {
    if (!window.confirm(`Queue ${todo.length} TS search${todo.length > 1 ? 'es' : ''}? They run a few at a time.`)) return;
    for (const r of todo) await findTs(r);
  };

  if (!all.length && !nav.compose) {
    return html`<div class="analyze empty-hint">
      <h3>No reactions yet</h3>
      <p>Run the <b>Nanoreactor</b> on some molecules (Explore › select them › Reaction network expansion), or compose a reaction
        from species you already have.</p>
      <button class="btn primary" onClick=${() => set({ analyze: { ...nav, compose: true } })}>New reaction</button>
    </div>`;
  }
  return html`<div class="analyze">
    <div class="an-list">
      <div class="an-bar">
        <input type="search" placeholder="Filter by species" value=${q} onInput=${(e) => setQ(e.target.value)} />
        <div class="segmented">
          ${[['all', 'All'], ['todo', 'No TS yet'], ['ts', 'With TS'], ['shuttle', 'Shuttled']].map(([k, l]) => html`
            <button class=${only === k ? 'on' : ''} onClick=${() => setOnly(k)}>${l}</button>`)}
        </div>
        <button class="btn primary small" onClick=${() => set({ analyze: { ...nav, compose: true, reaction: null } })}>New reaction</button>
      </div>
      ${filterIds && html`<p class="small an-filter">Reactions of ${filterIds.map((id) => structures[id]?.name || '?').join(' + ')}
        <button class="btn-link small" onClick=${() => set({ analyze: { ...nav, species: null } })}>show all</button></p>`}
      <div class="small muted an-count">${shown.length} reaction${shown.length === 1 ? '' : 's'} · ${ordered.length} transformation${ordered.length === 1 ? '' : 's'}
        ${todo.length > 0 && html` · <button class="btn-link small" onClick=${findAll}>find TS for the ${todo.length} without one</button>`}</div>
      <div class="an-groups">
        ${ordered.map((rs) => html`<div class="an-group">
          <div class="an-net">${netLabel(rs[0], structures)}${rs.length > 1 ? html` <span class="small muted">· ${rs.length} routes</span>` : ''}</div>
          ${rs.map((r) => {
            const t = tsState(r, edges, jobs);
            return html`<button class=${`an-route ${selected?.id === r.id ? 'on' : ''}`} onClick=${() => pick(r.id)}>
              <span class="an-via">${routeLabel(r, structures)}</span>
              <span class=${`an-ts ${t.kind}`} title=${t.title || ''}>${t.text}</span>
              <span class="an-de mono small">${r.delta_e_kcal != null ? `ΔE ${r.delta_e_kcal >= 0 ? '+' : ''}${fmtKcal(r.delta_e_kcal)}` : ''}</span>
              <span class="an-src small muted">${r.origin?.kind === 'composed' ? 'composed' : `seen ${r.count || 0}×`}</span>
            </button>`;
          })}
        </div>`)}
      </div>
    </div>
    <div class="an-detail">
      ${nav.compose ? html`<${Composer} initial=${nav.composeReactants} onDone=${(rid) => set({ analyze: { ...nav, compose: false, composeReactants: null, reaction: rid || nav.reaction } })} />`
        : selected ? html`<${ReactionCard} key=${selected.id} r=${selected} />`
          : html`<div class="empty-hint"><p>Pick a reaction on the left: routes that do the same thing are grouped, the lowest barrier first.</p></div>`}
    </div>
  </div>`;
}

// From Explore: open Analyze on a reaction, on the reactions of some
// species, or with the composer started from them.
export function openAnalyze(patch) {
  set({ analyze: { ...(state.analyze || {}), compose: false, ...patch }, view: { tab: 'analyze', jobId: state.view.jobId } });
}
