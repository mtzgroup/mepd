// Nanoreactor reactions, wherever they are shown (a job's results, its live
// reactor, the Inspector): what each one's TS search stands at, and one
// click to start it on the reaction's subsystem (the hidden edge between
// its optimized reactant and product sides).
import { html, useEffect, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, select, set, state, useStore } from '../store.js';
import { depictUrl, edgeStatus, fmtKcal } from '../util.js';
import { defaultProfile } from './Actions.js';
import { Viewer3D } from './Viewer3D.js';
import { EventPlayer } from './ReactorLive.js';
import { openAnalyze } from './Analyze.js';

export function reactionOf(workspace, jobId, index) {
  return Object.values(workspace.reactions || {}).find((r) => r.origin?.job === jobId && r.origin?.index === index) || null;
}

// Why a reaction has no TS endpoints (its optimized reactant or product
// complex changed bonds), short for tables, in full for tooltips and cards.
export const NO_TS_SHORT = {
  barrierless: 'barrierless', recombines: 'barrierless (recombines)', reverts: 'unstable product',
  falls_apart: 'falls apart', rearranges: 'rearranges', failed: 'optimization failed',
};
export function noTsWhy(r) {
  const why = r.complex_error && !/no instance kept/.test(r.complex_error) ? r.complex_error
    : 'its reactant or product complex changed bonds when optimized';
  return `No TS search: ${why}.`;
}

export async function findTs(r) {
  if (!r?.edge) return null;
  return attempt(() => api.post('/api/jobs', { op: 'ts', edges: [r.edge], params: {}, profile: defaultProfile() }),
    'TS search queued on the reaction’s subsystem');
}

export function showInExplore(r) {
  if (!r?.edge) return;
  select({ edges: [r.edge] });
  set({ view: { tab: 'graph', jobId: null } });
}

// "TS 31.2" / "running…" / a Find TS button: what can be done about this
// reaction's barrier right now. `r` is the workspace reaction (or null).
export function TsCell({ r, why = 'not in Explore (deleted?)' }) {
  useStore((s) => s.jobs);   // re-render as its searches run
  const edges = useStore((s) => s.workspace.edges);
  if (!r) return html`<span class="small muted" title=${why}>—</span>`;
  if (!r.edge || !edges[r.edge]) {
    return html`<span class="small muted" title=${noTsWhy(r)}>${NO_TS_SHORT[r.complex_reason] || 'no TS endpoints'}</span>`;
  }
  const st = edgeStatus(edges[r.edge], state.jobs);
  if (st.status === 'running' || st.status === 'queued') return html`<span class="small">${st.status}…</span>`;
  const label = st.barrier != null ? `ΔE‡ ${fmtKcal(st.barrier)}` : st.barrierUnverified != null ? `≈${fmtKcal(st.barrierUnverified)}?` : null;
  return html`<span class="ts-cell">
    ${label && (st.barrierJob
    ? html`<a href="#" class="mono" onClick=${(e) => { e.preventDefault(); openJob(st.barrierJob); }}
        title=${`${st.barrier != null ? 'Barrier from the reactant side, TS + IRC verified' : 'Path maximum; no TS/IRC confirmed it'}. Open its TS search (TS, IRC, sample more paths)`}>${label}</a>`
    : html`<span class="mono" title=${st.barrier != null ? 'Barrier from the reactant side, TS + IRC verified' : 'Path maximum; no TS/IRC confirmed it'}>${label}</span>`)}
    <button class="btn small" onClick=${() => findTs(r)} title="Path search, TS optimization and IRC on only this reaction’s molecules">
      ${label ? 'Search again' : 'Find TS'}</button>
  </span>`;
}

// The reactions of one nanoreactor job, at the top of its results: click a
// row to open its reaction card underneath.
export function ReactionTable({ job, reactions }) {
  const ws = useStore((s) => s.workspace);
  const [open, setOpen] = useState(null);
  if (!reactions.length) return null;
  const openRx = open == null ? null : reactionOf(ws, job.id, open);
  return html`<div class="rx-table-wrap">
    <p class="small muted">Click a reaction to see its species, energies, reaction complex and MD event, and to find its TS.</p>
    <table class="rx-table">
      <thead><tr><th>Reaction</th><th title="Products minus reactants, each optimized alone">ΔE</th><th>Seen</th><th>TS</th></tr></thead>
      <tbody>${reactions.map((rx) => {
        const r = reactionOf(ws, job.id, rx.id);
        return html`<tr class=${`clickable ${open === rx.id ? 'on' : ''}`} onClick=${() => setOpen(open === rx.id ? null : rx.id)}>
          <td class="rx-eq">${rx.label}</td>
          <td class="mono">${rx.delta_e_kcal != null ? `${rx.delta_e_kcal >= 0 ? '+' : ''}${fmtKcal(rx.delta_e_kcal)}` : '—'}</td>
          <td class="mono">${rx.count}${rx.reverse_count ? `/${rx.reverse_count}` : ''}</td>
          <td onClick=${(e) => e.stopPropagation()}><${TsCell} r=${r} /></td>
        </tr>
        ${open === rx.id && html`<tr class="rx-card-row"><td colspan="4">
          ${openRx ? html`<${ReactionCard} key=${openRx.id} r=${openRx} />` : html`<p class="small muted">This reaction is not in Explore (deleted?).</p>`}
        </td></tr>`}`;
      })}</tbody>
    </table>
  </div>`;
}

// ------------------------------------------------------------ reaction card
// Everything about one reaction in one place: what reacts (structures),
// how the energy goes (separated -> complex -> TS -> complex -> separated,
// all over the same atoms), the reaction complex in 3D (optimized ends,
// the MD event, the TS), and its TS search.

function speciesTiles(ids, structures, shuttles) {
  const count = {};
  ids.forEach((id) => { count[id] = (count[id] || 0) + 1; });
  return Object.entries(count).map(([id, n]) => ({ id, n, rec: structures[id], shuttle: shuttles.has(id) }));
}

function Tile({ t }) {
  const img = t.rec?.smiles ? depictUrl(t.rec.smiles, 120, 90) : null;
  return html`<button class=${`rc-tile ${t.shuttle ? 'shuttle' : ''}`} title=${`${t.rec?.name || '?'}${t.shuttle ? ' (shuttle: comes out unchanged)' : ''} — select in Explore`}
    onClick=${() => { select({ structures: [t.id] }); set({ view: { tab: 'graph', jobId: null } }); }}>
    ${img ? html`<img src=${img} alt=${t.rec?.name} />` : html`<span class="mono">${t.rec?.name || '?'}</span>`}
    <span class="rc-name">${t.n > 1 ? `${t.n} × ` : ''}${t.rec?.name || '?'}</span>
    ${t.shuttle && html`<span class="rc-badge">shuttle</span>`}
  </button>`;
}

function Equation({ r, structures }) {
  const shuttles = new Set(r.shuttles || []);
  const side = (ids) => speciesTiles(ids, structures, shuttles);
  const join = (tiles) => tiles.map((t, i) => html`${i ? html`<span class="rc-op">+</span>` : ''}<${Tile} t=${t} />`);
  return html`<div class="rc-eq">${join(side(r.reactants))}<span class="rc-op arrow">→</span>${join(side(r.products))}</div>`;
}

// Energy levels, kcal/mol from the separated reactants.
// Energies from the graph itself (composed reactions, or when the run's own
// are missing): every species and both complexes at one level of theory.
function ladderFromGraph(r, structures) {
  const recs = [...r.reactants, ...r.products].map((id) => structures[id]);
  const cx = (r.complexes || []).map((id) => structures[id]);
  const all = [...recs, ...cx].filter(Boolean);
  if (!recs.every((x) => x && x.energy != null)) return null;
  const key = recs[0].level?.key;
  if (!all.every((x) => x.level?.key === key)) return null;
  const k = 627.509474, sum = (ids) => ids.reduce((a, id) => a + structures[id].energy, 0);
  const base = sum(r.reactants);
  const e = (x) => (x && x.energy != null && x.level?.key === key ? (x.energy - base) * k : null);
  return { reactants: 0, reactant_complex: e(cx[0]), product_complex: e(cx[1]), products: (sum(r.products) - base) * k };
}

function Ladder({ r, st, structures }) {
  const L = (r.ladder && r.ladder.reactant_complex != null ? r.ladder : ladderFromGraph(r, structures)) || r.ladder || {};
  const ts = st.barrier != null && L.reactant_complex != null ? L.reactant_complex + st.barrier
    : st.barrierUnverified != null && L.reactant_complex != null ? L.reactant_complex + st.barrierUnverified : null;
  const levels = [
    // One molecule on a side: its "complex" is the molecule itself, no extra level.
    { key: 'reactants', label: r.reactants.length > 1 ? 'separated' : 'reactant', v: L.reactants },
    r.reactants.length > 1 && { key: 'rc', label: 'reactant complex', v: L.reactant_complex },
    { key: 'ts', label: st.barrier != null ? 'TS' : st.barrierUnverified != null ? 'path max?' : 'TS (not searched)', v: ts },
    r.products.length > 1 && { key: 'pc', label: 'product complex', v: L.product_complex },
    { key: 'products', label: r.products.length > 1 ? 'separated' : 'product', v: L.products },
  ].filter(Boolean);
  const known = levels.filter((l) => l.v != null);
  if (known.length < 2) return html`<p class="small muted">No energies for this reaction yet.</p>`;
  const W = 520, H = 150, pad = 26, lo = Math.min(...known.map((l) => l.v)), hi = Math.max(...known.map((l) => l.v));
  const y = (v) => pad + (H - 2 * pad) * (hi === lo ? 0.5 : (hi - v) / (hi - lo));
  const x = (i) => 20 + i * ((W - 40) / (levels.length - 1));
  const pts = levels.map((l, i) => ({ ...l, x: x(i), y: l.v != null ? y(l.v) : null }));
  const drawn = pts.filter((p) => p.y != null);
  return html`<svg class="rc-ladder" viewBox=${`0 0 ${W} ${H + 18}`} role="img" aria-label="Energy along the reaction">
    ${drawn.slice(1).map((p, i) => html`<line x1=${drawn[i].x + 30} y1=${drawn[i].y} x2=${p.x - 30} y2=${p.y} class="rc-link" />`)}
    ${pts.map((p) => (p.y == null
      ? html`<text x=${p.x} y=${H / 2} class="rc-missing" text-anchor="middle">—</text>`
      : html`<g class=${`rc-level ${p.key}`}>
          <line x1=${p.x - 30} x2=${p.x + 30} y1=${p.y} y2=${p.y} />
          <text x=${p.x} y=${p.y - 6} text-anchor="middle" class="rc-v">${p.v >= 0 ? '+' : ''}${p.v.toFixed(1)}</text>
        </g>`))}
    ${pts.map((p) => html`<text x=${p.x} y=${H + 12} text-anchor="middle" class="rc-l">${p.label}</text>`)}
  </svg>`;
}

function useText(url) {
  const [t, setT] = useState(null);
  useEffect(() => {
    let live = true;
    setT(null);
    if (url) api.get(url).then((x) => live && setT(x)).catch(() => live && setT(''));
    return () => { live = false; };
  }, [url]);
  return t;
}

export function ReactionCard({ r, event = null, compact = false }) {
  const structures = useStore((s) => s.workspace.structures);
  const edges = useStore((s) => s.workspace.edges);
  useStore((s) => s.jobs);
  const edge = r.edge ? edges[r.edge] : null;
  const st = edge ? edgeStatus(edge, state.jobs) : { status: 'idle' };
  const events = r.events || [];
  const [ev, setEv] = useState(event ?? events[0] ?? null);
  const [view, setView] = useState(ev != null ? 'event' : 'rc');
  const [rc, pc] = r.complexes || [];
  const xyzR = useText(view === 'rc' && rc ? `/api/structures/${rc}/xyz` : null);
  const xyzP = useText(view === 'pc' && pc ? `/api/structures/${pc}/xyz` : null);
  const xyzTS = useText(view === 'ts' && st.barrierJob ? `/api/jobs/${st.barrierJob}/route-ts` : null);
  const job = r.origin?.job;
  const running = st.status === 'running' || st.status === 'queued';
  const runningJob = running ? Object.values(state.jobs).find((j) => j.targets.edges.includes(r.edge) && ['running', 'queued'].includes(j.status)) : null;
  const views = [
    ev != null && job && ['event', 'MD event'],
    rc && ['rc', 'Reactant complex'],
    pc && ['pc', 'Product complex'],
    st.barrierJob && ['ts', 'TS'],
  ].filter(Boolean);
  const height = compact ? 230 : 340;
  return html`<div class=${`reaction-card ${compact ? 'compact' : ''}`}>
    <${Equation} r=${r} structures=${structures} />
    <div class="small muted rc-meta">${r.origin?.kind === 'composed' ? 'composed by hand' : html`seen ${r.count || 0}× in the MD${r.reverse_count ? `, reverse ${r.reverse_count}×` : ''}`}
      · energies over the same atoms, kcal/mol
      ${compact && html` · <a href="#" onClick=${(e) => { e.preventDefault(); openAnalyze({ reaction: r.id }); }}>open in Analyze</a>`}</div>
    <${Ladder} r=${r} st=${st} structures=${structures} />

    <div class="rc-ts">
      ${!edge ? html`<p class="small warn-box">${noTsWhy(r)}
          ${['barrierless', 'recombines'].includes(r.complex_reason) ? ' There is no barrier to find; its ΔE is the reaction energy.' : ''}</p>`
        : running ? html`<p class="small">TS search ${st.status}…
            ${runningJob && html` <a href="#" onClick=${(e) => { e.preventDefault(); openJob(runningJob.id); }}>watch it</a>`}</p>`
          : html`<div class="rc-ts-row">
            ${st.barrierJob ? html`<a href="#" class="rc-barrier" onClick=${(e) => { e.preventDefault(); openJob(st.barrierJob); }}
                title="Its TS search: TS, IRC, energy profile, sample more paths">
                ${st.barrier != null ? `ΔE‡ ${fmtKcal(st.barrier)}` : `≈${fmtKcal(st.barrierUnverified)}? (not verified)`} kcal/mol from the reactant complex →</a>`
              : html`<span class="small muted">No TS yet.</span>`}
            <span class="spacer"></span>
            <button class=${`btn ${st.barrierJob ? '' : 'primary'}`} onClick=${() => findTs(r)}
              title="Path search, TS optimization and IRC between the optimized reactant and product complexes (only this reaction's molecules)">
              ${st.barrierJob ? 'Search again' : 'Find transition state'}</button>
          </div>`}
    </div>

    ${views.length > 0 && html`<div class="rc-view">
      <div class="segmented wide">
        ${views.map(([k, l]) => html`<button class=${view === k ? 'on' : ''} onClick=${() => setView(k)}>${l}</button>`)}
      </div>
      ${view === 'event' && events.length > 1 && html`<div class="rc-events small">MD occurrence:
        ${events.map((e, i) => html`<button class=${`btn-link small ${e === ev ? 'on' : ''}`} onClick=${() => setEv(e)}>${i + 1}</button>`)}</div>`}
      ${view === 'event' && ev != null && job && html`<${EventPlayer} key=${`${job}:${ev}`} jobId=${job} event=${ev} height=${height} />`}
      ${view === 'rc' && html`<${Viewer3D} xyz=${xyzR} height=${height} />
        <p class="small muted">${r.origin?.kind === 'composed' ? 'The reactants placed side by side, then optimized together.'
          : 'Only the molecules this reaction needs, optimized together (from the MD frame before it).'}</p>`}
      ${view === 'pc' && html`<${Viewer3D} xyz=${xyzP} height=${height} />`}
      ${view === 'ts' && html`<${Viewer3D} xyz=${xyzTS} height=${height} />`}
    </div>`}
  </div>`;
}
