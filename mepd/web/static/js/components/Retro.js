// A retrosynthesis job's routes: one card per route, its steps in the order
// they are run (building blocks first), each step's TS search one click away
// (the step is an ordinary reaction in Explore).
import { html, useState } from '../lib.js';
import { useStore } from '../store.js';
import { depictUrl, fmtKcal } from '../util.js';
import { ReactionCard, TsCell, reactionOf } from './Reactions.js';

const STOCK = { stock: 'in stock', small: 'small molecule, counted as available' };

function Mol({ smiles, stock, size = 92 }) {
  return html`<span class=${`retro-mol ${stock ? 'stock' : ''}`} title=${`${smiles}${stock ? ` · ${STOCK[stock] || stock}` : ''}`}>
    <img src=${depictUrl(smiles, Math.round(size * 1.3), size)} alt=${smiles} loading="lazy" />
  </span>`;
}

function stepNote(s) {
  const bits = [s.method];
  if (s.reaction) bits.push(s.reaction);
  if (s.reagents?.length) bits.push(`reagents ${s.reagents.join(', ')}`);
  if (s.coreactants?.length) bits.push(`also uses ${s.coreactants.join(', ')}`);
  if (s.byproducts?.length) bits.push(`releases ${s.byproducts.join(', ')}`);
  if (s.byproducts == null) bits.push("atoms don't balance with a known byproduct");
  if (s.delta_e_kcal != null) bits.push(`ΔE ${fmtKcal(s.delta_e_kcal)} kcal/mol`);
  if (s.roundtrip === false) bits.push('forward model did not give it back');
  if (s.score != null) bits.push(`score ${s.score.toPrecision(2)}`);
  return bits.join(' · ');
}

function Step({ job, s, stock, open, onOpen }) {
  const ws = useStore((st) => st.workspace);
  const r = reactionOf(ws, job.id, s.key);
  return html`<div class=${`retro-step ${open ? 'on' : ''}`}>
    <div class="retro-eq clickable" onClick=${onOpen} title=${stepNote(s)}>
      ${s.reactants.map((m, k) => html`${k > 0 && html`<span class="retro-plus">+</span>`}<${Mol} smiles=${m} stock=${stock[m]} />`)}
      <span class="retro-arrow">→</span>
      <${Mol} smiles=${s.product} />
    </div>
    <div class="retro-ts" onClick=${(e) => e.stopPropagation()}>
      ${s.barrier_kcal != null
    ? html`<span class="mono small" title=${s.verified ? 'Checked in this job: path search, TS and IRC connect the step' : 'Checked in this job: path maximum, no TS/IRC confirmed it'}>${s.verified ? 'ΔE‡ ' : '≈'}${fmtKcal(s.barrier_kcal)}</span>`
    : html`<${TsCell} r=${r} why=${s.byproducts == null ? "atoms don't balance: no TS search" : 'not in Explore (deleted?)'} />`}
    </div>
    ${open && html`<div class="retro-card">${r ? html`<${ReactionCard} key=${r.id} r=${r} compact=${true} />`
    : html`<p class="small muted">This step is not in Explore${s.byproducts == null ? ": its atoms don't balance with a known byproduct" : ''}.</p>`}</div>`}
  </div>`;
}

export function RetroRoutes({ job, retro }) {
  const [openRoute, setOpenRoute] = useState(0);
  const [openStep, setOpenStep] = useState(null);
  if (retro.live) return html`<p class="small muted">Searching… routes appear when the search ends.</p>`;
  if (!retro.routes.length) return html`<p class="small muted">No route found within the budget: raise the search budget or Most steps, or try another method or stock.</p>`;
  return html`<div class="retro-routes">
    ${!retro.solved && html`<p class="small muted">No route reaches the stock; the closest one is shown (molecules without a green frame still need a route).</p>`}
    ${retro.routes.map((rt, i) => {
    const stock = Object.fromEntries(rt.leaves.map((l) => [l.smiles, l.in_stock]));
    const open = openRoute === i;
    return html`<div class=${`retro-route ${open ? 'open' : ''}`}>
        <div class="retro-route-head clickable" onClick=${() => setOpenRoute(open ? null : i)}>
          <strong>Route ${rt.rank}</strong>
          <span class="small">${rt.n_steps} step${rt.n_steps !== 1 ? 's' : ''}</span>
          ${rt.highest_barrier_kcal != null && html`<span class="badge accent" title="Highest barrier along the route (each step checked by path search)">ΔE‡ max ${rt.all_verified ? '' : '≈'}${fmtKcal(rt.highest_barrier_kcal)}</span>`}
          <span class="spacer"></span>
          <span class="small muted" title="Product of the step scores (the search's cost is −log of it)">score ${rt.score != null ? rt.score.toPrecision(2) : '—'}</span>
          ${!open && html`<span class="retro-mini">${rt.leaves.slice(0, 4).map((l) => html`<${Mol} smiles=${l.smiles} stock=${l.in_stock} size=${38} />`)}</span>`}
        </div>
        ${open && rt.steps.map((s, k) => html`<${Step} job=${job} s=${s} stock=${stock}
          open=${openStep === `${i}:${k}`} onOpen=${() => setOpenStep(openStep === `${i}:${k}` ? null : `${i}:${k}`)} />`)}
      </div>`;
  })}
  </div>`;
}
