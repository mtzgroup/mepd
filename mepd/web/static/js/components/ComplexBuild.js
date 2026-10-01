// Building a complex's geometries from its molecules (mepd.complexes), one
// form for every place that does it: "Combine into a complex" (species
// picked, so many of each) and a complex node's Conformers card (other
// arrangements of the molecules it has). Docked, an ensemble or a solvation
// shell run as a job; packed / side by side are placed at once. Every
// geometry joins the complex node and is minimized at the workspace level.
import { html, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, select, useStore } from '../store.js';

export function ComplexMethodForm({ counts, exclude = [], runLabel, onDone }) {
  const op = useStore((s) => s.operations.find((o) => o.key === 'complex'));
  const methods = (op?.methods || []).filter((m) => !exclude.includes(m.fixed.method));
  const [method, setMethod] = useState(null);
  const pick = methods.find((m) => m.fixed.method === method) || methods.find((m) => m.available) || methods[0];
  const total = Object.values(counts).reduce((t, n) => t + n, 0);
  const make = () => attempt(() => api.post('/api/complexes', {
    counts: Object.fromEntries(Object.entries(counts).filter(([, n]) => n > 0)), method: pick?.fixed.method || 'side',
  }), null).then((out) => {
    if (!out) return;
    onDone?.(out);
    if (out.job) openJob(out.job);
    else select({ structures: [out.complex.id] });
  });
  return html`
    ${methods.length > 0 && html`<label class="combine-row small">How
      <select value=${pick?.fixed.method} onChange=${(e) => setMethod(e.target.value)}>
        ${methods.map((m) => html`<option value=${m.fixed.method} disabled=${!m.available} title=${m.reason}>${m.label}${m.available ? '' : ' (not installed)'}</option>`)}
      </select></label>
    <p class="small muted combine-help">${pick?.summary}</p>`}
    <div class="row-actions">
      <button class="btn primary" disabled=${total < 2 || total > 12 || !pick?.available} onClick=${make}
        title="Then minimized at the workspace level; shown in the complex node if it holds together">
        ${runLabel || `Create complex (${total} molecules)`}</button>
    </div>`;
}

// A complex node's members as {species id: how many}.
export function memberCounts(members) {
  return (members || []).reduce((m, id) => ({ ...m, [id]: (m[id] || 0) + 1 }), {});
}
