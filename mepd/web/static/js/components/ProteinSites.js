// Where a species docks in a protein (`mepd qmmm protein-sites`): the
// protein as a cartoon with the species at every site, the sites listed best
// first; pick one and build the QM/MM system there (`qmmm-protein-build`).
import { html, useEffect, useRef, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, useStore } from '../store.js';

function themeBackground() {
  return getComputedStyle(document.documentElement).getPropertyValue('--viewer-bg').trim() || '#ffffff';
}

function speciesXyz(symbols, coords) {
  return `${symbols.length}\n\n${coords.map((c, i) => `${symbols[i]} ${c[0]} ${c[1]} ${c[2]}`).join('\n')}\n`;
}

// "A:ARG90" -> {chain: 'A', resi: 90}
function residueSel(tag) {
  const [chain, res] = tag.split(':');
  return { chain, resi: parseInt(res.replace(/^\D+/, ''), 10) };
}

export function ProteinSites({ job, data }) {
  const host = useRef(null);
  const viewer = useRef(null);
  const [pdb, setPdb] = useState(null);
  const [symbols, setSymbols] = useState(null);
  const [error, setError] = useState(null);
  const sites = data.sites || [];
  const [sel, setSel] = useState(sites[0]?.id ?? 0);
  const [showAll, setShowAll] = useState(true);
  const [form, setForm] = useState({ qm_residues: '', water_shell: 8, active_radius: 6, freeze_protein: false });
  const builds = useStore((s) => Object.values(s.jobs).filter((j) => j.op === 'qmmm-protein-build' && j.source_job === job.id)
    .sort((a, b) => b.created - a.created));
  const site = sites.find((s) => s.id === sel);
  // Reactions of the docked species in Explore: bring one along (its other end, and TS) into the site.
  const speciesId = job.targets?.structures?.[0];
  const reactions = useStore((s) => Object.values(s.workspace.edges || {}).filter((e) => e.source === speciesId || e.target === speciesId)
    .map((e) => ({ id: e.id, other: s.workspace.structures[e.source === speciesId ? e.target : e.source] }))
    .filter((r) => r.other && !r.other.qmmm));
  const [edge, setEdge] = useState('');

  useEffect(() => {
    let live = true;
    Promise.all([
      fetch(`/api/jobs/${job.id}/files/${data.protein}`).then((r) => (r.ok ? r.text() : Promise.reject(new Error('no protein.pdb')))),
      fetch(`/api/jobs/${job.id}/files/${data.species}`).then((r) => (r.ok ? r.text() : Promise.reject(new Error('no species.xyz')))),
    ]).then(([p, x]) => {
      if (!live) return;
      setPdb(p);
      setSymbols(x.trim().split('\n').slice(2).map((l) => l.trim().split(/\s+/)[0]));
    }).catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [job.id, data.protein, data.species]);

  useEffect(() => {
    if (!pdb || !symbols || !host.current || !window.$3Dmol) return;
    if (!viewer.current) viewer.current = window.$3Dmol.createViewer(host.current, { backgroundColor: themeBackground(), antialias: true });
    const v = viewer.current;
    v.resize();   // the panel's size may have changed since the viewer was made
    v.removeAllModels(); v.removeAllLabels(); v.removeAllShapes();
    const prot = v.addModel(pdb, 'pdb');
    prot.setStyle({}, { cartoon: { color: 'spectrum', opacity: 0.85 } });
    if (site) {
      // The residues lining the chosen site, as sticks.
      for (const tag of site.residues) prot.setStyle(residueSel(tag), { cartoon: { color: 'spectrum', opacity: 0.85 }, stick: { radius: 0.12, colorscheme: 'grayCarbon' } });
    }
    let chosenModel = null;
    for (const s of sites) {
      const chosen = s.id === sel;
      if (!chosen && !showAll) continue;
      const m = v.addModel(speciesXyz(symbols, s.coords), 'xyz');
      if (chosen) chosenModel = m;
      m.setStyle({}, chosen
        ? { stick: { radius: 0.2, colorscheme: 'greenCarbon' }, sphere: { scale: 0.25, colorscheme: 'greenCarbon' } }
        : { stick: { radius: 0.12, colorscheme: 'orangeCarbon', opacity: 0.8 } });
      v.addLabel(String(s.id), { position: { x: s.centroid[0], y: s.centroid[1], z: s.centroid[2] + 3 },
        fontSize: chosen ? 14 : 11, backgroundColor: chosen ? '#1f7a3a' : '#7a5a1f', backgroundOpacity: 0.8, fontColor: 'white' });
    }
    // The chosen site in the middle, with its surroundings (~30 Å) in view.
    if (chosenModel) {
      v.zoomTo({ model: chosenModel });
      v.zoom(0.25);
    } else v.zoomTo();
    v.render();
  }, [pdb, symbols, sel, showAll]);

  const build = async () => {
    const out = await attempt(() => api.post('/api/jobs', { op: 'qmmm-protein-build', source_job: job.id,
      params: { site: sel, ...form, water_shell: +form.water_shell, active_radius: +form.active_radius, edge } }),
      `Building the QM/MM system at site ${sel}`);
    const made = Array.isArray(out) ? out[0] : out;
    if (made?.id) openJob(made.id);
  };
  const toggleResidue = (tag) => {
    const now = form.qm_residues.split(/\s+/).filter(Boolean);
    setForm({ ...form, qm_residues: (now.includes(tag) ? now.filter((t) => t !== tag) : [...now, tag]).join(' ') });
  };
  const inQm = new Set(form.qm_residues.split(/\s+/).filter(Boolean));

  if (error) return html`<div class="warn-box">Could not load the protein: ${error}</div>`;
  return html`<div class="protein-sites">
    <div class="protein-view">
      ${window.$3Dmol
        ? html`<div class="viewer" ref=${host} style=${{ height: '440px', position: 'relative' }}></div>`
        : html`<div class="viewer viewer-missing" style=${{ height: '440px' }}>3D viewer unavailable (3Dmol.js did not load)</div>`}
      <label class="small"><input type="checkbox" checked=${showAll} onChange=${(e) => setShowAll(e.target.checked)} /> show every site</label>
    </div>
    <div class="protein-side">
      <table class="site-table small">
        <thead><tr><th>Site</th><th title="AutoDock Vina score of the best pose here (kcal/mol; lower binds better)">Score</th>
          <th title="How many poses, from all docking boxes, landed here">Found</th><th>Lined by</th></tr></thead>
        <tbody>${sites.map((s) => html`<tr class=${s.id === sel ? 'selected' : ''} onClick=${() => setSel(s.id)}>
          <td>${s.id}</td><td>${s.score.toFixed(1)}</td><td>${s.hits}×</td>
          <td class="muted" title=${s.residues.join(' ')}>${s.residues.slice(0, 4).join(' ')}${s.residues.length > 4 ? ' …' : ''}</td></tr>`)}</tbody>
      </table>
      ${site && html`<div class="qmmm-panel">
        <h4>Build the QM/MM system at site ${site.id}</h4>
        <p class="small muted">The species is the QM region; the protein (AMBER ff14SB) and a TIP3P water shell are the
          environment, and their charges act on the QM region (needs xTB or Psi4 as the QM level, in Settings).</p>
        <div class="small"><b>Residues in the QM region</b> (click to add their side chains):</div>
        <div class="chip-row">${site.residues.map((t) => html`<button class=${`chip small ${inQm.has(t) ? 'on' : ''}`} onClick=${() => toggleResidue(t)}>${t}</button>`)}</div>
        <div class="row small">
          <label>water shell (Å) <input class="tiny" type="number" min="0" step="1" value=${form.water_shell} onInput=${(e) => setForm({ ...form, water_shell: e.target.value })} /></label>
          <label>moving shell (Å) <input class="tiny" type="number" min="0" step="0.5" value=${form.active_radius} onInput=${(e) => setForm({ ...form, active_radius: e.target.value })} /></label>
          <label title="Only the water around the species moves; the protein still acts through its charges">
            <input type="checkbox" checked=${form.freeze_protein} onChange=${(e) => setForm({ ...form, freeze_protein: e.target.checked })} /> freeze the protein</label>
        </div>
        ${reactions.length > 0 && html`<label class="field small"><span class="field-label">Bring a reaction along</span>
          <select value=${edge} onChange=${(e) => setEdge(e.target.value)}>
            <option value="">No: the species only</option>
            ${reactions.map((r) => html`<option value=${r.id}>→ ${r.other.name} (and its TS, if known)</option>`)}
          </select></label>`}
        <div class="row-actions"><button class="btn primary" onClick=${build}>Build QM/MM system here</button></div>
      </div>`}
      ${builds.length > 0 && html`<div class="small"><b>Built here</b>
        ${builds.map((b) => html`<div><a href="#" onClick=${(e) => { e.preventDefault(); openJob(b.id); }}>site ${b.params?.site}</a>
          <span class="muted"> · ${b.status}</span></div>`)}</div>`}
    </div>
  </div>`;
}
