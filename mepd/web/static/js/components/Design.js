// Design tab: build or edit a 3D molecule, then send it to the Graph.
//
// The molecule lives on the server (workspace "design", an MDL molblock with
// explicit hydrogens); every edit is one request that returns the new one,
// so RDKit checks valences, re-adds hydrogens and places new atoms (see
// mepd/web/design.py). This view is the 3Dmol canvas plus tools: click atoms
// to change them. Undo/redo keep the molblocks seen in this browser tab.
import { html, useEffect, useRef, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, openTab, prefs, select, toast, useStore } from '../store.js';
import { conformerRows, conformerLabel } from './Inspector.js';

const COMMON = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'B', 'Si'];
// The periodic table as [symbol, row, column] (lanthanides/actinides on rows 8-9).
const PT_ROWS = [
  'H . . . . . . . . . . . . . . . . He',
  'Li Be . . . . . . . . . . B C N O F Ne',
  'Na Mg . . . . . . . . . . Al Si P S Cl Ar',
  'K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr',
  'Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe',
  'Cs Ba * Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn',
  'Fr Ra ** Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og',
  '. . La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu .',
  '. . Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr .',
];
const TABLE = PT_ROWS.flatMap((row, r) => row.split(' ').map((sym, c) => [sym, r + 1, c + 1]))
  .filter(([sym]) => sym !== '.' && !sym.startsWith('*'));

// Pick any element: common ones up front, the whole table one click away.
// Elements the workspace level of theory is not parametrized for are marked.
function ElementPicker({ value, onPick, covered, method }) {
  const [open, setOpen] = useState(false);
  const off = (sym) => covered && !covered.includes(sym);
  const pick = (sym) => {
    onPick(sym);
    if (off(sym)) toast(`${method} is not parametrized for ${sym}: results with it are unlikely to be meaningful`, 'error', 7000);
  };
  const btn = (sym) => html`<button class=${`el ${value === sym ? 'on' : ''} ${off(sym) ? 'uncovered' : ''}`}
    title=${off(sym) ? `${sym}: not covered by ${method}` : sym} onClick=${() => pick(sym)}>${sym}</button>`;
  return html`<div>
    <div class="palette">${COMMON.map(btn)}
      <button class=${`el wide ${open ? 'on' : ''}`} onClick=${() => setOpen(!open)}>${open ? 'Hide table' : 'All elements…'}</button></div>
    ${open && html`<div class="ptable">${TABLE.map(([sym, r, c]) => html`<button style=${`grid-row:${r};grid-column:${c}`}
      class=${`pt ${value === sym ? 'on' : ''} ${off(sym) ? 'uncovered' : ''}`} title=${off(sym) ? `${sym}: not covered by ${method}` : sym}
      onClick=${() => pick(sym)}>${sym}</button>`)}</div>
      ${covered && html`<p class="small muted">Greyed: not covered by ${method}.</p>`}`}
  </div>`;
}

// Numbered "what to do now" for the tools that act on a clicked atom.
function Steps({ steps }) {
  return html`<ol class="steps-mini">${steps.map((s) => html`<li>${s}</li>`)}</ol>`;
}
const TOOLS = [
  ['view', 'View', 'Rotate and zoom; click an atom to see what it is.'],
  ['element', 'Element', 'Click an atom to turn it into the chosen element (hydrogens are re-added to fit).'],
  ['add', 'Add atom', 'Click an atom to bond a new atom of the chosen element to it (it takes a hydrogen\'s place).'],
  ['place', 'Add molecule', 'Click an atom to put whole molecules or ions next to it, not bonded: a metal ion or Lewis acid to coordinate it, or several waters to solvate it.'],
  ['group', 'Swap group', 'Click a hydrogen, or the first atom of a terminal group (e.g. a methyl carbon), to replace it with the chosen group.'],
  ['bond', 'Bond', 'Click two atoms to give them the chosen bond order (or remove their bond).'],
  ['charge', 'Charge', 'Click an atom to add (+) or remove (−) a formal charge.'],
  ['delete', 'Delete', 'Click an atom to remove it with its hydrogens.'],
];
const ORDERS = [[1, 'single'], [2, 'double'], [3, 'triple'], [1.5, 'aromatic'], [0, 'remove']];

function themeBackground() {
  return getComputedStyle(document.documentElement).getPropertyValue('--viewer-bg').trim() || '#ffffff';
}

function Canvas({ molblock, liveXyz, picked, changed, onAtom, labels, clickable }) {
  const host = useRef(null);
  const viewer = useRef(null);
  const last = useRef({ n: 0 });
  const clickRef = useRef(onAtom);
  clickRef.current = onAtom;

  useEffect(() => {
    if (!window.$3Dmol || !host.current) return undefined;
    viewer.current = window.$3Dmol.createViewer(host.current, { backgroundColor: themeBackground(), antialias: true });
    host.current.viewer = viewer.current;   // lets UI tests find atoms on screen
    const ro = new ResizeObserver(() => { viewer.current?.resize(); viewer.current?.render(); });
    ro.observe(host.current);
    return () => { ro.disconnect(); viewer.current?.clear(); viewer.current = null; };
  }, []);

  useEffect(() => {
    const v = viewer.current;
    if (!v) return;
    const view = v.getView();
    v.removeAllModels();
    v.removeAllLabels();
    v.removeAllShapes();
    const data = liveXyz || molblock;
    if (!data) { v.render(); return; }
    const model = v.addModel(data, liveXyz ? 'xyz' : 'sdf');
    v.setStyle({}, { stick: { radius: 0.14 }, sphere: { scale: 0.26 } });
    if (!liveXyz) {
      // The picked atom(s) and what the last edit touched.
      if (picked.length) v.addStyle({ index: picked }, { sphere: { scale: 0.42, color: '#e0a100', opacity: 0.85 } });
      if (changed.length) v.addStyle({ index: changed }, { sphere: { scale: 0.34, color: '#3aa76d', opacity: 0.75 } });
      model.setClickable({}, true, (atom) => clickRef.current(atom.index));
      model.setHoverable({}, true,
        (atom) => { if (!atom.label) atom.label = v.addLabel(`${atom.elem}${atom.index + 1}`, { position: atom, fontSize: 11, backgroundOpacity: 0.6, inFront: true }); },
        (atom) => { if (atom.label) { v.removeLabel(atom.label); delete atom.label; } });
      if (labels) {
        model.selectedAtoms({}).forEach((a) => v.addLabel(`${a.elem}${a.index + 1}`, {
          position: a, fontSize: 10, backgroundOpacity: 0.5, inFront: true }));
      }
    }
    const n = model.selectedAtoms({}).length;
    // Same atoms (a minimization, a bond order): keep the camera; new or
    // removed atoms: frame the whole molecule again.
    if (last.current.n === n) v.setView(view); else v.zoomTo();
    last.current.n = n;
    v.render();
  }, [molblock, liveXyz, picked.join(','), changed.join(','), labels]);

  return html`<div class=${`design-canvas ${clickable ? 'picking' : ''}`} ref=${host}></div>`;
}

function Start({ structures }) {
  const [smiles, setSmiles] = useState('');
  const [sid, setSid] = useState('');
  const [cid, setCid] = useState('');
  const rec = structures[sid];
  const list = Object.values(structures).sort((a, b) => b.created - a.created);
  return html`<div class="design-start">
    <label class="field"><span class="field-label">From SMILES</span>
      <div class="row"><input value=${smiles} placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O" onInput=${(e) => setSmiles(e.target.value)}
        onKeyDown=${(e) => e.key === 'Enter' && smiles.trim() && attempt(() => api.post('/api/design/new', { smiles }))} />
      <button class="btn primary" disabled=${!smiles.trim()} onClick=${() => attempt(() => api.post('/api/design/new', { smiles }))}>Build</button></div></label>
    <label class="field"><span class="field-label">From the graph</span>
      <div class="row"><select value=${sid} onChange=${(e) => { setSid(e.target.value); setCid(''); }}>
        <option value="">Choose a structure…</option>
        ${list.map((r) => html`<option value=${r.id}>${r.name}${r.role === 'ts' ? ' (TS)' : ''}</option>`)}
      </select>
      ${rec && (rec.conformers || []).length > 1 && html`<select value=${cid} onChange=${(e) => setCid(e.target.value)}>
        <option value="">Lowest conformer</option>
        ${conformerRows(rec).map((c) => html`<option value=${c.id}>${conformerLabel(c)}</option>`)}</select>`}
      <button class="btn" disabled=${!sid} onClick=${() => attempt(() => api.post('/api/design/load', { structure: sid, conformer: cid || null }))}>Load</button></div></label>
  </div>`;
}

export function DesignView() {
  const design = useStore((s) => s.workspace.design);
  const structures = useStore((s) => s.workspace.structures);
  const levelProfile = useStore((s) => s.levelProfile);
  const level = useStore((s) => s.levels[s.levelProfile ?? '']);
  const jobs = useStore((s) => s.jobs);
  const progress = useStore((s) => s.progress);
  const [tool, setTool] = useState(() => prefs.get('designTool', 'view'));
  const [element, setElement] = useState('C');
  const [order, setOrder] = useState(2);
  const [group, setGroup] = useState('methyl');
  const [groups, setGroups] = useState([]);
  const [species, setSpecies] = useState([]);
  const [placeWhat, setPlaceWhat] = useState('water');
  const [placeCount, setPlaceCount] = useState(1);
  const [placeSmiles, setPlaceSmiles] = useState('');
  const [groupSmiles, setGroupSmiles] = useState('');
  const [hessian, setHessian] = useState(() => prefs.get('designHessian', false));
  const [cov, setCov] = useState({ method: '', elements: null });
  const [picked, setPicked] = useState([]);
  const [changed, setChanged] = useState([]);
  const [labels, setLabels] = useState(false);
  const [busy, setBusy] = useState(false);
  const [info, setInfo] = useState('');
  const history = useRef({ undo: [], redo: [] });
  const [, bump] = useState(0);

  useEffect(() => {
    api.get('/api/design/groups').then(setGroups).catch(() => {});
    api.get('/api/design/species').then(setSpecies).catch(() => {});
  }, []);
  useEffect(() => { api.get('/api/design/coverage').then(setCov).catch(() => {}); }, [levelProfile, level?.key]);
  useEffect(() => { prefs.set('designTool', tool); setPicked([]); }, [tool]);

  // The minimization of this design that is running (its frames animate here).
  const running = Object.values(jobs).filter((j) => ['design-optimize', 'design-tsopt'].includes(j.op) && ['queued', 'running'].includes(j.status))
    .sort((a, b) => b.created - a.created)[0];
  const stream = running && progress[running.id]?.streams?.opt_0;
  const frames = stream?.geometry?.frames || [];
  const liveXyz = running && frames.length ? frames[frames.length - 1] : null;

  const apply = async (fn, { record = true } = {}) => {
    if (busy) return null;
    setBusy(true);
    const before = design?.molblock;
    const out = await attempt(fn);
    setBusy(false);
    if (out && out.molblock !== undefined) {
      if (record && before && before !== out.molblock) {
        history.current.undo.push(before);
        history.current.redo = [];
        bump((x) => x + 1);
      }
      setChanged(out.changed || []);
      (out.warnings || []).forEach((w) => toast(w, 'info', 7000));
    }
    return out;
  };
  const edit = (op) => apply(() => api.post('/api/design/edit', { op }));

  const onAtom = (idx) => {
    if (running) { toast('Wait for the minimization to finish (or stop it) before editing', 'info'); return; }
    if (tool === 'view') { setPicked([idx]); setInfo(`Atom ${idx + 1}`); return; }
    if (tool === 'element') edit({ op: 'element', atom: idx, element });
    else if (tool === 'add') edit({ op: 'add', atom: idx, element });
    else if (tool === 'group') edit(groupSmiles.trim() ? { op: 'group', atom: idx, smiles: groupSmiles.trim() } : { op: 'group', atom: idx, group });
    else if (tool === 'place') edit(placeSmiles.trim() ? { op: 'place', atom: idx, smiles: placeSmiles.trim(), count: placeCount }
      : { op: 'place', atom: idx, species: placeWhat, count: placeCount });
    else if (tool === 'delete') edit({ op: 'delete', atom: idx });
    else if (tool === 'charge') edit({ op: 'charge', atom: idx, delta: order === 0 ? -1 : 1 });
    else if (tool === 'bond') {
      if (!picked.length || picked[0] === idx) { setPicked([idx]); return; }
      const a = picked[0];
      setPicked([]);
      edit({ op: 'bond', a, b: idx, order });
    }
  };

  const undo = () => {
    const h = history.current;
    const prev = h.undo.pop();
    if (!prev) return;
    h.redo.push(design.molblock);
    bump((x) => x + 1);
    apply(() => api.put('/api/design', { molblock: prev }), { record: false });
  };
  const redo = () => {
    const h = history.current;
    const next = h.redo.pop();
    if (!next) return;
    h.undo.push(design.molblock);
    bump((x) => x + 1);
    apply(() => api.put('/api/design', { molblock: next }), { record: false });
  };
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest('input, textarea, select')) return;
      if ((e.metaKey || e.ctrlKey) && e.key === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const toGraph = async () => {
    const rec = await attempt(() => api.post('/api/design/to-graph'));
    if (!rec) return;
    toast(rec.duplicate ? `${rec.name} already has this geometry` : rec.merged
      ? `Added as a new conformer of ${rec.name}` : `Added ${rec.name} to the graph`, 'ok', 6000,
    { label: 'Show', run: () => { select({ structures: [rec.id] }); openTab('graph'); } });
  };
  const tsopt = async () => {
    const created = await attempt(() => api.post('/api/design/tsopt', { profile: levelProfile ?? null }));
    if (created) toast(`Optimizing the design as a TS at ${level?.label ?? 'the workspace level'}, then an IRC…`, 'info', 6000);
  };
  const tsToGraph = async () => {
    const out = await attempt(() => api.post('/api/design/ts-to-graph'));
    if (!out) return;
    const ids = [out.ts?.id, ...(out.added || []).map((r) => r.id), ...(out.reused || []).map((r) => r.id)].filter(Boolean);
    toast(`Added the TS${out.edge ? ' and its IRC ends, joined by an edge' : ''} to the graph`, 'ok', 7000,
      { label: 'Show', run: () => { select({ structures: ids }); openTab('graph'); } });
  };
  const minimize = async () => {
    const created = await attempt(() => api.post('/api/design/minimize', {
      profile: levelProfile ?? null, params: { validate_minima_with_hessian: hessian } }));
    if (created) toast(`Minimizing at ${level?.label ?? 'the workspace level'}…`, 'info');
  };

  const toolHelp = TOOLS.find(([k]) => k === tool)?.[2];
  const energy = design?.energy != null ? `${design.energy.toFixed(6)} Eh` : null;

  return html`
    <div class="design-view">
      <div class="page-head">
        <h2>Design</h2>
        <p class="small muted">Build or edit a structure in 3D, minimize it, and add it to the graph for further calculations.</p>
      </div>
      ${!design ? html`<div class="design-empty"><${Start} structures=${structures} /></div>` : html`
      <div class="design-body">
        <div class="design-tools">
          ${TOOLS.map(([k, label, help]) => html`<button class=${`tool ${tool === k ? 'on' : ''}`} title=${help} onClick=${() => setTool(k)}>${label}</button>`)}
          ${(tool === 'element' || tool === 'add') && html`<div>
            <${Steps} steps=${[`Pick the element (now ${element}).`, tool === 'add'
              ? 'Click the atom in the 3D view to bond it to: it takes one of that atom\'s hydrogens, or points away from its neighbours.'
              : 'Click the atom in the 3D view to change: its hydrogens are redone to fit.']} />
            <${ElementPicker} value=${element} onPick=${setElement} covered=${cov.elements} method=${cov.method} />
          </div>`}
          ${tool === 'bond' && html`<div class="palette">
            ${ORDERS.map(([o, l]) => html`<button class=${`el wide ${order === o ? 'on' : ''}`} onClick=${() => setOrder(o)}>${l}</button>`)}
          </div>`}
          ${tool === 'charge' && html`<div class="palette">
            <button class=${`el ${order !== 0 ? 'on' : ''}`} onClick=${() => setOrder(1)}>+</button>
            <button class=${`el ${order === 0 ? 'on' : ''}`} onClick=${() => setOrder(0)}>−</button>
          </div>`}
          ${tool === 'place' && html`<div>
            <${Steps} steps=${[`Choose what to add (now ${placeSmiles.trim() || placeWhat}${placeCount > 1 ? ` ×${placeCount}` : ''}), or type any SMILES.`,
              'Click the atom in the 3D view it should go next to: it lands in the free space beside that atom, not bonded (click a hydrogen to H-bond to it).']} />
            <input class="smiles-in" placeholder="any molecule or ion, as SMILES (e.g. [Pd], [Cl-], O=C=O)" value=${placeSmiles}
              onInput=${(e) => setPlaceSmiles(e.target.value)} />
            <div class="palette groups">
              ${species.map((g) => html`<button class=${`el wide ${!placeSmiles.trim() && placeWhat === g ? 'on' : ''}`} onClick=${() => { setPlaceWhat(g); setPlaceSmiles(''); }}>${g}</button>`)}
            </div>
            <label class="small place-count">how many <input type="number" min="1" max="30" value=${placeCount}
              onChange=${(e) => setPlaceCount(Math.max(1, Math.min(30, +e.target.value || 1)))} /></label>
          </div>`}
          ${tool === 'group' && html`<div>
            <${Steps} steps=${[`Choose the new group (now ${groupSmiles.trim() || group}), or type any group as SMILES with [*] where it attaches.`,
              'Click a hydrogen to replace, or the first atom of a terminal group (e.g. a methyl carbon) to replace that whole group.']} />
            <input class="smiles-in" placeholder="any group, e.g. [*]C(=O)N(C)C" value=${groupSmiles} onInput=${(e) => setGroupSmiles(e.target.value)} />
            <div class="palette groups">
              ${groups.map((g) => html`<button class=${`el wide ${!groupSmiles.trim() && group === g ? 'on' : ''}`} onClick=${() => { setGroup(g); setGroupSmiles(''); }}>${g}</button>`)}
            </div>
          </div>`}
        </div>
        <div class="design-stage">
          <${Canvas} molblock=${design.molblock} liveXyz=${liveXyz} picked=${picked} changed=${changed} onAtom=${onAtom} labels=${labels} clickable=${tool !== 'view'} />
          <div class="design-hint small">${running ? html`${running.op === 'design-tsopt' ? 'Optimizing as a TS (then IRC)' : 'Minimizing'} at ${level?.label ?? 'the workspace level'}… <a href="#" onClick=${(e) => { e.preventDefault(); openJob(running.id); }}>details</a>`
            : busy ? 'Working…' : tool === 'bond' && picked.length ? `Atom ${picked[0] + 1} picked: click the second atom.` : toolHelp}</div>
          <div class="design-bar">
            <button class="btn small" disabled=${!history.current.undo.length || busy} onClick=${undo} title="Undo (Ctrl+Z)">Undo</button>
            <button class="btn small" disabled=${!history.current.redo.length || busy} onClick=${redo} title="Redo (Ctrl+Shift+Z)">Redo</button>
            <button class="btn small" disabled=${busy || running} onClick=${() => apply(() => api.post('/api/design/edit', { op: { op: 'hydrogens' } }))} title="Re-add hydrogens everywhere from valences">Fix H</button>
            <button class="btn small" disabled=${busy || running} onClick=${() => apply(() => api.post('/api/design/clean'))}
              title="Quick force-field cleanup (MMFF94, else UFF). Moves every atom: not for a transition state.">Clean (MMFF)</button>
            <button class="btn small" disabled=${busy || running} onClick=${tsopt}
              title=${`Treat this structure as a TS guess: saddle optimization at ${level?.label ?? 'the workspace level'}, then an IRC. If it doesn't converge, the design goes back to what you submitted.`}>Optimize as TS</button>
            <button class="btn small primary" disabled=${busy || running} onClick=${minimize}
              title=${`Minimize at ${level?.label ?? 'the workspace level'} (mepd optimize); the result replaces the design`}>Minimize</button>
            <label class="small" title="After minimizing, check for imaginary frequencies (and try to push off a saddle). Slow for large or floppy structures, e.g. with explicit solvent or ions, which rarely pass a strict check.">
              <input type="checkbox" checked=${hessian} onChange=${(e) => { setHessian(e.target.checked); prefs.set('designHessian', e.target.checked); }} /> Hessian check</label>
            <label class="small"><input type="checkbox" checked=${labels} onChange=${(e) => setLabels(e.target.checked)} /> atom numbers</label>
          </div>
        </div>
        <aside class="design-side">
          <input class="title-input" value=${design.name || ''} placeholder="Name"
            onChange=${(e) => attempt(() => api.put('/api/design', { name: e.target.value }))} />
          ${design.smiles && html`<div class="smiles">${design.smiles}</div>`}
          <dl class="props">
            <dt>Formula</dt><dd>${design.formula} (${design.natoms} atoms)</dd>
            <dt>Charge</dt><dd><input type="number" class="tiny" value=${design.charge}
              onChange=${(e) => attempt(() => api.put('/api/design', { charge: +e.target.value }))} />
              <span class="small muted">${design.charge_offset ? ` formal charges ${design.charge - design.charge_offset >= 0 ? '+' : ''}${design.charge - design.charge_offset}, ${design.charge_offset > 0 ? '+' : ''}${design.charge_offset} set by hand` : ' from formal charges'}</span></dd>
            <dt>Multiplicity</dt><dd><input type="number" class="tiny" min="1" value=${design.multiplicity}
              onChange=${(e) => attempt(() => api.put('/api/design', { multiplicity: +e.target.value }))} /></dd>
            <dt>Energy</dt><dd>${energy ? html`<span class="mono">${energy}</span> <span class="small muted">${design.level?.label}</span>`
              : html`<span class="small muted">not minimized since the last edit</span>`}</dd>
            ${design.source?.kind === 'graph' && html`<dt>From</dt><dd>${design.source.name}${design.source.ts ? ' (a TS: Clean/Minimize would move it off the saddle)' : ''}</dd>`}
          </dl>
          ${design.last_ts && html`<div class=${`ts-status ${design.last_ts.ok ? 'ok' : 'failed'}`}>
            ${design.last_ts.ok ? html`
              <b>TS converged.</b> ${design.last_ts.barrier_kcal != null ? html`Barrier ${design.last_ts.barrier_kcal.toFixed(1)} kcal/mol from its IRC. ` : ''}
              ${design.last_ts.irc_note && html`<div class="small mono">${design.last_ts.irc_note}</div>`}
              <p class="small">Its IRC ends are the reactant and product <i>with</i> everything you added (catalyst, solvent): add all three to the graph to compare with the uncatalyzed step.</p>
              <button class="btn small primary" onClick=${tsToGraph}>Add TS + IRC ends to Graph</button>`
            : html`<b>The TS optimization did not converge</b>; the design is back to the structure you submitted.
              <div class="small">${design.last_ts.error || design.last_ts.headline}</div>
              <p class="small">Edit it (e.g. move the catalyst closer to the bonds that change, or start from a better guess) and try again.</p>`}
            <a class="small" href="#" onClick=${(e) => { e.preventDefault(); openJob(design.last_ts.job); }}>Calculation details</a>
          </div>`}
          ${(design.coverage || []).map((w) => html`<p class="warn-box small">${w}</p>`)}
          ${(design.warnings || []).map((w) => html`<p class="level-note small">${w}</p>`)}
          ${info && tool === 'view' && html`<p class="small muted">${info}</p>`}
          <div class="design-actions">
            <button class="btn primary" onClick=${toGraph} disabled=${busy || running}>Add to Graph</button>
            <a class="btn" href="/api/design/xyz" download=${`${design.name || 'design'}.xyz`}>Download xyz</a>
            <button class="btn-link small" onClick=${() => { if (confirm('Start a new design? The current one is not kept (add it to the graph first if you want it).')) attempt(() => api.del('/api/design')); }}>New design…</button>
          </div>
          <details class="advanced"><summary>Load another</summary><${Start} structures=${structures} /></details>
        </aside>
      </div>`}
    </div>`;
}
