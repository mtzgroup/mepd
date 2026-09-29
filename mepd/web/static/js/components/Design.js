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

  // Styles only: the picked atom(s) and what the last edit touched. Cheap,
  // so a pick shows at once however big the molecule is.
  const highlight = useRef(() => {});
  highlight.current = () => {
    const v = viewer.current;
    if (!v || !v.getModel()) return;
    v.setStyle({}, { stick: { radius: 0.14 }, sphere: { scale: 0.26 } });
    if (!liveXyz) {
      if (picked.length) v.addStyle({ index: picked }, { sphere: { scale: 0.42, color: '#e0a100', opacity: 0.85 } });
      if (changed.length) v.addStyle({ index: changed }, { sphere: { scale: 0.34, color: '#3aa76d', opacity: 0.75 } });
    }
    v.render();
  };

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
    if (!liveXyz) {
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
    highlight.current();   // styles + render
  }, [molblock, liveXyz, labels]);

  useEffect(() => { highlight.current(); }, [picked.join(','), changed.join(',')]);

  return html`<div class=${`design-canvas ${clickable ? 'picking' : ''}`} ref=${host}></div>`;
}

// How many structures an xyz text holds (frames with an atom-count line;
// bare "El x y z" lines count as one).
export function countXyzFrames(text) {
  const lines = (text || '').replace(/\r/g, '').split('\n');
  let i = 0, n = 0;
  while (i < lines.length) {
    const t = lines[i].trim();
    if (!t) { i += 1; continue; }
    if (/^\d+$/.test(t)) { n += 1; i += parseInt(t, 10) + 2; continue; }
    return n || 1;
  }
  return n;
}

// An xyz from a file (button or drag-and-drop) or pasted text. Calls
// onXyz(text, name) once read; with `multiple`, several files arrive as one
// multi-frame text, in the order given.
function XyzInput({ onXyz, compact = false, multiple = false }) {
  const [over, setOver] = useState(false);
  const [paste, setPaste] = useState(false);
  const [text, setText] = useState('');
  const file = useRef(null);
  const read = (list) => {
    const files = [...(list || [])].slice(0, multiple ? undefined : 1);
    if (!files.length) return;
    if (files.some((f) => f.size > 5e6)) { toast('That file is too large for an xyz (over 5 MB)', 'error'); return; }
    Promise.all(files.map((f) => f.text())).then((texts) => onXyz(
      texts.map((t) => t.replace(/\s+$/, '')).join('\n') + '\n',
      files.map((f) => f.name.replace(/\.[^.]+$/, '')).join(' → ')));
  };
  return html`<div class=${`xyz-drop ${over ? 'over' : ''} ${compact ? 'compact' : ''}`}
    onDragOver=${(e) => { e.preventDefault(); setOver(true); }} onDragLeave=${() => setOver(false)}
    onDrop=${(e) => { e.preventDefault(); setOver(false); read(e.dataTransfer.files); }}>
    <input type="file" accept=".xyz,.txt,text/plain" ref=${file} style="display:none" multiple=${multiple}
      onChange=${(e) => { read(e.target.files); e.target.value = ''; }} />
    <div class="row">
      <button class="btn" onClick=${() => file.current?.click()}>Upload .xyz</button>
      <span class="small muted">or drop ${multiple ? 'one or two files' : 'a file'} here · <a href="#" onClick=${(e) => { e.preventDefault(); setPaste(!paste); }}>${paste ? 'hide' : 'paste text'}</a></span>
    </div>
    ${paste && html`<div>
      <textarea class="xyz-paste mono" rows=${compact ? 4 : 6} placeholder=${'3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.470\nH 0.000 -0.757 -0.470'}
        value=${text} onInput=${(e) => setText(e.target.value)} />
      <button class="btn small" disabled=${!text.trim()} onClick=${() => onXyz(text, 'pasted')}>Use this xyz</button>
    </div>`}
  </div>`;
}

function Start({ structures }) {
  const [smiles, setSmiles] = useState('');
  const [xyzCharge, setXyzCharge] = useState(0);
  const [many, setMany] = useState(null);   // {xyz, n}: more structures than Design takes
  const [xyzMult, setXyzMult] = useState(1);
  const [sid, setSid] = useState('');
  const [cid, setCid] = useState('');
  const rec = structures[sid];
  const list = Object.values(structures).sort((a, b) => b.created - a.created);
  return html`<div class="design-start">
    <label class="field"><span class="field-label">From SMILES or a reaction SMILES</span>
      <div class="row"><input value=${smiles} placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O, or a reaction: CC(=O)C>>CC(O)=C" onInput=${(e) => setSmiles(e.target.value)}
        onKeyDown=${(e) => e.key === 'Enter' && smiles.trim() && attempt(() => api.post('/api/design/new', { smiles }))} />
      <button class="btn primary" disabled=${!smiles.trim()} onClick=${() => attempt(() => api.post('/api/design/new', { smiles }))}>Build</button></div></label>
    <p class="small muted start-note">A reaction SMILES (reactants>>products, atom map numbers optional) opens the reactant and the
      product side by side: an edit on one is made on the matching atom of the other, so you can vary a reaction and compare barriers.</p>
    <div class="field"><span class="field-label">From XYZ: one structure, or two for a reaction</span>
      <${XyzInput} multiple=${true} onXyz=${(xyz, name) => {
        const n = countXyzFrames(xyz);
        if (n > 2) { setMany({ xyz, n }); return; }
        setMany(null);
        attempt(() => api.post('/api/design/new', { xyz, name, charge: xyzCharge, multiplicity: xyzMult }));
      }} />
      <p class="small muted start-note">Two structures (two files, or two frames in one) open as a reaction: the first is the
        reactant, the second the product, with the same atoms, matched in the order they are listed (or by SLAPMapper).</p>
      ${many && html`<div class="warn-box small">
        <p>${many.n} structures: Design opens one, or two as a reaction. To build a network from many, add them in Explore.</p>
        <button class="btn small primary" onClick=${async () => {
          const added = await attempt(() => api.post('/api/structures', { text: many.xyz, charge: xyzCharge, multiplicity: xyzMult,
            optimize: prefs.get('optimizeOnAdd', true) }), (a) => `Added ${a.length} structure(s) to Explore`);
          if (added) { setMany(null); select({ structures: [...new Set(added.map((a) => a.id))] }); openTab('graph'); }
        }}>Add all ${many.n} to Explore</button></div>`}
      <div class="row small xyz-cm">
        <label>charge <input type="number" value=${xyzCharge} onChange=${(e) => setXyzCharge(parseInt(e.target.value, 10) || 0)} /></label>
        <label>multiplicity <input type="number" min="1" value=${xyzMult} onChange=${(e) => setXyzMult(Math.max(1, parseInt(e.target.value, 10) || 1))} /></label>
        <span class="muted">Set these before loading: bond orders are read from the geometry for this charge.</span>
      </div></div>
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
  const [placeXyz, setPlaceXyz] = useState(null);   // {xyz, name, natoms, charge}
  const [groupSmiles, setGroupSmiles] = useState('');
  const [hessian, setHessian] = useState(() => prefs.get('designHessian', false));
  const [cov, setCov] = useState({ method: '', elements: null });
  const [picked, setPicked] = useState([]);
  const [pickSide, setPickSide] = useState('reactant');   // reaction mode: which molecule `picked` is on
  const [changed, setChanged] = useState([]);
  const [changedP, setChangedP] = useState([]);
  const [linked, setLinked] = useState(() => prefs.get('designLinked', true));
  const rx = design?.reaction || null;
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

  // The molecule edits apply to: the last edit's result as soon as it is
  // back (the store's copy follows over the event stream, a moment later).
  const current = useRef(design?.molblock);
  useEffect(() => { current.current = design?.molblock; }, [design?.molblock]);
  const currentP = useRef(rx?.product?.molblock);
  useEffect(() => { currentP.current = rx?.product?.molblock; }, [rx?.product?.molblock]);
  const amapRef = useRef(rx?.amap);
  useEffect(() => { amapRef.current = rx?.amap; }, [rx?.amap]);
  // What undo restores: the molecule, and in reaction mode the product and the atom match too.
  const snapshot = () => ({ mb: current.current, pmb: currentP.current, amap: amapRef.current });
  const busyRef = useRef(false);
  const queue = useRef([]);            // edits clicked while another was being applied
  const pumpRef = useRef(() => {});
  const [queued, setQueued] = useState(0);

  const apply = async (fn, { record = true } = {}) => {
    if (busyRef.current) return null;
    busyRef.current = true;
    setBusy(true);
    const before = snapshot();
    const out = await attempt(fn);
    busyRef.current = false;
    setBusy(false);
    // After this edit's own follow-up (its caller's `done`), start the next queued one.
    setTimeout(() => pumpRef.current(), 0);
    if (out && out.molblock !== undefined) {
      current.current = out.molblock;
      currentP.current = out.reaction?.product?.molblock;
      amapRef.current = out.reaction?.amap;
      if (record && before.mb && (before.mb !== out.molblock || before.pmb !== currentP.current)) {
        history.current.undo.push(before);
        history.current.redo = [];
        bump((x) => x + 1);
      }
      setChanged(out.changed || []);
      setChangedP(out.changed_product || []);
      (out.warnings || []).forEach((w) => toast(w, 'info', 7000));
      (out.reaction?.product?.warnings || []).forEach((w) => toast(`Product: ${w}`, 'info', 7000));
      if (out.note) toast(out.note, 'info', 6000);
    }
    return out;
  };
  // The molecule's atoms, in order: [element, x, y, z] (V2000 molblock).
  const atomsOf = (mb) => {
    const lines = (mb || '').split('\n');
    const n = parseInt((lines[3] || '').slice(0, 3), 10);
    if (!Number.isFinite(n)) return null;
    return lines.slice(4, 4 + n).map((l) => { const f = l.trim().split(/\s+/); return [f[3], +f[0], +f[1], +f[2]]; });
  };
  // An edit refers to atoms by index. Queued behind another edit it runs
  // only if those indices still hold the same atoms: same element, within
  // 0.5 Å of where they were (edits relax only nearby atoms, a little).
  // An edit that renumbered them (a deleted heavy atom shifts the rest)
  // would make it act on other atoms, so it is skipped.
  const stillValid = (item) => {
    const now = atomsOf(item.side === 'product' ? currentP.current : current.current);
    if (!now || !item.atoms) return false;
    return item.indices.every((i) => {
      const a = now[i], b = item.atoms[i];
      return a && b && a[0] === b[0] && Math.hypot(a[1] - b[1], a[2] - b[2], a[3] - b[3]) < 0.5;
    });
  };
  const runEdit = async (item) => {
    const out = await apply(() => api.post('/api/design/edit', { op: item.op, side: item.side, linked }));
    item.done?.();
    return out;
  };
  // The queued edits, one after another, once nothing is being applied
  // (whatever started the last one: a click, Clean, Fix H, undo, ...).
  const pump = async () => {
    while (!busyRef.current && queue.current.length) {
      const next = queue.current.shift();
      setQueued(queue.current.length);
      if (!stillValid(next)) {
        next.done?.();
        toast('Skipped an edit clicked while the previous one was applying: that edit changed the atoms. Pick them again.', 'info', 6000);
        continue;
      }
      await runEdit(next);
    }
  };
  pumpRef.current = pump;
  const edit = (op, done, side = 'reactant') => {
    const item = { op, done, side, atoms: atomsOf(side === 'product' ? currentP.current : current.current),
      indices: ['atom', 'a', 'b'].filter((k) => op[k] != null).map((k) => op[k]) };
    if (busyRef.current) {
      queue.current.push(item);
      setQueued(queue.current.length);
      return null;
    }
    return runEdit(item);
  };

  const onAtom = (idx, side = 'reactant') => {
    if (running) { toast('Wait for the minimization to finish (or stop it) before editing', 'info'); return; }
    const e = (op, done) => edit(op, done, side);
    if (tool === 'view') { setPickSide(side); setPicked([idx]); setInfo(`${rx ? `${side === 'product' ? 'Product' : 'Reactant'} a` : 'A'}tom ${idx + 1}`); return; }
    if (tool === 'element') e({ op: 'element', atom: idx, element });
    else if (tool === 'add') e({ op: 'add', atom: idx, element });
    else if (tool === 'group') e(groupSmiles.trim() ? { op: 'group', atom: idx, smiles: groupSmiles.trim() } : { op: 'group', atom: idx, group });
    else if (tool === 'place') e(placeXyz ? { op: 'place', atom: idx, xyz: placeXyz.xyz, charge: placeXyz.charge, count: placeCount }
      : placeSmiles.trim() ? { op: 'place', atom: idx, smiles: placeSmiles.trim(), count: placeCount }
      : { op: 'place', atom: idx, species: placeWhat, count: placeCount });
    else if (tool === 'delete') e({ op: 'delete', atom: idx });
    else if (tool === 'charge') e({ op: 'charge', atom: idx, delta: order === 0 ? -1 : 1 });
    else if (tool === 'bond') {
      // A new pair unless exactly its first atom is picked (a pair being applied stays lit).
      // Both atoms of a bond are on the same molecule.
      if (side !== pickSide || picked.length !== 1 || picked[0] === idx) { setPickSide(side); setPicked([idx]); return; }
      const a = picked[0];
      const pair = `${a},${idx}`;
      setPicked([a, idx]);   // both stay lit until the edit is back
      e({ op: 'bond', a, b: idx, order }, () => setPicked((p) => (p.join(',') === pair ? [] : p)));
    }
  };
  // Reaction mode: an atom picked on one side lights its match on the other.
  const counterpart = (idxs, from) => {
    const amap = rx?.amap || [];
    if (from === 'reactant') return idxs.map((i) => amap[i]).filter((j) => j != null && j >= 0);
    return idxs.map((j) => amap.indexOf(j)).filter((i) => i >= 0);
  };
  const pickedR = !rx || pickSide === 'reactant' ? picked : counterpart(picked, 'product');
  const pickedP = rx ? (pickSide === 'product' ? picked : counterpart(picked, 'reactant')) : [];

  const restore = (snap) => api.put('/api/design', snap.pmb && rx
    ? { molblock: snap.mb, reaction: { product_molblock: snap.pmb, amap: snap.amap } } : { molblock: snap.mb });
  const undo = () => {
    const h = history.current;
    const prev = h.undo.pop();
    if (!prev) return;
    h.redo.push(snapshot());
    bump((x) => x + 1);
    apply(() => restore(prev), { record: false });
  };
  const redo = () => {
    const h = history.current;
    const next = h.redo.pop();
    if (!next) return;
    h.undo.push(snapshot());
    bump((x) => x + 1);
    apply(() => restore(next), { record: false });
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
    if (rec.edge) {
      toast(`Added the reactant, the product and their edge to Explore${rec.new.length < 2 ? ' (as conformers of nodes already there)' : ''}`, 'ok', 6000,
        { label: 'Show', run: () => { select({ structures: [rec.reactant.id, rec.product.id] }); openTab('graph'); } });
      return;
    }
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
  const search = async (op) => {
    const out = await attempt(() => api.post('/api/design/search', { op, profile: levelProfile ?? null }));
    if (out?.jobs?.length) {
      toast(`${op === 'ts' ? 'TS search' : 'Reaction channels'} started at ${level?.label ?? 'the workspace level'} (ends are minimized first)`, 'ok', 7000,
        { label: 'Open', run: () => openJob(out.jobs[0].id) });
    }
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
            <${Steps} steps=${[`Choose what to add (now ${placeXyz ? placeXyz.name : placeSmiles.trim() || placeWhat}${placeCount > 1 ? ` ×${placeCount}` : ''}): a preset, any SMILES, or an xyz file.`,
              placeXyz ? 'Click the atom in the 3D view it should go next to: the file\'s first atom goes beside that atom, the structure kept exactly as in the file.'
                : 'Click the atom in the 3D view it should go next to: it lands in the free space beside that atom, not bonded (click a hydrogen to H-bond to it).']} />
            ${placeXyz ? html`<div class="xyz-chosen">
                <span><b>${placeXyz.name}</b> <span class="muted small">${placeXyz.natoms} atoms, from xyz</span></span>
                <label class="small">charge <input type="number" value=${placeXyz.charge}
                  onChange=${(e) => setPlaceXyz({ ...placeXyz, charge: parseInt(e.target.value, 10) || 0 })} /></label>
                <button class="btn-icon" title="Use a preset or SMILES instead" onClick=${() => setPlaceXyz(null)}>✕</button>
              </div>`
              : html`<div>
                <input class="smiles-in" placeholder="any molecule or ion, as SMILES (e.g. [Pd], [Cl-], O=C=O)" value=${placeSmiles}
                  onInput=${(e) => setPlaceSmiles(e.target.value)} />
                <div class="palette groups">
                  ${species.map((g) => html`<button class=${`el wide ${!placeSmiles.trim() && placeWhat === g ? 'on' : ''}`} onClick=${() => { setPlaceWhat(g); setPlaceSmiles(''); }}>${g}</button>`)}
                </div>
                <${XyzInput} compact onXyz=${(xyz, name) => {
                  const n = parseInt(xyz.trimStart().split('\n', 1)[0], 10);
                  const natoms = n > 0 ? n : xyz.split('\n').filter((l) => l.trim().split(/\s+/).length >= 4).length;
                  if (!natoms) { toast('No atoms found in that xyz', 'error'); return; }
                  setPlaceXyz({ xyz, name, natoms, charge: 0 });
                }} />
              </div>`}
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
          ${rx ? html`<div class="design-pair">
              ${[['reactant', 'Reactant', design.molblock, design.smiles, pickedR, changed], ['product', 'Product', rx.product.molblock, rx.product.smiles, pickedP, changedP]]
                .map(([side, label, mb, smi, pk, ch]) => html`<div class=${`design-side-pane ${pickSide === side && picked.length ? 'active' : ''}`}>
                  <div class="pane-head"><b>${label}</b> <span class="mono small">${smi || ''}</span></div>
                  <${Canvas} molblock=${mb} liveXyz=${null} picked=${pk} changed=${ch} onAtom=${(i) => onAtom(i, side)} labels=${labels} clickable=${tool !== 'view'} />
                </div>`)}
              <div class="pair-link"><button class=${`btn small ${linked ? 'on' : ''}`} title=${linked ? 'Edits are made on both molecules (on the matching atom). Click to edit one side alone.' : 'Edits change only the molecule you click. Click to edit both.'}
                onClick=${() => { setLinked(!linked); prefs.set('designLinked', !linked); }}>${linked ? '⇄ Edit both' : '→ One side'}</button></div>
            </div>`
          : html`<${Canvas} molblock=${design.molblock} liveXyz=${liveXyz} picked=${picked} changed=${changed} onAtom=${onAtom} labels=${labels} clickable=${tool !== 'view'} />`}
          <div class="design-hint small">${running ? html`${running.op === 'design-tsopt' ? 'Optimizing as a TS (then IRC)' : 'Minimizing'} at ${level?.label ?? 'the workspace level'}… <a href="#" onClick=${(e) => { e.preventDefault(); openJob(running.id); }}>details</a>`
            : busy ? `Applying the edit…${queued ? ` (${queued} more queued)` : ''}`
              : tool === 'bond' && picked.length === 1 ? `Atom ${picked[0] + 1} picked: click the second atom.` : toolHelp}</div>
          <div class="design-bar">
            <button class="btn small" disabled=${!history.current.undo.length || busy} onClick=${undo} title="Undo (Ctrl+Z)">Undo</button>
            <button class="btn small" disabled=${!history.current.redo.length || busy} onClick=${redo} title="Redo (Ctrl+Shift+Z)">Redo</button>
            <button class="btn small" disabled=${busy || running} onClick=${() => apply(() => api.post('/api/design/edit', { op: { op: 'hydrogens' } }))} title="Re-add hydrogens everywhere from valences">Fix H</button>
            <button class="btn small" disabled=${busy || running} onClick=${() => apply(() => api.post('/api/design/clean'))}
              title="Quick force-field cleanup (MMFF94, else UFF). Moves every atom: not for a transition state.">Clean (MMFF)</button>
            ${!rx && html`<button class="btn small" disabled=${busy || running} onClick=${tsopt}
              title=${`Treat this structure as a TS guess: saddle optimization at ${level?.label ?? 'the workspace level'}, then an IRC. If it doesn't converge, the design goes back to what you submitted.`}>Optimize as TS</button>
            <button class="btn small primary" disabled=${busy || running} onClick=${minimize}
              title=${`Minimize at ${level?.label ?? 'the workspace level'} (mepd optimize); the result replaces the design`}>Minimize</button>
            <label class="small" title="After minimizing, check for imaginary frequencies (and try to push off a saddle). Slow for large or floppy structures, e.g. with explicit solvent or ions, which rarely pass a strict check.">
              <input type="checkbox" checked=${hessian} onChange=${(e) => { setHessian(e.target.checked); prefs.set('designHessian', e.target.checked); }} /> Hessian check</label>`}
            <label class="small"><input type="checkbox" checked=${labels} onChange=${(e) => setLabels(e.target.checked)} /> atom numbers</label>
          </div>
        </div>
        <aside class="design-side">
          <input class="title-input" value=${design.name || ''} placeholder="Name"
            onChange=${(e) => attempt(() => api.put('/api/design', { name: e.target.value }))} />
          ${design.smiles && html`<div class="smiles">${design.smiles}${rx ? html` >> ${rx.product.smiles}` : ''}</div>`}
          ${rx && html`<div class="rxn-box small">
            ${!rx.balanced && html`<p class="warn-box small">The two sides no longer have the same atoms (an edit was made on one side only).
              Make them match (or undo) before searching.</p>`}
            <p class="muted">Atoms matched ${rx.mapping?.source === 'given' ? 'from the map numbers in your reaction SMILES'
              : rx.mapping?.source === 'xyz-order' ? 'in the order the two xyz list them' : 'by SLAPMapper'}; new atoms
              by their neighbours and the closest fit in 3D (endpoint RMSD). Fast, not exhaustive: the path search checks other mappings with atom mapping on.</p>
          </div>`}
          <dl class="props">
            <dt>Formula</dt><dd>${design.formula} (${design.natoms} atoms)${rx && rx.product.formula !== design.formula ? html` · product ${rx.product.formula}` : ''}</dd>
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
              <button class="btn small primary" onClick=${tsToGraph}>Add TS + IRC ends to Explore</button>`
            : html`<b>The TS optimization did not converge</b>; the design is back to the structure you submitted.
              <div class="small">${design.last_ts.error || design.last_ts.headline}</div>
              <p class="small">Edit it (e.g. move the catalyst closer to the bonds that change, or start from a better guess) and try again.</p>`}
            <a class="small" href="#" onClick=${(e) => { e.preventDefault(); openJob(design.last_ts.job); }}>Calculation details</a>
          </div>`}
          ${(design.coverage || []).map((w) => html`<p class="warn-box small">${w}</p>`)}
          ${(design.warnings || []).map((w) => html`<p class="level-note small">${w}</p>`)}
          ${info && tool === 'view' && html`<p class="small muted">${info}</p>`}
          <div class="design-actions">
            ${rx && html`<button class="btn primary" onClick=${() => search('ts')} disabled=${busy || !rx.balanced}
              title=${`Add both ends and their edge to Explore, then find the TS between them at ${level?.label ?? 'the workspace level'}`}>Search TS</button>
              <button class="btn" onClick=${() => search('channels')} disabled=${busy || !rx.balanced}
                title="Sample conformers and atom mappings of both ends and find every distinct channel">Reaction channels</button>`}
            <button class=${`btn ${rx ? '' : 'primary'}`} onClick=${toGraph} disabled=${busy || running || (rx && !rx.balanced)}>Add to Explore</button>
            <a class="btn" href="/api/design/xyz" download=${`${design.name || 'design'}${rx ? '_reactant' : ''}.xyz`}>Download xyz${rx ? ' (reactant)' : ''}</a>
            ${rx && html`<a class="btn" href="/api/design/xyz?side=product" download=${`${design.name || 'design'}_product.xyz`}>Download xyz (product)</a>`}
            <button class="btn-link small" onClick=${() => { if (confirm('Start a new design? The current one is not kept (add it to the graph first if you want it).')) attempt(() => api.del('/api/design')); }}>New design…</button>
          </div>
          <details class="advanced"><summary>Load another</summary><${Start} structures=${structures} /></details>
        </aside>
      </div>`}
    </div>`;
}
