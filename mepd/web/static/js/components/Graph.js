// The reaction graph: every library structure is a node, every declared
// pair an edge. Edges are coloured by what's been computed on them.
import { html, useEffect, useRef, useState } from '../lib.js';
import cytoscape from '../../vendor/cytoscape.esm.min.js';
import { api, attempt, deleteSelection } from '../api.js';
import { clearSelection, openJob, openTab, prefs, select, set, state, useStore } from '../store.js';
import { depictUrl, edgeStatus, edgeStatusKey, playgroundFitMargins, reactionOfEdge } from '../util.js';
import { uploadFiles } from './Library.js';

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function stylesheet() {
  return [
    { selector: 'node', style: {
      shape: 'round-rectangle', width: 96, height: 72,
      'background-color': css('--node-bg'), 'border-width': 1, 'border-color': css('--border-strong'), 'corner-radius': 4,
      'background-image': 'data(img)', 'background-fit': 'contain', 'background-clip': 'node',
      'background-image-containment': 'inside', 'background-width': '90%', 'background-height': '90%',
      label: 'data(label)', 'font-size': 12, 'font-family': css('--font-ui'), color: css('--heading'),
      'text-valign': 'bottom', 'text-margin-y': 6, 'text-wrap': 'ellipsis', 'text-max-width': 140,
      'text-background-color': css('--bg'), 'text-background-opacity': 0.85, 'text-background-padding': 2,
    } },
    { selector: 'node[?noimg]', style: { 'background-image': 'none', label: 'data(label)', 'text-valign': 'center', 'text-margin-y': 0 } },
    { selector: 'node:selected', style: { 'border-width': 3, 'border-color': css('--accent') } },
    { selector: 'node.connect-source', style: { 'border-width': 3, 'border-style': 'dashed', 'border-color': css('--accent') } },
    { selector: 'edge', style: {
      width: 2, 'curve-style': 'bezier', 'line-color': css('--edge-idle'), 'target-arrow-shape': 'triangle',
      'target-arrow-color': css('--edge-idle'), 'arrow-scale': 1.1, 'line-style': 'dashed', 'line-dash-pattern': [6, 4],
      label: 'data(label)', 'font-size': 11, 'font-family': css('--font-mono'), color: css('--text'),
      'text-background-color': css('--bg'), 'text-background-opacity': 0.9, 'text-background-padding': 3,
      'text-background-shape': 'roundrectangle',
    } },
    { selector: 'edge[status = "done"]', style: { 'line-style': 'solid', 'line-color': css('--ok'), 'target-arrow-color': css('--ok') } },
    { selector: 'edge[status = "running"]', style: { 'line-style': 'solid', 'line-color': css('--accent'), 'target-arrow-color': css('--accent'), width: 3.5 } },
    { selector: 'edge[status = "queued"]', style: { 'line-color': css('--warn'), 'target-arrow-color': css('--warn') } },
    { selector: 'edge[status = "failed"]', style: { 'line-color': css('--danger'), 'target-arrow-color': css('--danger') } },
    { selector: 'edge:selected', style: { width: 5, 'underlay-color': css('--accent'), 'underlay-opacity': 0.25, 'underlay-padding': 5 } },
    // Reactions with any number of species (nanoreactor): a small dot joined
    // to its reactants and products by spokes; shuttles hang off it dashed.
    { selector: 'node.rxn', style: {
      shape: 'ellipse', width: 22, height: 22, 'background-image': 'none', 'background-color': css('--edge-idle'),
      'border-width': 0, label: 'data(label)', 'font-size': 10, 'font-family': css('--font-mono'), color: css('--text'),
      'text-valign': 'bottom', 'text-margin-y': 4, 'text-wrap': 'none',
    } },
    { selector: 'node.rxn[status = "done"]', style: { 'background-color': css('--ok') } },
    { selector: 'node.rxn[status = "running"]', style: { 'background-color': css('--accent') } },
    { selector: 'node.rxn[status = "queued"]', style: { 'background-color': css('--warn') } },
    { selector: 'node.rxn[status = "failed"]', style: { 'background-color': css('--danger') } },
    { selector: 'node.rxn:selected', style: { 'border-width': 3, 'border-color': css('--accent') } },
    { selector: 'edge.spoke', style: { 'line-style': 'solid', width: 1.5, 'target-arrow-shape': 'none', label: 'data(label)',
      'curve-style': 'bezier' } },
    { selector: 'edge.spoke.out', style: { 'target-arrow-shape': 'triangle', 'arrow-scale': 0.9 } },
    { selector: 'edge.spoke.shuttle', style: { 'line-style': 'dashed', 'line-dash-pattern': [3, 3], opacity: 0.8 } },
    // A node being pointed at from the list (Library double-click): pulses a few times.
    { selector: 'node.flash', style: { 'underlay-color': css('--accent'), 'underlay-opacity': 0.35, 'underlay-padding': 16,
      'underlay-shape': 'round-rectangle', 'border-width': 3, 'border-color': css('--accent') } },
    // View filters (see applyView).
    { selector: '.vhidden', style: { display: 'none' } },
    { selector: '.dim', style: { opacity: 0.16 } },
    { selector: 'node.collapsed', style: { 'border-width': 3, 'border-style': 'double', 'border-color': css('--accent'),
      'font-weight': 600 } },
  ];
}

// Fit everything in view, but never blow a two-node graph up to poster size.
function fitCapped(c, padding = 70) {
  if (!c.nodes().length) return;
  c.fit(undefined, padding);
  if (c.zoom() > 1.1) { c.zoom(1.1); c.center(); }
  c.panBy({ x: 0, y: 22 });   // clear of the floating toolbar
}

// Fit everything into the canvas minus margins (px) kept free for overlays.
function fitInto(c, m) {
  const eles = c.elements();
  if (!eles.length) return;
  const bb = eles.boundingBox();
  const aw = Math.max(80, c.width() - m.l - m.r), ah = Math.max(80, c.height() - m.t - m.b);
  const z = Math.max(c.minZoom(), Math.min(1.1, aw / Math.max(1, bb.w), ah / Math.max(1, bb.h)));
  c.zoom(z);
  c.pan({ x: m.l + (aw - bb.w * z) / 2 - bb.x1 * z, y: m.t + (ah - bb.h * z) / 2 - bb.y1 * z });
}

// Single-structure calculations whose running target "breathes" in the graph.
const ANIMATED_OPS = new Set(['hessian-sample', 'hessian-global', 'nanoreactor', 'graph-enumeration', 'tsopt', 'optimize', 'vri']);
const NODE_W = 96, NODE_H = 72;

function activeStructureKey(jobs) {
  const ids = new Set();
  for (const j of Object.values(jobs)) {
    if (j.status === 'running' && ANIMATED_OPS.has(j.op)) j.targets.structures.forEach((id) => ids.add(id));
  }
  return [...ids].sort().join(',');
}

// Where a node found from `parent` (e.g. a species a network expansion just
// reported) should settle: next to its parent, on the side away from the
// parent's other neighbours, at the first free spot of a widening fan.
function spawnSpot(c, parent, taken) {
  const p = parent.position();
  const others = parent.neighborhood('node');
  let ax = 1, ay = 0;
  if (others.nonempty()) {
    let sx = 0, sy = 0;
    others.forEach((n) => { sx += n.position('x') - p.x; sy += n.position('y') - p.y; });
    const len = Math.hypot(sx, sy);
    if (len > 1) { ax = -sx / len; ay = -sy / len; }
  }
  const base = Math.atan2(ay, ax);
  const busy = (x, y) => c.nodes().some((n) => Math.abs(n.position('x') - x) < 120 && Math.abs(n.position('y') - y) < 100)
    || taken.some((q) => Math.abs(q.x - x) < 120 && Math.abs(q.y - y) < 100);
  for (const r of [190, 300, 420]) {
    for (let k = 0; k < 12; k += 1) {
      const a = base + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * (Math.PI / 6);
      const x = p.x + r * Math.cos(a), y = p.y + r * Math.sin(a);
      if (!busy(x, y)) return { x, y };
    }
  }
  return { x: p.x + 190 * Math.cos(base), y: p.y + 190 * Math.sin(base) + 40 * taken.length };
}

const HARTREE_KCAL = 627.509;

// Reactions (workspace.reactions) as graph elements: the dot, and one spoke
// per distinct species (coefficient as its label; shuttles once, dashed).
function reactionElements(r, structures, edges, jobs) {
  const count = (ids) => ids.reduce((m, id) => ({ ...m, [id]: (m[id] || 0) + 1 }), {});
  const shut = count(r.shuttles || []);
  const inn = count(r.reactants), out = count(r.products);
  const edge = r.edge ? edges[r.edge] : null;
  const st = edge ? edgeStatus(edge, jobs) : { status: 'idle' };
  const status = st.status === 'proposed' ? 'idle' : st.status;
  let label = '';
  if (st.barrier != null) label = `${st.warning ? '⚠ ' : ''}${st.barrier.toFixed(1)}`;
  else if (st.status === 'running') label = 'running…';
  else if (r.delta_e_kcal != null) label = `ΔE ${r.delta_e_kcal >= 0 ? '+' : ''}${r.delta_e_kcal.toFixed(1)}`;
  const els = [{ group: 'nodes', classes: 'rxn', data: { id: r.id, label, status, rxn: 1 } }];
  const spoke = (sid, n, kind) => {
    if (!structures[sid]) return;
    els.push({ group: 'edges', classes: `spoke ${kind}`, data: {
      id: `${r.id}|${kind}|${sid}`, rxn: r.id, status, label: n > 1 ? String(n) : '',
      source: kind === 'out' ? r.id : sid, target: kind === 'out' ? sid : r.id } });
  };
  for (const [sid, n] of Object.entries(inn)) if (n - (shut[sid] || 0) > 0) spoke(sid, n - (shut[sid] || 0), 'in');
  for (const [sid, n] of Object.entries(out)) if (n - (shut[sid] || 0) > 0) spoke(sid, n - (shut[sid] || 0), 'out');
  for (const [sid, n] of Object.entries(shut)) spoke(sid, n, 'shuttle');
  return els;
}

// A reaction's subsystem ends are hidden structures; their edge is the
// reaction's dot (selecting the dot selects that edge).
const isComplex = (s) => s?.role === 'complex';

const VIEW_DEFAULT = { q: '', only: false, maxRel: null, hideProposed: false, hideFailed: false, hideTS: false, focusHops: 0 };

function nodeLabel(s) {
  const nConf = (s.conformers || []).length;
  return nConf > 1 ? `${s.name} · ${nConf} conf.` : s.name;
}

// structure id -> ids of the structures found from it (origin.parent), e.g.
// a network expansion's species.
function childrenOf(structures) {
  const kids = {};
  for (const s of Object.values(structures)) {
    const p = s.origin?.parent;
    if (p && p !== s.id && structures[p]) (kids[p] ||= []).push(s.id);
  }
  return kids;
}

// Which structures and edges the view shows. Returns {hide, dim, folded}:
// sets of ids, and collapsed node -> how many structures it folds away.
// Selected structures always stay visible.
function computeView(workspace, view, collapsed, selected, focusIds, cyc) {
  const { structures, edges } = workspace;
  const kids = childrenOf(structures);
  const hide = new Set(), dim = new Set(), folded = {};
  for (const cid of collapsed) {
    if (!structures[cid] || hide.has(cid)) continue;
    const stack = [...(kids[cid] || [])];
    let n = 0;
    while (stack.length) {
      const x = stack.pop();
      if (x === cid || hide.has(x)) continue;
      hide.add(x); n += 1;
      stack.push(...(kids[x] || []));
    }
    if (n) folded[cid] = n;
  }
  // Energy above the lowest isomer (same formula, charge, spin and level of theory).
  const low = {};
  const isoKey = (s) => `${s.formula}|${s.charge}|${s.multiplicity}|${s.level?.key || ''}`;
  for (const s of Object.values(structures)) {
    if (s.energy == null || s.role === 'ts') continue;
    const k = isoKey(s);
    if (low[k] == null || s.energy < low[k]) low[k] = s.energy;
  }
  const q = view.q.trim().toLowerCase();
  for (const s of Object.values(structures)) {
    if (selected.has(s.id)) continue;
    if (view.hideTS && s.role === 'ts') hide.add(s.id);
    if (view.maxRel != null && s.energy != null && s.role !== 'ts' && low[isoKey(s)] != null
        && (s.energy - low[isoKey(s)]) * HARTREE_KCAL > view.maxRel) hide.add(s.id);
    if (q && ![s.name, s.smiles, s.formula].some((t) => (t || '').toLowerCase().includes(q))) (view.only ? hide : dim).add(s.id);
  }
  if (focusIds && focusIds.length && view.focusHops > 0 && cyc) {
    let near = cyc.collection();
    focusIds.forEach((id) => { near = near.union(cyc.getElementById(id)); });
    for (let k = 0; k < view.focusHops; k += 1) near = near.union(near.closedNeighborhood('node'));
    const keep = new Set(near.map((n) => n.id()));
    for (const id of Object.keys(structures)) if (!keep.has(id)) hide.add(id);
  }
  for (const s of Object.values(structures)) if (selected.has(s.id)) hide.delete(s.id);
  for (const r of Object.values(workspace.reactions || {})) {
    const ids = [...r.reactants, ...r.products];
    if (ids.some((id) => hide.has(id))) hide.add(r.id);
    else if (ids.every((id) => dim.has(id))) dim.add(r.id);
  }
  for (const e of Object.values(edges)) {
    const st = edgeStatus(e, state.jobs).status;
    if (view.hideProposed && e.origin?.proposed && !['done', 'running', 'queued'].includes(st)) hide.add(e.id);
    if (view.hideFailed && st === 'failed') hide.add(e.id);
    if (dim.has(e.source) && dim.has(e.target)) dim.add(e.id);
  }
  return { hide, dim, folded, kids };
}

function ViewPanel({ view, setView, collapsed, setCollapsed, kids, shown, total, onClose, focusCount }) {
  const up = (patch) => setView({ ...view, ...patch });
  const roots = Object.keys(kids);
  const active = JSON.stringify(view) !== JSON.stringify(VIEW_DEFAULT) || collapsed.length;
  return html`<div class="graph-view-panel" onClick=${(e) => e.stopPropagation()}>
    <div class="gv-head"><b>View</b> <span class="small muted">showing ${shown} of ${total} structures</span>
      <button class="btn-icon" onClick=${onClose} title="Close">✕</button></div>
    <label class="gv-row"><input type="search" placeholder="Find by name, SMILES or formula" value=${view.q}
      onInput=${(e) => up({ q: e.target.value })} /></label>
    <label class="gv-row small"><input type="checkbox" checked=${view.only} onChange=${(e) => up({ only: e.target.checked })} />
      only matches (else the rest is faded)</label>
    <div class="gv-row small">
      <span>Hide minima more than</span>
      <input type="number" class="tiny" min="0" step="5" placeholder="—" value=${view.maxRel ?? ''}
        onInput=${(e) => up({ maxRel: e.target.value === '' ? null : Math.max(0, +e.target.value) })} />
      <span>kcal/mol above the lowest isomer</span>
    </div>
    <label class="gv-row small"><input type="checkbox" checked=${view.hideProposed} onChange=${(e) => up({ hideProposed: e.target.checked })} />
      hide proposed reactions not searched yet</label>
    <label class="gv-row small"><input type="checkbox" checked=${view.hideFailed} onChange=${(e) => up({ hideFailed: e.target.checked })} />
      hide failed searches</label>
    <label class="gv-row small"><input type="checkbox" checked=${view.hideTS} onChange=${(e) => up({ hideTS: e.target.checked })} />
      hide transition-state structures</label>
    <div class="gv-row small">
      <span>Only the selection and</span>
      <select value=${view.focusHops} onChange=${(e) => up({ focusHops: +e.target.value })}>
        <option value="0">everything</option><option value="1">its neighbours</option>
        <option value="2">2 steps out</option><option value="3">3 steps out</option>
      </select>
      ${view.focusHops > 0 && !focusCount && html`<span class="muted">(select structures first)</span>`}
    </div>
    <div class="gv-sep"></div>
    <p class="small muted">Double-click a structure to fold away everything found from it (e.g. an expansion's species);
      double-click again to unfold. Folded structures have a double border and a +N count.</p>
    <div class="gv-row">
      <button class="btn small" disabled=${!roots.length} onClick=${() => setCollapsed([...new Set([...collapsed, ...roots])])}>Fold all</button>
      <button class="btn small" disabled=${!collapsed.length} onClick=${() => setCollapsed([])}>Unfold all</button>
      <button class="btn small" disabled=${!active} onClick=${() => { setView({ ...VIEW_DEFAULT }); setCollapsed([]); }}>Reset view</button>
    </div>
  </div>`;
}

function edgeLabel(e, st) {
  const parts = [];
  if (st.barrier != null) parts.push(`${st.warning ? '⚠ ' : ''}${st.barrier.toFixed(1)}`);
  else if (st.barrierUnverified != null) parts.push(`≈${st.barrierUnverified.toFixed(1)}?`);
  else if (st.status === 'running') parts.push('running…');
  else if (st.status === 'queued') parts.push('queued');
  else if (st.status === 'failed') parts.push('failed');
  if (e.label && !/^Channel|IRC$/.test(e.label)) parts.unshift(e.label);
  return parts.join(' · ');
}

export function Graph() {
  const host = useRef(null);
  const cy = useRef(null);
  const connectFrom = useRef(null);
  const saveTimer = useRef(null);
  const flashTimer = useRef(null);
  const workspace = useStore((s) => s.workspace);
  const statusKey = useStore((s) => edgeStatusKey(s.jobs));
  const activeKey = useStore((s) => activeStructureKey(s.jobs));
  const glowing = useRef(new Set());    // nodes currently wearing the 'calculation running' glow
  const arranging = useRef(false);      // a layout is animating
  const pendingArrange = useRef(false); // new nodes arrived while the graph was hidden
  const arrangeRef = useRef(() => {});
  const selection = useStore((s) => s.selection);
  const connectMode = useStore((s) => s.connectMode);
  const howToHidden = useStore((s) => s.howToHidden);
  const [drag, setDrag] = useState(false);
  const [view, setViewState] = useState(() => ({ ...VIEW_DEFAULT, ...prefs.get('graphView', {}) }));
  const [collapsed, setCollapsedState] = useState(() => prefs.get('graphCollapsed', []));
  const [panel, setPanel] = useState(false);
  const setPanelRef = useRef(setPanel);
  const [counts, setCounts] = useState({ shown: 0, total: 0, kids: {} });
  const setView = (v) => { setViewState(v); prefs.set('graphView', v); };
  const setCollapsed = (l) => { setCollapsedState(l); prefs.set('graphCollapsed', l); };
  const collapsedRef = useRef(collapsed);
  collapsedRef.current = collapsed;
  const setCollapsedRef = useRef(setCollapsed);
  setCollapsedRef.current = setCollapsed;

  const savePositions = (c) => {
    const pos = {};
    c.nodes().forEach((n) => { pos[n.id()] = n.position(); });
    api.put('/api/positions', pos).catch(() => {});
  };
  const nSelected = selection.structures.length + selection.edges.length;

  // --- init once
  useEffect(() => {
    const c = cytoscape({
      container: host.current, style: stylesheet(), minZoom: 0.15, maxZoom: 3,
      boxSelectionEnabled: true, selectionType: 'additive',
    });
    cy.current = c;
    // Labels are drawn on a canvas: redraw once the web fonts have arrived.
    document.fonts?.ready.then(() => { if (cy.current === c) c.style(stylesheet()); });

    c.on('tap', (evt) => {
      const additive = evt.originalEvent && (evt.originalEvent.shiftKey || evt.originalEvent.metaKey || evt.originalEvent.ctrlKey);
      if (evt.target === c) {
        if (connectFrom.current) { connectFrom.current = null; c.nodes().removeClass('connect-source'); }
        clearSelection();
        return;
      }
      const id = evt.target.id();
      if (evt.target.isNode() && state.connectMode) {
        if (!connectFrom.current) {
          connectFrom.current = id;
          evt.target.addClass('connect-source');
        } else if (connectFrom.current !== id) {
          const src = connectFrom.current;
          connectFrom.current = null;
          c.nodes().removeClass('connect-source');
          attempt(() => api.post('/api/edges', { source: src, target: id })).then((edge) => {
            if (edge) select({ edges: [edge.id] });
          });
        }
        return;
      }
      const rid = evt.target.data('rxn') ? (evt.target.isNode() ? id : evt.target.data('rxn')) : null;
      if (rid) {
        const r = state.workspace.reactions?.[rid];
        if (r?.edge && state.workspace.edges[r.edge]) select({ edges: [r.edge] }, additive);
        else if (r) select({ structures: [...new Set([...r.reactants, ...r.products])] }, additive);
        return;
      }
      if (evt.target.isNode()) select({ structures: [id] }, additive);
      else select({ edges: [id] }, additive);
    });
    c.on('dbltap', 'node', (evt) => {
      const id = evt.target.id();
      if (evt.target.data('rxn')) {
        // A reaction's dot: open the calculation behind its barrier.
        const r = state.workspace.reactions?.[id];
        const edge = r?.edge && state.workspace.edges[r.edge];
        const job = edge && edgeStatus(edge, state.jobs).barrierJob;
        if (job) openJob(job);
        return;
      }
      const kids = childrenOf(state.workspace.structures);
      const cur = collapsedRef.current;
      if (cur.includes(id)) setCollapsedRef.current(cur.filter((x) => x !== id));
      else if (kids[id]?.length) setCollapsedRef.current([...cur, id]);
    });
    c.on('boxend', () => {
      // Box selection: take whatever cytoscape selected, in any order.
      setTimeout(() => {
        const nodes = c.nodes(':selected').filter((n) => !n.data('rxn')).map((n) => n.id());
        const edges = nodes.length ? [] : c.edges(':selected').map((e) => e.id());
        select({ structures: nodes, edges });
      }, 0);
    });
    c.on('dragfree', 'node', (evt) => {
      clearTimeout(saveTimer.current);
      saveTimer.current = setTimeout(() => {
        savePositions(c);
      }, 400);
    });
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onTheme = () => c.style(stylesheet());
    mq.addEventListener('change', onTheme);
    // The graph stays mounted while other tabs are shown; re-measure on reveal.
    const ro = new ResizeObserver(() => {
      c.resize();
      // Structures added while another tab was showing: arrange on reveal.
      if (pendingArrange.current && host.current && host.current.offsetWidth > 0) arrangeRef.current();
    });
    ro.observe(host.current);
    return () => { mq.removeEventListener('change', onTheme); ro.disconnect(); c.destroy(); };
  }, []);

  // --- sync elements with workspace + job state
  useEffect(() => {
    const c = cy.current;
    if (!c) return;
    const { structures, edges, positions } = workspace;
    const ids = new Set();
    let placed = 0;
    let unplaced = 0;   // new nodes with no saved position -> the graph re-arranges
    const spawned = [];  // new nodes found from a node already shown: grow out of it instead
    const extent = c.extent();
    c.batch(() => {
      for (const s of Object.values(structures)) {
        if (isComplex(s)) continue;
        ids.add(s.id);
        const img = depictUrl(s.smiles, 200, 150);
        const data = { id: s.id, label: nodeLabel(s), img: img || '', noimg: !img };
        const el = c.getElementById(s.id);
        if (el.nonempty()) {
          if (el.data('label') !== data.label || el.data('img') !== data.img) el.data(data);
        } else {
          const parent = !positions[s.id] && s.origin?.parent ? c.getElementById(s.origin.parent) : null;
          if (parent && parent.nonempty()) {
            const to = spawnSpot(c, parent, spawned.map((n) => n.to));
            spawned.push({ id: s.id, to });
            c.add({ group: 'nodes', data, position: { ...parent.position() }, style: { opacity: 0 } });
            continue;
          }
          if (!positions[s.id]) unplaced += 1;
          const p = positions[s.id] || {
            x: (extent.x1 + extent.x2) / 2 + ((placed % 4) - 1.5) * 130,
            y: (extent.y1 + extent.y2) / 2 + Math.floor(placed / 4) * 120,
          };
          placed += 1;
          c.add({ group: 'nodes', data, position: { ...p } });
        }
      }
      for (const e of Object.values(edges)) {
        if (isComplex(structures[e.source]) || isComplex(structures[e.target])) continue;
        ids.add(e.id);
        const st = edgeStatus(e, state.jobs);
        const data = { id: e.id, source: e.source, target: e.target, status: st.status, label: edgeLabel(e, st) };
        const el = c.getElementById(e.id);
        if (el.nonempty() && (el.data('source') !== e.source || el.data('target') !== e.target)) el.remove();
        const cur = c.getElementById(e.id);
        if (cur.nonempty()) {
          if (cur.data('status') !== data.status || cur.data('label') !== data.label) cur.data(data);
        } else c.add({ group: 'edges', data });
      }
      for (const r of Object.values(workspace.reactions || {})) {
        const els = reactionElements(r, structures, edges, state.jobs);
        const species = [...new Set([...r.reactants, ...r.products])].map((id) => c.getElementById(id)).filter((n) => n.nonempty());
        for (const el of els) {
          ids.add(el.data.id);
          const cur = c.getElementById(el.data.id);
          if (cur.nonempty()) {
            if (cur.data('label') !== el.data.label || cur.data('status') !== el.data.status) cur.data(el.data);
            continue;
          }
          if (el.group === 'nodes') {
            // A new dot: its saved spot, else between its species (a little off
            // the straight line, so parallel reactions between two species part).
            let p = positions[r.id];
            if (!p && species.length) {
              const mx = species.reduce((a, n) => a + n.position('x'), 0) / species.length;
              const my = species.reduce((a, n) => a + n.position('y'), 0) / species.length;
              const k = Object.values(workspace.reactions).filter((o) => o.created < r.created
                && [...o.reactants, ...o.products].some((id) => r.reactants.includes(id) || r.products.includes(id))).length;
              p = { x: mx + (k % 2 ? 1 : -1) * 26 * Math.ceil(k / 2), y: my + 34 * (k % 2 ? 1 : -1) * Math.ceil(k / 2) };
            }
            c.add({ ...el, position: { ...(p || { x: 0, y: 0 }) } });
          } else c.add(el);
        }
      }
      c.elements().forEach((el) => { if (!ids.has(el.id())) el.remove(); });
    });
    if (spawned.length) {
      const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      for (const { id, to } of spawned) {
        const el = c.getElementById(id);
        el.animate({ position: to, style: { opacity: 1 } }, {
          duration: reduced ? 0 : 700, easing: 'ease-out-cubic',
          complete: () => { el.removeStyle('opacity'); savePositions(c); },
        });
      }
      // Keep what just appeared in sight: pan (never zoom in) to include it.
      const ext = c.extent();
      const out = spawned.some(({ to }) => to.x - NODE_W < ext.x1 || to.x + NODE_W > ext.x2
        || to.y - NODE_H < ext.y1 || to.y + NODE_H > ext.y2);
      if (out && !arranging.current) {
        const xs = c.nodes().map((n) => n.position('x')).concat(spawned.map((n) => n.to.x));
        const ys = c.nodes().map((n) => n.position('y')).concat(spawned.map((n) => n.to.y));
        const w = Math.max(...xs) - Math.min(...xs) + 2 * NODE_W, h = Math.max(...ys) - Math.min(...ys) + 2 * NODE_H;
        const zoom = Math.min(c.zoom(), c.width() / w, c.height() / h);
        const cx = (Math.max(...xs) + Math.min(...xs)) / 2, cyy = (Math.max(...ys) + Math.min(...ys)) / 2;
        c.animate({ zoom, pan: { x: c.width() / 2 - cx * zoom, y: c.height() / 2 - cyy * zoom } },
          { duration: reduced ? 0 : 500 });
      }
    }
    if (unplaced) arrangeRef.current();   // e.g. minima just added from a result
    else if (placed && Object.keys(positions).length === 0) fitCapped(c);
  }, [workspace, statusKey]);

  // --- "breathing" nodes: a structure with a calculation running on it
  // glows softly, the halo slowly swelling and fading (no shaking).
  useEffect(() => {
    const c = cy.current;
    if (!c) return undefined;
    const ids = activeKey ? activeKey.split(',') : [];
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const base = glowing.current;
    const settle = (id) => {
      const el = c.getElementById(id);
      if (el.nonempty()) el.removeStyle('width height underlay-color underlay-opacity underlay-padding underlay-shape');
      base.delete(id);
    };
    for (const id of [...base.keys()]) if (!ids.includes(id)) settle(id);
    if (!ids.length) return undefined;
    const accent = css('--accent');
    const phase = Object.fromEntries(ids.map((id, i) => [id, i * 1.3]));   // don't breathe in lockstep
    const PERIOD = 3.6;                                                    // seconds per slow breath
    let frame;
    const tick = (now) => {
      const t = now / 1000;
      c.batch(() => {
        for (const id of ids) {
          const el = c.getElementById(id);
          if (el.empty()) continue;
          base.add(id);
          // 0..1, eased at both ends so it lingers at rest and at full glow.
          const u = (1 - Math.cos((2 * Math.PI * t) / PERIOD + phase[id])) / 2;
          const breath = u * u * (3 - 2 * u);
          el.style({
            'underlay-color': accent,
            'underlay-shape': 'round-rectangle',
            'underlay-opacity': reduced ? 0.16 : 0.08 + 0.14 * breath,
            'underlay-padding': reduced ? 6 : 3 + 7 * breath,
            ...(reduced ? {} : { width: NODE_W * (1 + 0.015 * breath), height: NODE_H * (1 + 0.015 * breath) }),
          });
        }
      });
      if (!reduced) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [activeKey]);

  // --- reflect store selection into cytoscape
  useEffect(() => {
    const c = cy.current;
    if (!c) return;
    c.batch(() => {
      c.elements().unselect();
      for (const id of [...selection.structures, ...selection.edges]) c.getElementById(id).select();
      for (const eid of selection.edges) {
        const r = reactionOfEdge(state.workspace, eid);
        if (r) c.getElementById(r.id).select();
      }
    });
  }, [selection]);

  useEffect(() => {
    if (!connectMode && cy.current) { connectFrom.current = null; cy.current.nodes().removeClass('connect-source'); }
  }, [connectMode]);

  // --- the view: filters and folded branches
  const focusKey = view.focusHops > 0 ? selection.structures.join(',') : '';
  useEffect(() => {
    const c = cy.current;
    if (!c) return;
    const sel = new Set(selection.structures);
    const { hide, dim, folded, kids } = computeView(workspace, view, collapsed, sel,
      view.focusHops > 0 ? selection.structures : null, c);
    c.batch(() => {
      c.elements().forEach((el) => {
        const id = el.id();
        el.toggleClass('vhidden', hide.has(id));
        el.toggleClass('dim', !hide.has(id) && dim.has(id));
        if (el.isNode()) {
          const s = workspace.structures[id];
          if (!s) return;
          el.toggleClass('collapsed', !!folded[id]);
          const label = nodeLabel(s) + (folded[id] ? `  ▸ +${folded[id]}` : '');
          if (el.data('label') !== label) el.data('label', label);
        }
      });
    });
    const shownStructures = Object.values(workspace.structures).filter((s) => !isComplex(s));
    const total = shownStructures.length;
    setCounts({ shown: total - shownStructures.filter((s) => hide.has(s.id)).length, total, kids });
  }, [workspace, statusKey, view, collapsed, focusKey]);

  const layout = () => {
    const c = cy.current;
    if (!c) return;
    if (!host.current || host.current.offsetWidth === 0) {   // hidden tab: do it when shown
      pendingArrange.current = true;
      return;
    }
    pendingArrange.current = false;
    arranging.current = true;
    c.elements().filter((el) => !el.hasClass('vhidden')).layout({
      name: 'cose', animate: true, animationDuration: 400, fit: false, randomize: false,
      nodeDimensionsIncludeLabels: true, nodeRepulsion: () => 400000, nodeOverlap: 40,
      idealEdgeLength: () => 180, componentSpacing: 120, padding: 40,
    })
      .on('layoutstop', () => {
        arranging.current = false;
        if (state.layout === 'playground') fitInto(c, playgroundFitMargins()); else fitCapped(c);
        savePositions(c);
      }).run();
  };
  arrangeRef.current = layout;

  // Commands from outside the graph (the playground layout's bottom dock):
  // 'arrange', 'select-all', or {cmd: 'fit', margins: {l, t, r, b}} to fit
  // into the part of the canvas that floating controls leave free.
  useEffect(() => {
    const on = (e) => {
      const c = cy.current;
      if (!c) return;
      const d = typeof e.detail === 'string' ? { cmd: e.detail } : (e.detail || {});
      if (d.cmd === 'arrange') arrangeRef.current();
      else if (d.cmd === 'fit') (d.margins ? fitInto(c, d.margins) : fitCapped(c));
      else if (d.cmd === 'select-all') select({ structures: c.nodes().filter((n) => !n.hasClass('vhidden') && !n.data('rxn')).map((n) => n.id()) });
      else if (d.cmd === 'view') setPanelRef.current((x) => !x);
      else if (d.cmd === 'restyle') c.style(stylesheet());          // colours come from CSS variables
      else if (d.cmd === 'center' && d.id) {
        // {ifHidden}: only when it is off screen; {zoom}: close enough to read; {flash}: pulse it.
        const el = c.getElementById(d.id);
        if (el.empty()) return;
        if (el.hasClass('vhidden')) el.removeClass('vhidden');   // filtered out of the view: show it anyway
        const ext = c.extent(), p = el.position();
        const inView = p.x > ext.x1 + 40 && p.x < ext.x2 - 40 && p.y > ext.y1 + 40 && p.y < ext.y2 - 40;
        if (!d.ifHidden || !inView) {
          const anim = { center: { eles: el }, duration: 350, easing: 'ease-in-out-cubic' };
          if (d.zoom && c.zoom() < 0.9) anim.zoom = { level: 0.9, position: p };
          c.stop(); c.animate(anim);
        }
        if (d.flash) {
          clearInterval(flashTimer.current);
          let n = 0;
          el.addClass('flash');
          flashTimer.current = setInterval(() => {
            n += 1;
            el.toggleClass('flash', n % 2 === 0);
            if (n >= 5) { clearInterval(flashTimer.current); el.removeClass('flash'); }
          }, 420);
        }
      }
    };
    window.addEventListener('mepd:graph', on);
    return () => window.removeEventListener('mepd:graph', on);
  }, []);
  const empty = Object.keys(workspace.structures).length === 0;

  return html`
    <div class=${`graph-wrap ${connectMode ? 'connecting' : ''} ${drag ? 'drag' : ''}`}
      onDragOver=${(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave=${() => setDrag(false)}
      onDrop=${(e) => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files.length) uploadFiles([...e.dataTransfer.files]); }}>
      <div class="graph" ref=${host}></div>
      <div class="graph-toolbar">
        <button class=${`btn small ${connectMode ? 'primary' : 'ghost'}`} onClick=${() => set({ connectMode: !state.connectMode })}
          title="Click a start structure, then an end structure, to draw an edge (C)">
          ${connectMode ? 'Connecting… Esc to stop' : '＋ Connect'}</button>
        <button class="btn small ghost" onClick=${layout} title="Auto-arrange the graph">Arrange</button>
        <button class="btn small ghost" onClick=${() => fitCapped(cy.current)} title="Fit everything in view">Fit</button>
        <button class="btn small ghost" onClick=${() => select({ structures: cy.current.nodes().filter((n) => !n.hasClass('vhidden') && !n.data('rxn')).map((n) => n.id()) })}
          title="Select every structure shown (then delete, download, or run one calculation on all)">Select all</button>
        <button class=${`btn small ${panel ? 'primary' : 'ghost'}`} onClick=${() => setPanel(!panel)}
          title="Filter what the graph shows, and fold branches away">View${counts.shown < counts.total ? ` · ${counts.shown}/${counts.total}` : ''}</button>
        ${howToHidden && html`<button class="btn small ghost" title="Show 'How it works'" aria-label="How it works"
          onClick=${() => { prefs.set('hideHowTo', false); set({ howToHidden: false }); }}>?</button>`}
        ${nSelected > 0 && html`<button class="btn small ghost danger-text" onClick=${deleteSelection}
          title="Delete the selected structures and edges (Delete key)">Delete ${nSelected}</button>`}
      </div>
      ${!empty && html`<div class="legend" title="Edge colours. Labels are the lowest barrier ΔE‡ in kcal/mol; ≈x? means the path maximum, not yet confirmed by TS + IRC.">
        <span><i class="lg idle"></i>not computed</span>
        <span><i class="lg running"></i>running</span>
        <span><i class="lg done"></i>ΔE‡, kcal/mol</span>
        <span><i class="lg failed"></i>failed</span>
      </div>`}
      ${panel && html`<${ViewPanel} view=${view} setView=${setView} collapsed=${collapsed} setCollapsed=${setCollapsed}
        kids=${counts.kids} shown=${counts.shown} total=${counts.total} onClose=${() => setPanel(false)} focusCount=${selection.structures.length} />`}
      ${connectMode && html`<div class="graph-hint">Click the <b>start</b> structure, then the <b>end</b> structure.</div>`}
      ${empty && html`<div class="graph-empty">
        <h3>Your reaction graph</h3>
        <p>Every molecule you make or add becomes a node here. Build one in Design (atom by atom, or from a SMILES,
          a reaction or an XYZ), then explore its reactions, or connect two molecules to find the step between them.</p>
        <button class="btn primary" onClick=${() => openTab('design')}>Design a molecule</button>
        <p class="small muted">or <a href="#" onClick=${(e) => { e.preventDefault(); set({ modal: { kind: 'quick' } }); }}>connect two structures you already have</a></p>
      </div>`}
    </div>`;
}
