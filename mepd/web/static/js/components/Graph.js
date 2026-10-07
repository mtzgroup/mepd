// The reaction graph: every library structure is a node, every declared
// pair an edge. Edges are coloured by what's been computed on them.
import { html, useEffect, useRef, useState } from '../lib.js';
import cytoscape from '../../vendor/cytoscape.esm.min.js';
import { api, attempt, deleteSelection } from '../api.js';
import { clearSelection, openJob, openTab, prefs, select, set, state, useStore } from '../store.js';
import { complexNodes, complexOfStructure, depictUrl, edgeStatus, edgeStatusKey, isComplex, nodeOf, playgroundFitMargins, reactionOfEdge } from '../util.js';
import { uploadFiles } from './Library.js';
import { SetupPanel, activeSetup, edgeUnderSetup, setupEdgeLabel, setupKey } from './Setups.js';

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
    // Only a multi-step route joins the two ends (no single step does): not drawn as a step.
    { selector: 'edge[status = "done"][route = "yes"]', style: { 'line-style': 'dashed', 'line-dash-pattern': [2, 5] } },
    { selector: 'edge[status = "running"]', style: { 'line-style': 'solid', 'line-color': css('--accent'), 'target-arrow-color': css('--accent'), width: 3.5 } },
    { selector: 'edge[status = "queued"]', style: { 'line-color': css('--warn'), 'target-arrow-color': css('--warn') } },
    { selector: 'edge[status = "failed"]', style: { 'line-color': css('--danger'), 'target-arrow-color': css('--danger') } },
    // Under an experimental setup (Setups.js): does the step run in the reaction time?
    { selector: 'edge[runs = "yes"]', style: { 'line-style': 'solid', 'line-color': css('--ok'), 'target-arrow-color': css('--ok'), width: 3.5 } },
    { selector: 'edge[runs = "slow"]', style: { 'line-style': 'solid', 'line-color': css('--warn'), 'target-arrow-color': css('--warn') } },
    { selector: 'edge[runs = "no"]', style: { 'line-style': 'solid', 'line-color': css('--edge-idle'), 'target-arrow-color': css('--edge-idle'), opacity: 0.7 } },
    { selector: 'edge:selected', style: { width: 5, 'underlay-color': css('--accent'), 'underlay-opacity': 0.25, 'underlay-padding': 5 } },
    // Reactions: a complex (several molecules together) is a small circle joined
    // to its molecules; a reaction is an edge between two complexes (a one-molecule
    // complex is that molecule's node), carrying the TS search like any edge.
    { selector: 'node.cx', style: {
      shape: 'ellipse', width: 30, height: 30, 'background-image': 'none', 'background-color': css('--panel'),
      'border-width': 2.5, 'border-color': css('--accent'), label: 'data(label)', 'text-valign': 'center',
      'text-margin-y': 0, 'font-size': 12, 'font-weight': 600, color: css('--accent'), 'text-background-opacity': 0,
    } },
    { selector: 'node.cx:selected', style: { 'border-width': 3, 'border-color': css('--accent') } },
    { selector: 'edge.member', style: { width: 1.2, 'line-style': 'solid', 'line-color': css('--border-strong'),
      'target-arrow-shape': 'none', 'curve-style': 'straight', label: 'data(label)', opacity: 0.8 } },
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

// The view fitted to the graph, as the layout in use wants it.
function fitView(c) {
  if (state.layout === 'playground') fitInto(c, playgroundFitMargins()); else fitCapped(c);
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

// Complex nodes (util.complexNodes), each joined to its molecules by member
// lines (x2 for two of one). Edges need no special case: every workspace
// edge is drawn between the nodes its ends are drawn as (util.nodeOf), so a
// reaction's edge joins its two complexes, and a calculation run on a
// complex adds its results to the complex node like to any other.
function complexGraph(workspace) {
  const complexes = Object.values(complexNodes(workspace));
  const members = [];
  for (const cx of complexes) {
    const n = cx.members.reduce((m, id) => ({ ...m, [id]: (m[id] || 0) + 1 }), {});
    for (const [sid, k] of Object.entries(n)) {
      members.push({ group: 'edges', classes: 'member', data: { id: `${cx.id}|${sid}`, source: cx.id, target: sid,
        label: k > 1 ? `×${k}` : '', member: 1 } });
    }
  }
  return { complexes, members };
}

// The molecules an edge end stands for: a complex's members, else itself.
const sideOf = (ws, sid) => (isComplex(ws.structures[sid]) ? ws.structures[sid].members || [] : [sid]);

// Tree: each subnetwork (molecules joined by reactions; shuttles and tiny
// molecules such as water do not join them) laid out on its own, the
// subnetworks side by side along x, the largest first. Within one, a
// synthesis reads left to right: every molecule sits in the column after
// the step that first can make it (longest path from what it starts from),
// every step between its inputs and outputs with its two complexes, and a
// building block just left of the step that uses it -- for a
// retrosynthesis, the target on the right and its routes fanning out to the
// left. Rows follow a few ordering sweeps (each node towards the mean row of
// its neighbours) to cut crossings. Tiny molecules (H2, water, OH) hang just
// under the first step that uses or releases them. Molecules in no reaction
// share a small grid at the end. Returns {id: {x, y}}, or null when nothing
// reacts into anything.
const heavyAtoms = (s) => (s?.formula || '').replace(/H\d*/g, '').match(/[A-Z][a-z]?\d*/g)?.reduce(
  (n, t) => n + (parseInt(t.replace(/[A-Za-z]/g, ''), 10) || 1), 0) || 0;

export function treePositions(ws, shown) {
  const COL = 235, DY = 150, GAP = 320, CXO = 62;
  const tiny = (id) => heavyAtoms(ws.structures[id]) <= 1;
  const steps = [];
  const inReaction = new Set();
  for (const r of Object.values(ws.reactions || {})) {
    const e = r.edge && ws.edges[r.edge];
    const ends = e ? [nodeOf(ws, e.source), nodeOf(ws, e.target)] : [null, null];
    const allR = [...new Set(r.reactants)].filter((id) => shown.has(id));
    const allP = [...new Set(r.products)].filter((id) => shown.has(id));
    const shuttles = allR.filter((id) => allP.includes(id));
    const rs = allR.filter((id) => !shuttles.includes(id));
    const ps = allP.filter((id) => !shuttles.includes(id));
    if (!rs.length || !ps.length) continue;
    // Tiny molecules are drawn beside the step, unless the step is about them alone.
    const ins = rs.some((id) => !tiny(id)) ? rs.filter((id) => !tiny(id)) : rs;
    const outs = ps.some((id) => !tiny(id)) ? ps.filter((id) => !tiny(id)) : ps;
    steps.push({ id: `s${steps.length}`, ins, outs, small: [...rs, ...ps, ...shuttles].filter((id) => !ins.includes(id) && !outs.includes(id)),
      cx: ends.map((x) => (x && x.startsWith('cx:') ? x : null)), rank: Math.min(...(r.retro?.routes || [99])) });
    if (e) inReaction.add(e.id);
  }
  // A plain edge between two molecules (an isomerization): source -> target.
  for (const e of Object.values(ws.edges)) {
    if (inReaction.has(e.id)) continue;
    const a = nodeOf(ws, e.source), b = nodeOf(ws, e.target);
    if (a && b && a !== b && shown.has(a) && shown.has(b) && !a.startsWith('cx:') && !b.startsWith('cx:')) {
      steps.push({ id: `s${steps.length}`, ins: [a], outs: [b], small: [], cx: [null, null], rank: 99 });
    }
  }
  if (!steps.length) return null;
  steps.sort((x, y) => x.rank - y.rank);

  // Subnetworks.
  const parent = {};
  const find = (x) => { while (parent[x] !== x) x = parent[x]; return x; };
  for (const st of steps) {
    const ids = [st.id, ...st.ins, ...st.outs];
    for (const id of ids) if (parent[id] === undefined) parent[id] = id;
    for (const id of ids.slice(1)) { const a = find(ids[0]), b = find(id); if (a !== b) parent[a] = b; }
  }
  const groups = {};
  for (const st of steps) (groups[find(st.id)] ||= []).push(st);

  const pos = {};
  const blocks = [];
  for (const group of Object.values(groups)) {
    const mols = [...new Set(group.flatMap((st) => [...st.ins, ...st.outs]))];
    const made = new Set(group.flatMap((st) => st.outs));
    // Columns: molecules even, steps odd, by the longest path from the
    // starting molecules, with the links that close a cycle set aside
    // (found by a depth-first search from them).
    const out = {};
    for (const st of group) {
      for (const m of st.ins) (out[m] ||= []).push(st.id);
      out[st.id] = [...st.outs];
    }
    const nodes = [...mols, ...group.map((st) => st.id)];
    const back = new Set();
    const state = {};
    const starts = [...mols.filter((m) => !made.has(m)), ...nodes];
    for (const s0 of starts) {
      if (state[s0]) continue;
      const stack = [[s0, 0]];
      state[s0] = 1;
      while (stack.length) {
        const top = stack[stack.length - 1];
        const next = (out[top[0]] || [])[top[1]++];
        if (next === undefined) { state[top[0]] = 2; stack.pop(); continue; }
        if (state[next] === 1) back.add(`${top[0]}>${next}`);
        else if (!state[next]) { state[next] = 1; stack.push([next, 0]); }
      }
    }
    const indeg = Object.fromEntries(nodes.map((n) => [n, 0]));
    for (const a of nodes) for (const b of out[a] || []) if (!back.has(`${a}>${b}`)) indeg[b] += 1;
    const isStep = new Set(group.map((st) => st.id));
    const col = {};
    const queue = nodes.filter((n) => indeg[n] === 0);
    for (const n of queue) col[n] = 0;
    while (queue.length) {
      const a = queue.shift();
      if (isStep.has(a) ? col[a] % 2 === 0 : col[a] % 2 === 1) col[a] += 1;   // steps odd, molecules even
      for (const b of out[a] || []) {
        if (back.has(`${a}>${b}`)) continue;
        col[b] = Math.max(col[b] ?? 0, col[a] + 1);
        if (--indeg[b] === 0) queue.push(b);
      }
    }
    for (const n of nodes) if (!(n in col)) col[n] = isStep.has(n) ? 1 : 0;
    // A molecule nothing here makes sits just before its first use.
    for (const m of mols) {
      if (made.has(m)) continue;
      const uses = group.filter((st) => st.ins.includes(m)).map((st) => col[st.id]);
      if (uses.length) col[m] = Math.min(...uses) - 1;
    }
    // Rows: barycentre sweeps over the columns.
    const nbr = {};
    const link = (a, b) => { (nbr[a] ||= []).push(b); (nbr[b] ||= []).push(a); };
    for (const st of group) { st.ins.forEach((m) => link(m, st.id)); st.outs.forEach((m) => link(m, st.id)); }
    const cols = {};
    const order = [...group.map((st) => st.id), ...mols];
    for (const id of order) (cols[col[id]] ||= []).push(id);
    const keys = Object.keys(cols).map(Number).sort((a, b) => a - b);
    const row = {};
    const setRows = () => keys.forEach((k) => cols[k].forEach((id, i) => { row[id] = i - (cols[k].length - 1) / 2; }));
    setRows();
    for (let sweep = 0; sweep < 8; sweep++) {
      const ks = sweep % 2 ? [...keys].reverse() : keys;
      for (const k of ks) {
        const side = sweep % 2 ? 1 : -1;
        const bc = (id) => {
          const ns = (nbr[id] || []).filter((n) => col[n] === k + side);
          return ns.length ? ns.reduce((s, n) => s + row[n], 0) / ns.length : row[id];
        };
        cols[k].sort((a, b) => bc(a) - bc(b));
        cols[k].forEach((id, i) => { row[id] = i - (cols[k].length - 1) / 2; });
      }
    }
    const local = {};
    for (const id of order) local[id] = { x: col[id] * COL, y: row[id] * DY };
    const block = {};
    for (const m of mols) block[m] = local[m];
    for (const st of group) {
      const { x, y } = local[st.id];
      const [rcx, pcx] = st.cx;
      if (rcx && !block[rcx]) block[rcx] = { x: x - CXO, y };
      if (pcx && !block[pcx]) block[pcx] = { x: x + CXO, y };
    }
    // Tiny molecules: just under the first step that has them, never on another node.
    const taken = Object.values(block);
    for (const st of group) {
      st.small.forEach((m, k) => {
        if (block[m] || pos[m]) return;
        const { x, y } = local[st.id];
        let q = { x: x + (k - (st.small.length - 1) / 2) * 105, y: y + 82 };
        while (taken.some((o) => Math.abs(o.x - q.x) < 95 && Math.abs(o.y - q.y) < 70)) q = { x: q.x, y: q.y + 72 };
        block[m] = q;
        taken.push(q);
      });
    }
    blocks.push(block);
    for (const [id, q] of Object.entries(block)) if (!pos[id]) pos[id] = q;
  }
  // Side by side along x, the largest subnetwork first, tops aligned; then
  // the molecules in no reaction, in a small grid of their own.
  blocks.sort((a, b) => Object.keys(b).length - Object.keys(a).length);
  const lone = [...shown].filter((id) => !pos[id] && !id.startsWith('cx:'));
  if (lone.length) {
    const cols = Math.max(2, Math.ceil(Math.sqrt(lone.length)));
    const grid = {};
    lone.forEach((id, k) => { grid[id] = { x: (k % cols) * 160, y: Math.floor(k / cols) * DY }; });
    blocks.push(grid);
    Object.assign(pos, grid);
  }
  let cursor = 0;
  for (const block of blocks) {
    const qs = Object.entries(block).filter(([id, q]) => pos[id] === q).map(([, q]) => q);
    if (!qs.length) continue;
    const x0 = Math.min(...qs.map((q) => q.x)), x1 = Math.max(...qs.map((q) => q.x));
    const y0 = Math.min(...qs.map((q) => q.y));
    for (const q of qs) { q.x += cursor - x0; q.y -= y0; }
    cursor += x1 - x0 + GAP;
  }
  return pos;
}

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
  for (const cx of Object.values(complexNodes(workspace))) {
    if (cx.members.some((id) => hide.has(id))) hide.add(cx.id);
    else if (cx.members.every((id) => dim.has(id))) dim.add(cx.id);
  }
  for (const e of Object.values(edges)) {
    const st = edgeStatus(e, state.jobs).status;
    if (view.hideProposed && e.origin?.proposed && !['done', 'running', 'queued'].includes(st)) hide.add(e.id);
    if (view.hideFailed && st === 'failed') hide.add(e.id);
    const [a, b] = [nodeOf(workspace, e.source), nodeOf(workspace, e.target)];
    if (dim.has(a) && dim.has(b)) dim.add(e.id);
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

function edgeLabel(e, st, reaction = null) {
  const parts = [];
  if (st.routeSteps > 1) parts.push(`${st.routeSteps} steps`);
  if (st.barrier != null) parts.push(`${st.warning ? '⚠ ' : ''}${st.barrier.toFixed(1)}`);
  else if (st.barrierUnverified != null) parts.push(`≈${st.barrierUnverified.toFixed(1)}?`);
  else if (st.status === 'running') parts.push('running…');
  else if (st.status === 'queued') parts.push('queued');
  else if (st.status === 'failed') parts.push('failed');
  else if (reaction?.delta_e_kcal != null) parts.push(`ΔE ${reaction.delta_e_kcal >= 0 ? '+' : ''}${reaction.delta_e_kcal.toFixed(1)}`);
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
  const fittedFor = useRef(null);       // the workspace whose saved layout the view was fitted to
  const arrangeRef = useRef(() => {});
  const treeRef = useRef(() => {});
  const selection = useStore((s) => s.selection);
  const connectMode = useStore((s) => s.connectMode);
  const howToHidden = useStore((s) => s.howToHidden);
  const [drag, setDrag] = useState(false);
  const [view, setViewState] = useState(() => ({ ...VIEW_DEFAULT, ...prefs.get('graphView', {}) }));
  const [collapsed, setCollapsedState] = useState(() => prefs.get('graphCollapsed', []));
  const [panel, setPanel] = useState(false);
  const [setupPanel, setSetupPanel] = useState(false);
  const condKey = useStore(setupKey);
  const setup = activeSetup(workspace);
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
      container: host.current, style: stylesheet(), minZoom: 0.05, maxZoom: 3,
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
          // A complex node is a composition (its id is not a structure): connect its
          // lowest-energy geometry, as selecting it does.
          const end = (nid) => (nid.startsWith('cx:') ? complexNodes(state.workspace)[nid]?.geometries[0] ?? nid : nid);
          attempt(() => api.post('/api/edges', { source: end(src), target: end(id) })).then((edge) => {
            if (edge) select({ edges: [edge.id] });
          });
        }
        return;
      }
      if (evt.target.data('cx')) {          // a complex: its geometry (else, if it has none, its molecules)
        const g = complexNodes(state.workspace)[id]?.geometries[0];
        select({ structures: g ? [g] : [...new Set(evt.target.data('members'))] }, additive);
        return;
      }
      if (evt.target.data('member')) {      // a member line: that molecule
        select({ structures: [evt.target.data('target')] }, additive);
        return;
      }
      if (evt.target.isNode()) select({ structures: [id] }, additive);
      else select({ edges: [id] }, additive);
    });
    c.on('dbltap', 'edge', (evt) => {
      // A reaction (or any) edge: open the calculation behind its barrier.
      const edge = state.workspace.edges[evt.target.id()];
      const job = edge && edgeStatus(edge, state.jobs).barrierJob;
      if (job) openJob(job);
    });
    c.on('dbltap', 'node', (evt) => {
      const id = evt.target.id();
      if (evt.target.data('cx')) return;
      const kids = childrenOf(state.workspace.structures);
      const cur = collapsedRef.current;
      if (cur.includes(id)) setCollapsedRef.current(cur.filter((x) => x !== id));
      else if (kids[id]?.length) setCollapsedRef.current([...cur, id]);
    });
    c.on('boxend', () => {
      // Box selection: take whatever cytoscape selected, in any order.
      setTimeout(() => {
        const nodes = c.nodes(':selected').filter((n) => !n.data('cx')).map((n) => n.id());
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
    let seen = '';
    const ro = new ResizeObserver(([entry]) => {
      // Whole pixels only: a sub-pixel change is rounding, not a resize, and
      // resizing on it can feed itself (fractional display scales).
      const { width, height } = entry.contentRect;
      const size = `${Math.round(width)}x${Math.round(height)}`;
      if (size === seen) return;
      seen = size;
      c.resize();
      // Structures added while another tab was showing: arrange on reveal.
      if (pendingArrange.current && host.current && host.current.offsetWidth > 0) {
        if (pendingArrange.current === 'fit') { pendingArrange.current = false; fitView(c); }
        else (pendingArrange.current === 'tree' ? treeRef : arrangeRef).current();
      }
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
    let unplacedRetro = 0;   // ... as a tree, when they are a retrosynthesis's routes
    const spawned = [];  // new nodes found from a node already shown: grow out of it instead
    const extent = c.extent();
    c.batch(() => {
      for (const s of Object.values(structures)) {
        if (isComplex(s)) continue;   // drawn as its complex node
        ids.add(s.id);
        const img = depictUrl(s.smiles, 200, 150);
        const data = { id: s.id, label: nodeLabel(s), img: img || '', noimg: !img };
        const el = c.getElementById(s.id);
        if (el.nonempty()) {
          if (el.data('label') !== data.label || el.data('img') !== data.img) el.data(data);
        } else {
          const parent = !positions[s.id] && s.origin?.parent ? c.getElementById(nodeOf(workspace, s.origin.parent) || '') : null;
          if (parent && parent.nonempty()) {
            const to = spawnSpot(c, parent, spawned.map((n) => n.to));
            spawned.push({ id: s.id, to });
            c.add({ group: 'nodes', data, position: { ...parent.position() }, style: { opacity: 0 } });
            continue;
          }
          if (!positions[s.id]) { unplaced += 1; if (s.origin?.retro) unplacedRetro += 1; }
          const p = positions[s.id] || {
            x: (extent.x1 + extent.x2) / 2 + ((placed % 4) - 1.5) * 130,
            y: (extent.y1 + extent.y2) / 2 + Math.floor(placed / 4) * 120,
          };
          placed += 1;
          c.add({ group: 'nodes', data, position: { ...p } });
        }
      }
      const rg = complexGraph(workspace);
      for (const cx of rg.complexes) {
        ids.add(cx.id);
        const label = cx.members.map((m) => structures[m]?.name || '?').join(' + ');
        if (c.getElementById(cx.id).nonempty()) continue;
        // A new complex: its saved spot, else among its molecules (pulled a little
        // toward the side, so complexes of the same molecules do not stack).
        let p = positions[cx.id];
        if (!p) {
          const ms = [...new Set(cx.members)].map((m) => c.getElementById(m)).filter((n) => n.nonempty());
          const mx = ms.reduce((acc, n) => acc + n.position('x'), 0) / Math.max(1, ms.length);
          const my = ms.reduce((acc, n) => acc + n.position('y'), 0) / Math.max(1, ms.length);
          const h = [...cx.id].reduce((acc, ch) => (acc * 31 + ch.charCodeAt(0)) % 997, 7);
          p = { x: mx + ((h % 7) - 3) * 18, y: my + 50 + ((h % 5) - 2) * 14 };
        }
        c.add({ group: 'nodes', classes: 'cx', data: { id: cx.id, cx: 1, members: cx.members, label: String(cx.members.length),
          title: label }, position: p });
      }
      for (const el of rg.members) {
        // A member that is not drawn (e.g. a complex listed inside another
        // complex) must not take the whole graph down: skip its line.
        if (c.getElementById(el.data.target).empty() || c.getElementById(el.data.source).empty()) {
          console.warn(`complex ${el.data.source}: member ${el.data.target} is not a drawn molecule; line skipped`);
          continue;
        }
        ids.add(el.data.id);
        const cur = c.getElementById(el.data.id);
        if (cur.nonempty() && (cur.data('source') !== el.data.source || cur.data('target') !== el.data.target)) cur.remove();
        const now = c.getElementById(el.data.id);
        if (now.nonempty()) {
          if (now.data('label') !== el.data.label || now.data('status') !== el.data.status) now.data(el.data);
        } else c.add(el);
      }
      const cond = activeSetup(state.workspace);
      for (const e of Object.values(edges)) {
        const source = nodeOf(workspace, e.source), target = nodeOf(workspace, e.target);
        if (!source || !target || source === target) continue;
        ids.add(e.id);
        const st = edgeStatus(e, state.jobs);
        const data = { id: e.id, source, target, status: st.status, label: edgeLabel(e, st, reactionOfEdge(workspace, e.id)), runs: '',
          route: st.routeSteps ? 'yes' : '' };
        if (cond && !['running', 'queued'].includes(st.status)) {
          const info = edgeUnderSetup(e, state.jobs, cond);
          data.label = [e.label && !/^Channel|IRC$/.test(e.label) ? e.label : '', setupEdgeLabel(e, info, cond)].filter(Boolean).join(' · ');
          data.runs = info.runs === 'none' ? '' : info.runs;
        }
        const el = c.getElementById(e.id);
        if (el.nonempty() && (el.data('source') !== source || el.data('target') !== target)) el.remove();
        const cur = c.getElementById(e.id);
        if (cur.nonempty()) {
          if (cur.data('status') !== data.status || cur.data('label') !== data.label || cur.data('runs') !== data.runs
            || cur.data('route') !== data.route) cur.data(data);
        } else c.add({ group: 'edges', data });
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
    if (unplacedRetro) treeRef.current();
    else if (unplaced) arrangeRef.current();   // e.g. minima just added from a result
    else if (placed && Object.keys(positions).length === 0) fitCapped(c);
    else if (placed && fittedFor.current !== workspace.root) {
      // Saved positions drawn for the first time (a reload, another
      // session): the view is not saved, and the default one showed them
      // under the toolbar or off-screen. Fit once; when shown, if hidden.
      if (host.current && host.current.offsetWidth > 0) fitView(c);
      else pendingArrange.current = pendingArrange.current || 'fit';
    }
    if (placed) fittedFor.current = workspace.root;
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
      for (const sid of selection.structures) {     // a complex's geometry: its circle
        if (!isComplex(state.workspace.structures[sid])) continue;
        const cx = complexOfStructure(state.workspace, sid);
        if (cx) c.getElementById(cx.key).select();
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
      // A complex sits next to its molecules; reactions get room.
      idealEdgeLength: (e) => (e.data('member') ? 60 : 200), edgeElasticity: (e) => (e.data('member') ? 200 : 100),
      componentSpacing: 120, padding: 40,
    })
      .on('layoutstop', () => {
        arranging.current = false;
        if (state.layout === 'playground') fitInto(c, playgroundFitMargins()); else fitCapped(c);
        savePositions(c);
      }).run();
  };
  arrangeRef.current = layout;

  const treeLayout = () => {
    const c = cy.current;
    if (!c) return;
    if (!host.current || host.current.offsetWidth === 0) {   // hidden tab: do it when shown
      pendingArrange.current = 'tree';
      return;
    }
    pendingArrange.current = false;
    const shown = new Set(c.nodes().filter((n) => !n.hasClass('vhidden')).map((n) => n.id()));
    const pos = treePositions(state.workspace, shown);
    if (!pos) { layout(); return; }
    // Complexes no reaction placed: among their molecules.
    c.nodes('.cx').forEach((n) => {
      if (pos[n.id()] || !shown.has(n.id())) return;
      const ms = [...new Set(n.data('members'))].filter((m) => pos[m]);
      if (ms.length) pos[n.id()] = { x: ms.reduce((a, m) => a + pos[m].x, 0) / ms.length, y: Math.max(...ms.map((m) => pos[m].y)) + 70 };
    });
    arranging.current = true;
    c.nodes().filter((n) => pos[n.id()]).layout({
      name: 'preset', positions: (n) => pos[n.id()], animate: true, animationDuration: 450, fit: false,
    }).on('layoutstop', () => {
      arranging.current = false;
      if (state.layout === 'playground') fitInto(c, playgroundFitMargins()); else fitCapped(c);
      savePositions(c);
    }).run();
  };
  treeRef.current = treeLayout;

  // Commands from outside the graph (the playground layout's bottom dock):
  // 'arrange', 'select-all', or {cmd: 'fit', margins: {l, t, r, b}} to fit
  // into the part of the canvas that floating controls leave free.
  useEffect(() => {
    const on = (e) => {
      const c = cy.current;
      if (!c) return;
      const d = typeof e.detail === 'string' ? { cmd: e.detail } : (e.detail || {});
      if (d.cmd === 'arrange') arrangeRef.current();
      else if (d.cmd === 'tree') treeRef.current();
      else if (d.cmd === 'fit') (d.margins ? fitInto(c, d.margins) : fitCapped(c));
      else if (d.cmd === 'select-all') select({ structures: c.nodes().filter((n) => !n.hasClass('vhidden') && !n.data('cx')).map((n) => n.id()) });
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
        <button class="btn small ghost" onClick=${treeLayout}
          title="Lay reactions out as a synthesis tree: each target on the right, the steps that make it to its left, building blocks at the far left">Tree</button>
        <button class="btn small ghost" onClick=${() => fitCapped(cy.current)} title="Fit everything in view">Fit</button>
        <button class="btn small ghost" onClick=${() => select({ structures: cy.current.nodes().filter((n) => !n.hasClass('vhidden') && !n.data('cx')).map((n) => n.id()) })}
          title="Select every structure shown (then delete, download, or run one calculation on all)">Select all</button>
        <button class=${`btn small ${panel ? 'primary' : 'ghost'}`} onClick=${() => { setSetupPanel(false); setPanel(!panel); }}
          title="Filter what the graph shows, and fold branches away">View${counts.shown < counts.total ? ` · ${counts.shown}/${counts.total}` : ''}</button>
        <button class=${`btn small ${setupPanel || setup ? 'primary' : 'ghost'}`} onClick=${() => { setPanel(false); setSetupPanel(!setupPanel); }}
          title="Experimental conditions: solvent, temperature, reaction time. See which steps run, and predict the outcome.">
          ${setup ? `Conditions · ${setup.name}` : 'Conditions'}</button>
        ${howToHidden && html`<button class="btn small ghost" title="Show 'How it works'" aria-label="How it works"
          onClick=${() => { prefs.set('hideHowTo', false); set({ howToHidden: false }); }}>?</button>`}
        ${nSelected > 0 && html`<button class="btn small ghost danger-text" onClick=${deleteSelection}
          title="Delete the selected structures and edges (Delete key)">Delete ${nSelected}</button>`}
      </div>
      ${!empty && setup && html`<div class="legend" title=${`Under ${setup.name}: barrier ΔE‡ (kcal/mol) and half-life; (gas) = no barrier in this solvent yet`}>
        <span><i class="lg done"></i>runs in ${setup.time_s >= 86400 ? `${+(setup.time_s / 86400).toFixed(1)} d` : setup.time_s >= 3600 ? `${+(setup.time_s / 3600).toFixed(1)} h` : `${Math.round(setup.time_s / 60)} min`}</span>
        <span><i class="lg queued"></i>within 100×</span>
        <span><i class="lg idle"></i>too slow / not computed</span>
      </div>`}
      ${!empty && !setup && html`<div class="legend" title="Edge colours. Labels are the lowest barrier ΔE‡ in kcal/mol, IRC-verified first; ≈x? means the path maximum, not yet confirmed by TS + IRC. Dotted: no single step joins the two, only a route through intermediates (its highest step).">
        <span><i class="lg idle"></i>not computed</span>
        <span><i class="lg running"></i>running</span>
        <span><i class="lg done"></i>ΔE‡, kcal/mol</span>
        <span><i class="lg route"></i>multi-step route</span>
        <span><i class="lg failed"></i>failed</span>
      </div>`}
      ${setupPanel && html`<${SetupPanel} onClose=${() => setSetupPanel(false)} />`}
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
