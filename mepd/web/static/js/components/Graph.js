// The reaction graph: every library structure is a node, every declared
// pair an edge. Edges are coloured by what's been computed on them.
import { html, useEffect, useRef, useState } from '../lib.js';
import cytoscape from '../../vendor/cytoscape.esm.min.js';
import { api, attempt, deleteSelection } from '../api.js';
import { clearSelection, prefs, select, set, state, useStore } from '../store.js';
import { depictUrl, edgeStatus, edgeStatusKey } from '../util.js';
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
  ];
}

// Fit everything in view, but never blow a two-node graph up to poster size.
function fitCapped(c, padding = 70) {
  if (!c.nodes().length) return;
  c.fit(undefined, padding);
  if (c.zoom() > 1.1) { c.zoom(1.1); c.center(); }
  c.panBy({ x: 0, y: 22 });   // clear of the floating toolbar
}

// Single-structure calculations whose running target "breathes" in the graph.
const ANIMATED_OPS = new Set(['hessian-sample', 'hessian-global', 'tsopt', 'optimize', 'vri']);
const NODE_W = 96, NODE_H = 72;

function activeStructureKey(jobs) {
  const ids = new Set();
  for (const j of Object.values(jobs)) {
    if (j.status === 'running' && ANIMATED_OPS.has(j.op)) j.targets.structures.forEach((id) => ids.add(id));
  }
  return [...ids].sort().join(',');
}

function edgeLabel(e, st) {
  const parts = [];
  if (st.barrier != null) parts.push(`${st.barrier.toFixed(1)}`);
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
      if (evt.target.isNode()) select({ structures: [id] }, additive);
      else select({ edges: [id] }, additive);
    });
    c.on('boxend', () => {
      // Box selection: take whatever cytoscape selected, in any order.
      setTimeout(() => {
        const nodes = c.nodes(':selected').map((n) => n.id());
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
    const extent = c.extent();
    c.batch(() => {
      for (const s of Object.values(structures)) {
        ids.add(s.id);
        const img = depictUrl(s.smiles, 200, 150);
        const data = { id: s.id, label: s.name, img: img || '', noimg: !img };
        const el = c.getElementById(s.id);
        if (el.nonempty()) {
          if (el.data('label') !== data.label || el.data('img') !== data.img) el.data(data);
        } else {
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
      c.elements().forEach((el) => { if (!ids.has(el.id())) el.remove(); });
    });
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
    });
  }, [selection]);

  useEffect(() => {
    if (!connectMode && cy.current) { connectFrom.current = null; cy.current.nodes().removeClass('connect-source'); }
  }, [connectMode]);

  const layout = () => {
    const c = cy.current;
    if (!c) return;
    if (!host.current || host.current.offsetWidth === 0) {   // hidden tab: do it when shown
      pendingArrange.current = true;
      return;
    }
    pendingArrange.current = false;
    arranging.current = true;
    c.layout({
      name: 'cose', animate: true, animationDuration: 400, fit: false, randomize: false,
      nodeDimensionsIncludeLabels: true, nodeRepulsion: () => 400000, nodeOverlap: 40,
      idealEdgeLength: () => 180, componentSpacing: 120, padding: 40,
    })
      .on('layoutstop', () => {
        arranging.current = false;
        fitCapped(c);
        savePositions(c);
      }).run();
  };
  arrangeRef.current = layout;
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
        <button class="btn small ghost" onClick=${() => select({ structures: Object.keys(state.workspace.structures) })}
          title="Select every structure (then delete, download, or run one calculation on all)">Select all</button>
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
      ${connectMode && html`<div class="graph-hint">Click the <b>start</b> structure, then the <b>end</b> structure.</div>`}
      ${empty && html`<div class="graph-empty">
        <h3>Your reaction graph</h3>
        <p>Structures you add appear here as nodes. Select two and run a calculation to connect them, or start with a transition-state search right away.</p>
        <button class="btn primary" onClick=${() => set({ modal: { kind: 'quick' } })}>Find a transition state</button>
      </div>`}
    </div>`;
}
