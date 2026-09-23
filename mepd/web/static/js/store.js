// App state: one plain object, mutated only through `set`/`update`, with a
// hook that re-renders a component when the slice it selected changes.
// The server is the source of truth for workspace/jobs; this store mirrors
// /api/state and then applies SSE events.
import { useEffect, useRef, useState } from './lib.js';

const listeners = new Set();

export const state = {
  connected: false,
  loaded: false,
  workspace: { structures: {}, edges: {}, positions: {}, root: '' },
  jobs: {},
  operations: [],
  profiles: [],
  levelProfile: null,    // the workspace's level of theory (a profile name)
  validateMinima: true,  // Hessian-check structures optimized on entry
  levels: {},            // profile name ('' = built-in defaults) -> {profile, key, label}
  pathSummaries: {},     // profile name -> {text, warnings}: what its path search will really do
  profileEditor: null,   // {name, dirty, save()} while the Profiles tab holds edits
  demo: null,            // public demo policy (limits, allowed ops) or null
  cpus: 1,
  maxConcurrent: 1,
  progress: {},          // job id -> latest progress payload (merged)
  selection: { structures: [], edges: [] },  // click order preserved
  view: { tab: 'graph', jobId: null },
  connectMode: false,
  toasts: [],
  modal: null,           // {kind, ...}
};

export function update(fn) {
  fn(state);
  for (const l of [...listeners]) l();
}

export function set(patch) {
  update((s) => Object.assign(s, patch));
}

function shallowEqual(a, b) {
  if (Object.is(a, b)) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || !a || !b) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => Object.is(a[k], b[k]));
}

export function useStore(selector) {
  const [, force] = useState(0);
  const sel = useRef(selector);
  sel.current = selector;
  const last = useRef(selector(state));
  last.current = selector(state);
  useEffect(() => {
    const l = () => {
      const next = sel.current(state);
      if (!shallowEqual(next, last.current)) {
        last.current = next;
        force((n) => n + 1);
      }
    };
    listeners.add(l);
    // State may have changed between render and subscription (e.g. the
    // initial /api/state landing first) -- catch up now or we'd never hear it.
    l();
    return () => listeners.delete(l);
  }, []);
  return last.current;
}

// ---- selection helpers -------------------------------------------------
export function select({ structures = [], edges = [] }, additive = false) {
  update((s) => {
    if (!additive) {
      s.selection = { structures: [...structures], edges: [...edges] };
      return;
    }
    const toggle = (arr, ids) => {
      const out = [...arr];
      for (const id of ids) {
        const i = out.indexOf(id);
        if (i >= 0) out.splice(i, 1); else out.push(id);
      }
      return out;
    };
    s.selection = {
      structures: toggle(s.selection.structures, structures),
      edges: toggle(s.selection.edges, edges),
    };
  });
}

export function clearSelection() {
  set({ selection: { structures: [], edges: [] } });
}

// ---- toasts -------------------------------------------------------------
let toastId = 0;
// `action`: optional {label, run} rendered as a button in the toast.
export function toast(message, kind = 'info', ms = 4500, action = null) {
  const id = ++toastId;
  update((s) => { s.toasts = [...s.toasts, { id, message, kind, action }]; });
  if (ms) setTimeout(() => update((s) => { s.toasts = s.toasts.filter((t) => t.id !== id); }), ms);
}

// ---- navigation ---------------------------------------------------------
export function openJob(jobId) {
  set({ view: { tab: 'job', jobId } });
}
// Leaving the Profiles tab with unsaved edits would silently drop them
// (jobs copy the profile *file*), so ask first.
export async function confirmProfileEdits() {
  const ed = state.profileEditor;
  if (!ed?.dirty) return true;
  if (confirm(`Save your changes to profile "${ed.name}"?\n\nOK saves them. Cancel discards them.`)) {
    return (await ed.save()) !== false;
  }
  set({ profileEditor: { ...ed, dirty: false } });
  return true;
}

export async function openTab(tab) {
  if (state.view.tab === 'profiles' && tab !== 'profiles' && !(await confirmProfileEdits())) return;
  set({ view: { tab, jobId: state.view.jobId } });
}

// ---- per-viewer conveniences (never required for correctness) ----------
export const prefs = {
  get(key, fallback) {
    try { const v = localStorage.getItem('mepd:' + key); return v === null ? fallback : JSON.parse(v); }
    catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem('mepd:' + key, JSON.stringify(value)); } catch { /* private mode */ }
  },
};
