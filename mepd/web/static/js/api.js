// REST wrappers + the SSE connection that keeps the store in sync.
import { state, update, set, toast } from './store.js';

async function request(method, url, body, { raw = false } = {}) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  if (res.status === 401) { window.location.href = '/login'; throw new Error('not logged in'); }
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); }
    catch { /* not json */ }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  if (raw) return res;
  const ct = res.headers.get('content-type') || '';
  const out = ct.includes('application/json') ? await res.json() : await res.text();
  // A job the server just created or re-queued is ours right away: don't
  // wait for its live update (which can lag behind, e.g. through a tunnel)
  // before a view that opens it can find it.
  if (method === 'POST' && url.startsWith('/api/jobs') && !(body && body.dry_run)) rememberJobs(out);
  return out;
}

function isJob(x) {
  return x && typeof x === 'object' && typeof x.id === 'string' && x.id.startsWith('j_') && 'status' in x && 'op' in x;
}

function rememberJobs(out) {
  const jobs = (Array.isArray(out) ? out : [out]).filter(isJob);
  if (!jobs.length) return;
  update((s) => {
    const next = { ...s.jobs };
    for (const j of jobs) next[j.id] = newer(next[j.id], j);
    s.jobs = next;
  });
}

export const api = {
  get: (u) => request('GET', u),
  post: (u, b) => request('POST', u, b ?? {}),
  put: (u, b) => request('PUT', u, b),
  patch: (u, b) => request('PATCH', u, b),
  del: (u) => request('DELETE', u),
  raw: (u) => request('GET', u, undefined, { raw: true }),
};

// Wrap an action: report failures as a toast instead of an unhandled rejection.
export async function attempt(fn, success) {
  try {
    const out = await fn();
    if (success) toast(typeof success === 'function' ? success(out) : success, 'ok');
    return out;
  } catch (err) {
    toast(err.message || String(err), 'error', 9000);
    return undefined;
  }
}

// Delete everything selected (structures take their edges with them).
export async function deleteSelection() {
  const { structures, edges } = state.selection;
  const n = structures.length + edges.length;
  if (!n) return;
  const what = [structures.length && `${structures.length} structure${structures.length > 1 ? 's' : ''}`,
    edges.length && `${edges.length} edge${edges.length > 1 ? 's' : ''}`].filter(Boolean).join(' and ');
  const extra = structures.length ? ' Edges attached to deleted structures go too.' : '';
  if (!confirm(`Delete ${what} from the graph?${extra} Calculation outputs are kept.`)) return;
  const out = await attempt(() => api.post('/api/delete', { structures, edges }));
  if (out) {
    toast(`Deleted ${out.structures.length} structure(s), ${out.edges.length} edge(s)`, 'ok');
    update((s) => { s.selection = { structures: [], edges: [] }; });
  }
}

function newer(a, b) {
  if (!a) return b;
  return (a.rev ?? 0) > (b.rev ?? 0) ? a : b;
}

function applyState(data) {
  update((s) => {
    s.workspace = data.workspace;
    // A reload that was already in flight when a live update arrived must not
    // roll that job back (e.g. 'done' -> 'running'): keep the newer copy.
    const fresh = Object.fromEntries(data.jobs.map((j) => [j.id, newer(s.jobs[j.id], j)]));
    // Jobs this page learned about after the server took this snapshot
    // (queued while the reload was in flight) are missing from it: keep them.
    if (data.now_ns != null) {
      for (const [id, j] of Object.entries(s.jobs)) if (!(id in fresh) && (j.rev ?? 0) > data.now_ns) fresh[id] = j;
    }
    s.jobs = fresh;
    s.operations = data.operations;
    s.profiles = data.profiles;
    s.levelProfile = data.level_profile;
    s.validateMinima = data.validate_minima ?? true;
    s.levels = data.levels || {};
    s.pathSummaries = data.path_summaries || {};
    s.cpus = data.cpus;
    s.auth = !!data.auth;
    s.demo = data.demo || null;
    s.maxConcurrent = data.max_concurrent;
    s.loaded = true;
    // Drop selections that no longer exist.
    s.selection = {
      structures: s.selection.structures.filter((id) => id in s.workspace.structures),
      edges: s.selection.edges.filter((id) => id in s.workspace.edges),
    };
  });
}

export async function refreshState() {
  applyState(await api.get('/api/state'));
}

// Live updates. Server-Sent Events first; if the server's `hello` doesn't
// arrive promptly (a proxy that buffers streams, e.g. Cloudflare quick
// tunnels), fall back to long-polling /api/poll with the same events.
const handlers = {
  hello: () => {},
  // The server dropped our backlog (we stopped listening for a while).
  resync: () => { refreshState().catch(() => {}); },
  workspace: (ws) => {
    update((s) => {
      s.workspace = { ...ws, root: s.workspace.root };
      s.selection = {
        structures: s.selection.structures.filter((id) => id in ws.structures),
        edges: s.selection.edges.filter((id) => id in ws.edges),
      };
    });
  },
  job: (job) => {
    const prev = state.jobs[job.id];
    if (prev && newer(prev, job) === prev) return;   // stale copy, arrived late
    update((s) => { s.jobs = { ...s.jobs, [job.id]: job }; });
    if (prev && prev.status !== job.status && ['done', 'failed'].includes(job.status)) {
      toast(`${job.status === 'done' ? 'Finished' : 'Failed'}: ${job.title}`, job.status === 'done' ? 'ok' : 'error');
    }
  },
  job_deleted: ({ id }) => {
    update((s) => { const j = { ...s.jobs }; delete j[id]; s.jobs = j; });
  },
  progress: (p) => {
    update((s) => {
      const prev = s.progress[p.id] || {};
      // Streams arrive as deltas (only the ones that changed): merge them.
      const streams = p.streams ? { ...(prev.streams || {}), ...p.streams } : prev.streams;
      s.progress = { ...s.progress, [p.id]: { ...prev, ...p, streams } };
    });
  },
  profiles: (profiles) => {
    set({ profiles });
    refreshState().catch(() => {});  // profile edits can change level fingerprints
  },
  level: (d) => set({ levelProfile: d.level_profile, levels: d.levels }),
  session: () => {
    // Another workspace became current (from this tab or another): start clean.
    update((s) => {
      s.selection = { structures: [], edges: [] };
      s.progress = {};
      s.view = { tab: 'graph', jobId: null };
      s.connectMode = false;
    });
    refreshState().catch((e) => toast(e.message, 'error'));
  },
};

function dispatch(event, data) {
  const h = handlers[event];
  if (h) h(data);
}

async function pollLoop() {
  let cid = null;
  let failures = 0;
  set({ liveTransport: 'poll' });
  for (;;) {
    try {
      const res = await fetch(`/api/poll?wait=20${cid ? `&cid=${encodeURIComponent(cid)}` : ''}`);
      if (res.status === 401) { window.location.href = '/login'; return; }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = await res.json();
      if (body.cid !== cid) {
        cid = body.cid;
        refreshState().catch(() => {});  // new subscription: resync first
      }
      if (!state.connected) set({ connected: true });
      failures = 0;
      for (const e of body.events) dispatch(e.event, e.data);
    } catch {
      failures += 1;
      set({ connected: false });
      await new Promise((r) => setTimeout(r, Math.min(10000, 1000 * failures)));
    }
  }
}

export function connect() {
  refreshState().catch((e) => toast(e.message, 'error'));
  // Safety net: while anything is queued or running, re-read the full state
  // now and then, so one lost "finished" event can't leave the UI stale.
  setInterval(() => {
    if (Object.values(state.jobs).some((j) => j.status === 'running' || j.status === 'queued')) {
      refreshState().catch(() => {});
    }
  }, 15000);
  // Phones freeze background tabs (and their connections): catch up on return.
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') refreshState().catch(() => {});
  });
  const es = new EventSource('/api/events');
  let sawHello = false;
  const fallback = setTimeout(() => {
    if (sawHello) return;
    es.close();
    pollLoop();
  }, 5000);
  es.onopen = () => {
    set({ connected: true, liveTransport: 'sse' });
    refreshState().catch((e) => toast(e.message, 'error'));  // resync after any gap
  };
  es.onerror = () => set({ connected: false });
  es.addEventListener('hello', () => { sawHello = true; clearTimeout(fallback); });
  for (const name of Object.keys(handlers)) {
    if (name === 'hello') continue;
    es.addEventListener(name, (ev) => dispatch(name, JSON.parse(ev.data)));
  }
  return es;
}
