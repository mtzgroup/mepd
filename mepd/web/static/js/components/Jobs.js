// All jobs, newest first, with live status lines.
import { html, useEffect, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, set, useStore } from '../store.js';
import { STATUS_LABEL, fmtAgo, fmtDuration, jobElapsed, lastLine } from '../util.js';

export function useTick(ms = 1000, on = true) {
  const [, setT] = useState(0);
  useEffect(() => {
    if (!on) return undefined;
    const id = setInterval(() => setT((t) => t + 1), ms);
    return () => clearInterval(id);
  }, [ms, on]);
}

export function JobControls({ job, compact = false }) {
  const cancel = () => attempt(() => api.post(`/api/jobs/${job.id}/cancel`), 'Cancelling…');
  const retry = () => attempt(() => api.post(`/api/jobs/${job.id}/retry`), 'Re-queued: it resumes from what is already on disk');
  const del = async () => {
    if (!confirm(`Delete "${job.title}"${job.external ? ' from the list (the output directory is left alone)' : ' and its output folder'}?`)) return;
    await attempt(() => api.del(`/api/jobs/${job.id}`), 'Deleted');
    set({ view: { tab: 'jobs', jobId: null } });
  };
  const stop = (fn) => (e) => { e.stopPropagation(); fn(); };
  return html`<span class="job-controls">
    ${['queued', 'running'].includes(job.status) && html`<button class="btn small" onClick=${stop(cancel)}>Cancel</button>`}
    ${['failed', 'cancelled', 'interrupted'].includes(job.status) && !job.external && html`<button class="btn small" onClick=${stop(retry)} title="Rerun into the same output folder; finished pieces are skipped">Resume</button>`}
    ${job.status === 'done' && !job.external && !compact && html`<button class="btn small" onClick=${stop(retry)} title="Rerun into the same output folder">Rerun</button>`}
    ${!compact && job.status !== 'queued' && html`<a class="btn small" href=${`/api/jobs/${job.id}/archive`} title="Output folder, inputs and logs as a zip">Download .zip</a>`}
    ${!['queued', 'running'].includes(job.status) && html`<button class="btn small ghost" onClick=${stop(del)} title="Delete">✕</button>`}
  </span>`;
}

export function JobsView() {
  const jobs = useStore((s) => s.jobs);
  const operations = useStore((s) => s.operations);
  const progress = useStore((s) => s.progress);
  const [filter, setFilter] = useState('all');
  const list = Object.values(jobs).sort((a, b) => b.created - a.created)
    .filter((j) => filter === 'all' || (filter === 'active' ? ['queued', 'running'].includes(j.status) : j.status === filter));
  const anyRunning = list.some((j) => j.status === 'running');
  useTick(1000, anyRunning);
  const opTitle = Object.fromEntries(operations.map((o) => [o.key, o.title]));
  const counts = Object.values(jobs).reduce((acc, j) => { acc[j.status] = (acc[j.status] || 0) + 1; return acc; }, {});

  return html`
    <div class="jobs-view">
      <div class="view-head">
        <h2>Calculations</h2>
        <div class="segmented">
          ${['all', 'active', 'done', 'failed'].map((f) => html`<button class=${filter === f ? 'on' : ''} onClick=${() => setFilter(f)}>
            ${f}${f === 'active' ? ` (${(counts.running || 0) + (counts.queued || 0)})` : f !== 'all' && counts[f] ? ` (${counts[f]})` : ''}</button>`)}
        </div>
        ${!useStore((s) => s.demo) && html`<button class="btn" onClick=${() => set({ modal: { kind: 'import' } })}>Open existing output…</button>`}
      </div>
      ${list.length === 0
        ? html`<div class="empty-hint"><p>No calculations${filter !== 'all' ? ` (${filter})` : ''} yet.</p>
            <p class="small muted">Select structures or edges in the graph and pick an action, or open an existing mepd output folder.</p></div>`
        : html`<table class="jobs-table">
            <thead><tr><th>Status</th><th>Calculation</th><th>Result / latest output</th><th>Time</th><th></th></tr></thead>
            <tbody>
              ${list.map((j) => html`
                <tr key=${j.id} onClick=${() => openJob(j.id)}>
                  <td><span class=${`pill ${j.status}`}>${STATUS_LABEL[j.status]}</span></td>
                  <td><div class="job-title">${j.title}</div>
                    <div class="small muted">${opTitle[j.op] || j.op}${j.batch ? ' · batch' : ''}${j.profile ? ` · ${j.profile}` : ''}</div></td>
                  <td class="job-line">${j.summary?.headline || (j.status === 'failed' ? (j.error || '').split('\n').pop() : lastLine(j, progress)) || ''}</td>
                  <td class="nowrap small">${j.external ? 'imported' : j.status === 'queued' ? fmtAgo(j.created) : fmtDuration(jobElapsed(j))}</td>
                  <td class="nowrap"><${JobControls} job=${j} compact /></td>
                </tr>`)}
            </tbody>
          </table>`}
    </div>`;
}
