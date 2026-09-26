// The methods each feature is built on, with citations: kept on this page so
// the rest of the UI can explain features without inline references.
import { html, useEffect, useState } from '../lib.js';
import { api } from '../api.js';

export function ReferencesView() {
  const [refs, setRefs] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    api.get('/api/references').then(setRefs).catch((e) => setError(String(e?.message || e)));
  }, []);
  return html`
    <div class="refs-view">
      <div class="view-head">
        <h2>References</h2>
        <p class="small muted">The published methods mepd's features use, and where to cite them.</p>
      </div>
      ${error && html`<p class="refs-body small">Could not load the references: ${error}</p>`}
      ${!refs && !error && html`<p class="refs-body small muted">Loading…</p>`}
      ${refs && html`<div class="refs-body">
        ${refs.map((group) => html`
          <section class="refs-group">
            <h3>${group.feature}</h3>
            <p class="small muted">Used by: ${group.where}</p>
            <dl>
              ${group.items.map((item) => html`
                <dt>${item.what}</dt>
                <dd>
                  ${item.note && html`<div class="small muted">${item.note}</div>`}
                  ${item.cite.length
                    ? html`<ul>${item.cite.map((c) => html`<li>${c.url
                        ? html`<a href=${c.url} target="_blank" rel="noopener noreferrer">${c.text}</a>` : c.text}</li>`)}</ul>`
                    : html`<div class="small muted">mepd's own method (not published).</div>`}
                </dd>`)}
            </dl>
          </section>`)}
      </div>`}
    </div>`;
}
