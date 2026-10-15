# Atlas research dashboard

A read-only, browser-local dashboard of recorded Atlas research. The owner requested this interface after the research pass; it does not implement the proposed profiler, training runner or optimization engine.

The private [Atlas dashboard](https://atlas-research-dashboard.andrewdang357.chatgpt.site) supports instant search, attention/checkpoint filters, sorting, memory/update-time views, configuration inspection and JSON export. Evidence and environment views keep execution results separate from scientific claims. The snapshot is dated and source-bound; there is no live training connection.

## Local preview

From the repository root:

```text
python frontend/scripts/build_snapshot.py
python frontend/scripts/check_dashboard.py
node --check frontend/app.js
python -m http.server 4173 --bind 127.0.0.1 --directory frontend
```

Open `http://127.0.0.1:4173`. Use `/` to focus search. Filters and selected configuration are shareable through the URL. Mobile selection opens a native dialog with keyboard focus handling. Tables scroll horizontally without overflowing the page.

## Latency and dependencies

Buildless HTML/CSS/JavaScript, a small local JSON snapshot, one self-hosted variable font and a tiny SVG mark. No framework, charting package, remote font request, backend, analytics, polling or hydration. The source checker enforces a 128,000-byte uncompressed first-load asset budget and exact snapshot/source identities. Its gzip estimate is not a measured transfer size.

The document exposes `data-results-ready-ms` (navigation to populated results) and `data-filter-commit-ms` (synchronous filter/DOM update) for local QA. Neither is a paint, INP or production latency guarantee. All data processing stays in the browser.

Design follows the Uncodixfy skill with a restrained warm palette, plain navigation, dense results table and directly labeled measured bars. Geist is self-hosted from a pinned upstream revision; [license](assets/OFL.txt) and [source identities](assets/font-source.json) are included. Scientific records remain unchanged.

## Private publication

The fixed identity is in `.openai/hosting.json`. `scripts/prepare_site.py` copies only the listed dashboard assets into `.tools/dashboard-site`, preserving its registered Site ID. Use the Sites source workflow from that separate ignored checkout, then publish the exact pushed commit/archive through the owner-private operation. Credentials are never stored in source. This excludes datasets, weights, runtime environments and the rest of the research repository from the Site.

The initial snapshot pins research revision `694bf49`; links open that frozen GitHub evidence and require repository access. Refresh by rerunning the generator after intentionally updating its revision/content and reviewing new evidence. Changed research records fail the snapshot identity check until regenerated; the dashboard never silently labels missing evidence valid.
