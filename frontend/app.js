const $ = (selector, root = document) => root.querySelector(selector);
const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = (value, digits = 2) => Number(value).toFixed(digits);
const state = {view:'experiments', query:'', attention:'all', checkpoint:'all', selected:'sdpa-32-0', metric:'memory', sort:'default', direction:1};
let snapshot;
const descriptions = {experiments:'Native BERT QA · logical batch 32 · sequence length 384', evidence:'What the recorded observations support—and what remains unknown.', environment:'Pinned software profiles · post-study reconciliation'};
const titles = {experiments:'Experiments', evidence:'Evidence', environment:'Environments'};
const iconSearch = '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="8" cy="8" r="5.5"/><path d="m12 12 5 5"/></svg>';

function readLocation() {
  const [view, query = ''] = location.hash.slice(1).split('?');
  state.view = Object.hasOwn(titles, view) ? view : 'experiments';
  const params = new URLSearchParams(query);
  state.query = (params.get('q') || '').slice(0, 120);
  state.attention = ['sdpa','eager'].includes(params.get('attention')) ? params.get('attention') : 'all';
  state.checkpoint = ['on','off'].includes(params.get('checkpoint')) ? params.get('checkpoint') : 'all';
  const selected = params.get('selected');
  if (snapshot.configurations.some(c => c.id === selected)) state.selected = selected;
}
function saveLocation() {
  const params = new URLSearchParams();
  if (state.query) params.set('q', state.query);
  if (state.attention !== 'all') params.set('attention', state.attention);
  if (state.checkpoint !== 'all') params.set('checkpoint', state.checkpoint);
  if (state.selected !== 'sdpa-32-0') params.set('selected', state.selected);
  history.replaceState(null, '', `#${state.view}${params.size ? '?'+params : ''}`);
}
function filtered() {
  const tokens = state.query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const rows = snapshot.configurations.filter(c => {
    const text = `${c.attention} microbatch ${c.microbatch} checkpoint ${c.checkpoint ? 'on' : 'off'} ${c.nativeDefault ? 'native default' : ''}`;
    return tokens.every(token => text.includes(token)) && (state.attention === 'all' || c.attention === state.attention) && (state.checkpoint === 'all' || c.checkpoint === (state.checkpoint === 'on'));
  });
  if (state.sort !== 'default') rows.sort((a,b) => state.direction * (state.sort === 'memory' ? a.allocatedGiB-b.allocatedGiB : Math.max(...a.updateSeconds)-Math.max(...b.updateSeconds)));
  return rows;
}
function source(name) { return escape(snapshot.sourceBase+name); }
function range(c, digits = 2) { return `${number(Math.min(...c.updateSeconds),digits)}–${number(Math.max(...c.updateSeconds),digits)}`; }
function chart(rows) {
  const memory = state.metric === 'memory';
  const scale = memory ? 16 : 5;
  const bars = rows.map(c => {
    const value = memory ? c.allocatedGiB : Math.max(...c.updateSeconds);
    const width = Math.min(100, value/scale*100);
    const line = snapshot.device.cudaTotalGiB/scale*100;
    return `<button class="chart-row ${c.attention}" data-select="${c.id}" aria-pressed="${state.selected === c.id}" aria-label="Inspect ${c.attention.toUpperCase()}, microbatch ${c.microbatch}, checkpoint ${c.checkpoint ? 'on' : 'off'}"><span class="chart-label">${c.attention.toUpperCase()} · ${c.microbatch} · CP ${c.checkpoint ? 'on' : 'off'}${c.nativeDefault ? '<span class="native">default</span>' : ''}</span><svg class="bar-track" viewBox="0 0 100 15" preserveAspectRatio="none" aria-hidden="true"><rect class="bar-fill" width="${width}" height="15" stroke="none"/>${memory ? `<line class="device-line" x1="${line}" x2="${line}" y1="0" y2="15" vector-effect="non-scaling-stroke"/>` : ''}</svg><span class="chart-value">${memory ? number(value)+' GiB' : range(c)+' s'}</span></button>`;
  }).join('');
  const ticks = memory ? ['0','4','8','12','16 GiB'] : ['0','1','2','3','4','5 s'];
  return `<section class="panel" aria-label="Native resource comparison"><div class="panel-header"><div><h2>Native BERT tradeoffs</h2><p>${memory ? 'Later allocated peak · same in both repetitions' : 'Two process medians · not confidence intervals'}</p></div><div class="tabs" aria-label="Chart measurement"><button data-metric="memory" aria-pressed="${memory}">Memory</button><button data-metric="time" aria-pressed="${!memory}">Update time</button></div></div><div class="chart">${bars || '<p class="empty">No matching configurations.</p>'}<div class="chart-axis"><span></span><div class="ticks">${ticks.map(t => '<span>'+t+'</span>').join('')}</div><span></span></div></div><p class="chart-foot">${memory ? '<strong>Dashed line: 11.74 GiB CUDA-reported total.</strong> Allocator peaks do not establish physical residency.' : 'Each process median summarizes five synchronized complete updates. Both recorded medians are shown as a range.'}</p></section>`;
}
function sortHeading(label, key) {
  return `<th aria-sort="${state.sort === key ? (state.direction === 1 ? 'ascending' : 'descending') : 'none'}"><button data-sort="${key}" aria-label="Sort by ${label}">${label} <span aria-hidden="true">${state.sort === key ? (state.direction === 1 ? '↑' : '↓') : '↕'}</span></button></th>`;
}
function table(rows) {
  return `<div class="table-scroll"><table aria-label="Recorded BERT resource configurations"><thead><tr><th>Configuration</th><th>Microbatch</th>${sortHeading('Allocated', 'memory')}${sortHeading('Update', 'time')}<th>Segments</th></tr></thead><tbody>${rows.map(c => `<tr class="${state.selected === c.id ? 'selected' : ''}" data-row="${c.id}"><td><button class="row-button" data-select="${c.id}" aria-label="Inspect ${c.attention.toUpperCase()} microbatch ${c.microbatch} checkpoint ${c.checkpoint ? 'on' : 'off'}">${c.attention.toUpperCase()} ${c.checkpoint ? '+ checkpoint' : ''}<span class="subtext">${c.nativeDefault ? 'Native default' : c.checkpoint ? 'Recompute activations' : 'No checkpointing'}</span></button></td><td class="numeric">${c.microbatch}<span class="subtext">Logical batch 32</span></td><td class="numeric">${number(c.allocatedGiB)} GiB</td><td class="numeric">${range(c)} s</td><td class="completion">✓ ${c.completed}/${c.intended}<span>complete</span></td></tr>`).join('')}</tbody></table></div>${rows.length ? '' : '<div class="empty"><strong>No configurations found</strong><span>Try a different search or clear the filters.</span><br><button class="button" data-clear>Clear filters</button></div>'}`;
}
function inspector(c) {
  if (!filtered().length) return '<div class="detail-section"><h3>No configuration selected</h3><p class="detail-text">Clear or change the filters to inspect a recorded configuration.</p></div>';
  return `<div class="inspector-head"><h2>${c.attention.toUpperCase()} · microbatch ${c.microbatch}</h2><p>Checkpoint ${c.checkpoint ? 'on' : 'off'} · FP32 · dropout 0.1</p>${c.nativeDefault ? '<p class="configuration-kind">Native library default</p>' : ''}</div><section class="detail-section"><h3>Resource segment</h3><dl class="detail-facts"><dt>Peak allocated</dt><dd>${number(c.allocatedGiB)} GiB</dd><dt>Update medians</dt><dd>${range(c,3)} s</dd><dt>Processes completed</dt><dd>${c.completed} / ${c.intended}</dd><dt>Updates per process</dt><dd>${c.updatesPerProcess}</dd><dt>Sequence length</dt><dd>384 tokens</dd><dt>Logical batch</dt><dd>32 features</dd></dl></section><section class="detail-section"><h3>Evidence scope</h3><div class="evidence-line"><span>Execution</span><span class="state pass">Completed</span></div><div class="evidence-line"><span>CUDA allocation</span><span class="state pass">Measured</span></div><div class="evidence-line"><span>Worker RSS</span><span class="state fail">Invalid</span></div><div class="evidence-line"><span>QA quality</span><span class="state unknown">Unverified</span></div><p class="detail-text">Completion supports this eight-update segment. It does not establish scientific equivalence or a rescued OOM workload.</p></section><section class="detail-section"><h3>Reference repeatability</h3><p class="detail-text">${c.hashRepeat ? 'Recorded state, gradient and optimizer hashes match between these two same-seed resource repetitions. Byte identity does not establish the intended objective or full-training quality.' : 'Recorded state, gradient or optimizer hashes differ between these same-seed resource repetitions. This screen has no tensor-distance verdict; keep reference stability separate from transformation attribution.'}</p><a class="text-link" href="${source('qa-reference32-control.md')}" target="_blank" rel="noopener noreferrer">Read the separate reference diagnostic ↗</a></section><section class="detail-section"><a class="button raw-link" href="${source('qa-resource-pilot.json')}" target="_blank" rel="noopener noreferrer">Open frozen source record ↗</a></section>`;
}
function filtersMarkup() {
  return `<div class="filters"><label class="search-field">${iconSearch}<span class="sr-only">Search configurations</span><input id="search" type="search" placeholder="Search configurations…" maxlength="120" value="${escape(state.query)}" autocomplete="off" spellcheck="false"></label><label class="filter-select"><span class="sr-only">Attention</span><select id="attention" aria-label="Attention"><option value="all">All attention</option><option value="sdpa" ${state.attention === 'sdpa' ? 'selected' : ''}>SDPA</option><option value="eager" ${state.attention === 'eager' ? 'selected' : ''}>Eager</option></select></label><label class="filter-select"><span class="sr-only">Checkpoint</span><select id="checkpoint" aria-label="Checkpoint"><option value="all">Checkpoint: all</option><option value="on" ${state.checkpoint === 'on' ? 'selected' : ''}>Checkpoint: on</option><option value="off" ${state.checkpoint === 'off' ? 'selected' : ''}>Checkpoint: off</option></select></label></div>`;
}
function experimentMarkup() {
  const rows = filtered();
  if (rows.length && !rows.some(c => c.id === state.selected)) state.selected = rows[0].id;
  return `<div class="workspace"><div class="results-column"><div id="chart-region">${chart(rows)}</div><div id="counts" class="result-count">${counts(rows)}</div><section class="panel table-panel" aria-label="Experiment results">${filtersMarkup()}<div id="table-region">${table(rows)}</div><div class="table-bottom"><span id="table-count">${rows.length} of ${snapshot.configurations.length} configurations</span><button class="clear-filters" data-clear>Reset filters</button><button class="button mobile-detail" data-inspect>Inspect selected</button></div></section><p class="scope-note">All results are saved development evidence on one RTX 3060. <a href="${source('qa-resource-pilot.md')}" target="_blank" rel="noopener noreferrer">Method and limitations ↗</a><br>Efficient attention is already the pinned library default. These results do not demonstrate an Atlas policy advantage.</p></div><aside id="inspector" class="panel inspector" aria-label="Selected configuration">${inspector(selected())}</aside></div>`;
}
function selected() { return snapshot.configurations.find(c => c.id === state.selected) || snapshot.configurations[0]; }
function counts(rows) {
  return `<span><strong>${rows.length}</strong> configurations</span><span><strong>${rows.reduce((n,c) => n+c.completed,0)}</strong> completed segments</span><span><strong>${rows.reduce((n,c) => n+c.completed*c.updatesPerProcess,0)}</strong> logical updates</span>`;
}
function evidenceMarkup() {
  return `<p class="notice">A completed run is an execution result. It is not a scientific-validity verdict.</p><section class="evidence-list" aria-label="Scoped evidence">${snapshot.evidence.map(e => `<article class="evidence-item"><div><h3>${escape(e.name)}</h3><p>${escape(e.scope)}</p></div><span class="state ${e.tone}">${escape(e.status)}</span><p>${escape(e.description)}</p><a href="${source(e.source)}" target="_blank" rel="noopener noreferrer" aria-label="Read ${escape(e.name)} source">↗</a></article>`).join('')}</section><p class="scope-note">PASS and FAIL above describe the named finite comparison. They cannot be extended to a different field, precision, workload or training horizon.<br><a href="${escape(snapshot.sourceBase.replace('workload-screen/',''))}../architecture/claim-applicability.md" id="claim-link" target="_blank" rel="noopener noreferrer">Evidence and claim boundaries ↗</a></p>`;
}
function environmentMarkup() {
  return `<div class="environment-info"><span><strong>8</strong> pinned profiles</span><span><strong>54</strong> selected source files</span><span>Recorded 15 October 2026</span></div><p class="notice">Post-study reconciliation covers package versions and selected source files. It is not contemporaneous worker attestation or a whole-package audit.</p><section class="panel table-panel"><div class="table-scroll"><table class="environment-table" aria-label="Recorded environment profiles"><thead><tr><th>Profile</th><th>Python</th><th>PyTorch</th><th>Packages</th><th>Sources checked</th><th>Reconciliation</th></tr></thead><tbody>${snapshot.environments.map(e => `<tr><td><a href="${escape(e.requirements)}" target="_blank" rel="noopener noreferrer">${escape(e.name)} ↗</a></td><td class="numeric">${escape(e.python)}</td><td class="numeric">${escape(e.torch || 'Not installed')}</td><td class="numeric">${e.packages}</td><td class="numeric">${e.checkedSources}</td><td class="state ${e.matched ? 'pass' : 'fail'}">${e.matched ? 'Matched' : 'Mismatch'}</td></tr>`).join('')}</tbody></table></div></section><p class="scope-note">The application does not need these eight research environments to run. This dashboard only reads a small saved snapshot.<br><a href="${source('post-study-environments.md')}" target="_blank" rel="noopener noreferrer">Read provenance and measurement limits ↗</a></p>`;
}
function render() {
  document.title = `${titles[state.view]} · Atlas`;
  $('#page-title').textContent = titles[state.view];
  $('#breadcrumb-view').textContent = titles[state.view];
  $('#page-description').textContent = descriptions[state.view];
  document.querySelectorAll('[data-view]').forEach(a => {
    if (a.dataset.view === state.view) a.setAttribute('aria-current','page'); else a.removeAttribute('aria-current');
  });
  $('#content').innerHTML = state.view === 'experiments' ? experimentMarkup() : state.view === 'evidence' ? evidenceMarkup() : environmentMarkup();
  saveLocation();
}
function refreshResults(announce = true) {
  const started = performance.now();
  const rows = filtered();
  if (!rows.some(c => c.id === state.selected) && rows.length) state.selected = rows[0].id;
  $('#chart-region').innerHTML = chart(rows);
  $('#table-region').innerHTML = table(rows);
  $('#counts').innerHTML = counts(rows);
  $('#table-count').textContent = `${rows.length} of ${snapshot.configurations.length} configurations`;
  $('#inspector').innerHTML = rows.length ? inspector(selected()) : '<div class="detail-section"><h3>No configuration selected</h3><p class="detail-text">Clear or change the filters to inspect a recorded configuration.</p></div>';
  $('[data-inspect]').disabled = !rows.length;
  if (announce) $('#announcement').textContent = `${rows.length} matching configurations`;
  saveLocation();
  document.documentElement.dataset.filterCommitMs = number(performance.now()-started, 2);
}
function showDetails() {
  $('#mobile-inspector').innerHTML = '<div class="dialog-head"><span>Configuration details</span><button class="close-dialog" data-close aria-label="Close configuration details">×</button></div>'+inspector(selected());
  $('#mobile-inspector').showModal();
}
function exportResults() {
  const result = {schemaVersion: snapshot.schemaVersion, snapshotDate: snapshot.snapshotDate, researchRevision: snapshot.researchRevision, sourceInputs: snapshot.sourceInputs, scope:snapshot.scope, view:state.view,
    ...(state.view === 'experiments' ? {filters:{query:state.query,attention:state.attention,checkpoint:state.checkpoint}, configurations:filtered()} : state.view === 'evidence' ? {evidence:snapshot.evidence} : {environments:snapshot.environments,scope:snapshot.environmentScope})};
  const url = URL.createObjectURL(new Blob([JSON.stringify(result,null,2)+'\n'],{type:'application/json'}));
  const link = document.createElement('a');
  link.href = url; link.download = `atlas-${state.view}-${snapshot.snapshotDate}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  $('#announcement').textContent = 'Results exported as JSON';
}
document.addEventListener('click', event => {
  const button = event.target.closest('button');
  if (!button || !snapshot) return;
  if (button.hasAttribute('data-select')) {
    state.selected = button.dataset.select;
    refreshResults(false);
    if (matchMedia('(max-width:1190px)').matches) showDetails();
    else $(`.row-button[data-select="${state.selected}"]`)?.focus({preventScroll:true});
  } else if (button.hasAttribute('data-metric')) {
    state.metric = button.dataset.metric; $('#chart-region').innerHTML = chart(filtered());
    $(`[data-metric="${state.metric}"]`)?.focus({preventScroll:true});
  } else if (button.hasAttribute('data-sort')) {
    state.direction = state.sort === button.dataset.sort ? -state.direction : 1;
    state.sort = button.dataset.sort; refreshResults();
    $(`[data-sort="${state.sort}"]`)?.focus({preventScroll:true});
  } else if (button.hasAttribute('data-clear')) {
    state.query = ''; state.attention = 'all'; state.checkpoint = 'all'; state.sort = 'default';
    $('#search').value = ''; $('#attention').value = 'all'; $('#checkpoint').value = 'all'; refreshResults();
  } else if (button.hasAttribute('data-inspect')) showDetails();
  else if (button.hasAttribute('data-close')) $('#mobile-inspector').close();
});
document.addEventListener('input', event => {
  if (event.target.id === 'search') { state.query = event.target.value; refreshResults(); }
});
document.addEventListener('change', event => {
  if (['attention','checkpoint'].includes(event.target.id)) { state[event.target.id] = event.target.value; refreshResults(); }
});
document.addEventListener('keydown', event => {
  if (event.key === '/' && !['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName) && !$('#mobile-inspector').open && $('#search')) {
    event.preventDefault(); $('#search').focus();
  }
});
window.addEventListener('hashchange', () => { if (snapshot && location.hash !== '#main') { readLocation(); render(); } });
$('#export').addEventListener('click', exportResults);
async function load() {
  try {
    const response = await fetch('data/snapshot.json');
    if (!response.ok) throw new Error('Snapshot unavailable');
    snapshot = await response.json();
    if (snapshot.schemaVersion !== 1 || snapshot.configurations.length !== 8) throw new Error('Unsupported snapshot');
    readLocation(); render(); $('#load-status').hidden = true; $('#content').hidden = false; $('#export').disabled = false;
    document.documentElement.dataset.resultsReadyMs = number(performance.now(), 2);
  } catch {
    $('#load-status').className = 'loading-state error';
    $('#load-status').innerHTML = '<p>The saved research snapshot could not be loaded. No experiment status has been inferred.</p><button class="button" id="retry">Try again</button>';
    $('#retry').addEventListener('click', load, {once:true});
  }
}
load();
