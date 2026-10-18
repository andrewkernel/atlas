const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors={gpu_dense:'#428e8c',resident:'#326b49',native:'#326b49',copy:'#bc8b48',cpu:'#7c8ca0',hybrid25:'#c2b4d1',hybrid50:'#9c85b5',hybrid75:'#68517f',adaptive:'#b06852'};
const labels={gpu_dense:'GPU dense reference',resident:'Native GPU',copy:'Transfer to GPU',cpu:'CPU attention',hybrid25:'Hybrid · 25% GPU',hybrid50:'Hybrid · 50% GPU',hybrid75:'Hybrid · 75% GPU',native:'Native model SDPA',adaptive:'Atlas adaptive'};
const state={token:'',runs:[],doc:null,selected:null,kind:'benchmark',view:'experiments',hidden:new Set(),sort:'context',ascending:true,active:null};
const fmt=(n,d=2)=>n==null?'—':Number(n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
const date=x=>x?new Date(x).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):'';

async function api(path,options={}){
  const response=await fetch(path,{...options,headers:{'Content-Type':'application/json','X-Atlas-Token':state.token,...options.headers}});
  if(!response.ok){let message;try{message=(await response.json()).detail}catch{message=response.statusText}throw new Error(typeof message==='string'?message:JSON.stringify(message))}
  return response.json();
}
function notice(message){$('#message').textContent=message;$('#message').hidden=false}
function clearNotice(){$('#message').hidden=true}
function pill(status){return `<span class="pill ${['failed','timed_out'].includes(status)?'fail':['running','queued','cancelled'].includes(status)?'warn':''}">${esc(status)}</span>`}

async function refresh(prefer){
  const [session,runs]=await Promise.all([api('/api/session'),api('/api/runs')]);
  state.token=session.token;state.runs=runs;state.active=session.active_run;
  $('#run-model').disabled=!session.model_ready||!!state.active;
  $('#new-run').disabled=!!state.active;
  const filtered=runs.filter(r=>r.config?.kind===state.kind);
  const selection=prefer||state.selected;
  state.selected=filtered.some(r=>r.id===selection)?selection:(filtered.find(r=>r.status==='completed')||filtered[0])?.id;
  $('#run-select').innerHTML=filtered.map(r=>`<option value="${esc(r.id)}">${esc(date(r.created_at))} · ${esc(r.config.profile||'model')} · ${esc(r.id.slice(-6))} · ${esc(r.status)}</option>`).join('');
  $('#run-select').value=state.selected||'';
  $('#active-job').hidden=!state.active;
  const active=runs.find(r=>r.id===state.active);
  if(active){$('#job-title').textContent=active.config.kind==='model'?'Model comparison':'Attention sweep';$('#job-detail').textContent=active.progress?.message||'Preparing';$('#job-progress').max=active.progress?.total||1;$('#job-progress').value=active.progress?.done||0}
  if(state.selected){await loadRun(state.selected)}else{state.doc=null;$('#results').hidden=true;$('#empty').hidden=false}
  const modelRun=runs.find(r=>r.config?.kind==='model'&&r.status==='completed');
  if(modelRun){const modelDoc=state.doc?.id===modelRun.id?state.doc:await api(`/api/runs/${modelRun.id}`);renderModel(modelDoc)}
}

async function loadRun(id){
  const previous=state.doc?.id;state.selected=id;state.doc=await api(`/api/runs/${id}`);
  if(previous!==id){clearNotice();state.hidden.clear();$('#phase-filter').value='all';$('#plan-filter').value='all'}
  const doc=state.doc,rows=doc.rows||[];
  $('#empty').hidden=rows.length>0||state.active===doc.id;$('#results').hidden=!rows.length;
  if(doc.error)notice(`${doc.status}: ${doc.error}`);
  if(!rows.length)return;
  const fastest=rows.reduce((a,b)=>a.median_ms<b.median_ms?a:b);
  $('#fastest').textContent=fastest.label;$('#fastest-detail').textContent=`${fmt(fastest.median_ms)} ms at ${fastest.context.toLocaleString()} tokens`;
  const passed=rows.filter(r=>r.correctness?.passed).length;
  $('#checks').textContent=`${passed} / ${rows.length} passed`;
  $('#checks').classList.toggle('numerics',passed===rows.length);
  $('#scope').textContent=doc.config.kind==='model'?'GPT-2 · 124M':`${new Set(rows.map(r=>r.context)).size} context lengths`;
  $('#scope-detail').textContent=doc.config.kind==='model'?`${doc.protocol?.decode_steps} decoding steps · float32`:`12 heads × 64 dimensions · float32`;
  $('#run-state').outerHTML=`<span id="run-state" class="pill ${doc.status==='completed'?'':doc.status==='failed'?'fail':'warn'}">${esc(doc.status)}</span>`;
  $('#run-date').textContent=date(doc.created_at);
  const ids=[...new Set(rows.map(r=>r.plan))];
  const prior=$('#plan-filter').value;
  $('#plan-filter').innerHTML='<option value="all">All execution plans</option>'+ids.map(p=>`<option value="${p}">${esc(labels[p]||p)}</option>`).join('');
  $('#plan-filter').value=ids.includes(prior)?prior:'all';
  $('#legend').innerHTML=ids.map(p=>`<button data-series="${p}" aria-pressed="${!state.hidden.has(p)}" class="${state.hidden.has(p)?'off':''}"><svg width="8" height="8" aria-hidden="true"><circle cx="4" cy="4" r="3.5" fill="${colors[p]||'#777'}"/></svg>${esc(labels[p]||p)}</button>`).join('');
  $$('#legend button').forEach(b=>b.onclick=()=>{const id=b.dataset.series;state.hidden.has(id)?state.hidden.delete(id):state.hidden.add(id);b.classList.toggle('off',state.hidden.has(id));b.setAttribute('aria-pressed',!state.hidden.has(id));renderChart()});
  $('#evidence').innerHTML=`<p>${(doc.limitations||[]).map(esc).join('<br>')}</p><pre>${esc(JSON.stringify({environment:doc.environment,protocol:doc.protocol,calibration_elapsed_s:doc.calibration_elapsed_s,policy_source:doc.policy_source},null,2))}</pre>`;
  renderPolicy();renderTable();renderChart();
}

function renderPolicy(){
  const doc=state.doc,evaluation=doc.evaluation||[];
  $('#policy-panel').hidden=!evaluation.length;
  if(!evaluation.length)return;
  $('#policy-panel').innerHTML=`<strong>Adaptive selection / held-out lengths</strong> <span>· calibrated in ${fmt(doc.calibration_elapsed_s,1)} s · fixed comparator: ${esc(labels[doc.fixed_plan])}</span><div class="decisions">${evaluation.map(e=>`<div class="decision"><span>${e.context.toLocaleString()} TOKENS</span><b>${esc(labels[e.selected])}</b> · ${fmt(e.selected_ms)} ms<br><small>${fmt(e.regret_percent,1)}% slower than measured optimum</small></div>`).join('')}</div>`;
}

function renderTable(){
  if(!state.doc)return;
  const phase=$('#phase-filter').value,plan=$('#plan-filter').value;
  const rows=state.doc.rows.map((r,i)=>({...r,index:i})).filter(r=>(phase==='all'||r.phase===phase)&&(plan==='all'||r.plan===plan)).sort((a,b)=>(a[state.sort]-b[state.sort])*(state.ascending?1:-1));
  $('#row-count').textContent=rows.length;
  $('#measurements').innerHTML=rows.map(r=>`<tr data-row="${r.index}"><td>${r.context.toLocaleString()}</td><td><span class="plan-name"><svg width="8" height="8" aria-hidden="true"><circle cx="4" cy="4" r="3.5" fill="${colors[r.plan]||'#777'}"/></svg>${esc(r.label)}</span></td><td class="phase">${r.phase==='validation'?'Held out':esc(r.phase)}</td><td>${fmt(r.median_ms)} <span class="muted">ms</span></td><td>${fmt(r.p90_ms)} <span class="muted">ms</span></td><td>${fmt(r.gpu_kv_mib,1)} <span class="muted">MiB</span></td><td class="numerics ${r.correctness?.passed?'':'fail'}">${r.correctness?.passed?'✓ Within bounds':'× Outside bounds'}</td><td><button class="inspect-row" aria-label="Inspect ${esc(r.label)}, ${r.context} tokens">↗</button></td></tr>`).join('');
  $$('#measurements tr').forEach(tr=>{tr.onclick=()=>inspect(state.doc.rows[Number(tr.dataset.row)]);tr.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();tr.click()}}});
}

function renderChart(){
  if(!state.doc)return;
  const all=state.doc.rows,rows=all.filter(r=>!state.hidden.has(r.plan));
  const model=state.doc.config.kind==='model';
  $('#chart-title').textContent=model?'Full-model decode latency':'Latency across context lengths';
  $('#chart-note').textContent=model?'Teacher-forced native token history · cache updates included':'Calibration points ● · held-out lengths ◇';
  const W=Math.max(260,$('#chart').clientWidth),H=$('#chart').clientHeight||265,L=45,R=12,T=15,B=model?58:33,cw=W-L-R,ch=H-T-B;
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="plot-title plot-desc"><title id="plot-title">${model?'Model decode latency':'Attention latency by context length'}</title><desc id="plot-desc">Synchronized wall time in milliseconds. Exact values are available in the measurements table.</desc>`;
  if(!rows.length){$('#chart').innerHTML=svg+'<text x="400" y="120">Select an execution plan in the legend.</text></svg>';return}
  const log=$('#log-scale').checked;
  const transform=x=>log?Math.log10(Math.max(x,.001)):x;
  const max=Math.max(...rows.map(r=>r.median_ms))*1.1;
  const min=log?Math.min(...rows.map(r=>r.median_ms))*.7:0;
  const ymin=transform(min),ymax=transform(max);
  const y=v=>T+ch-(transform(v)-ymin)/(ymax-ymin||1)*ch;
  for(let n=0;n<=4;n++){const py=T+ch*n/4;const val=log?10**(ymax-(ymax-ymin)*n/4):max*(1-n/4);svg+=`<line x1="${L}" y1="${py}" x2="${W-R}" y2="${py}" class="grid"/><text x="${L-10}" y="${py+3}" text-anchor="end">${fmt(val,val<1?2:1)}</text>`}
  svg+=`<text x="${L-10}" y="6" text-anchor="end">ms</text>`;
  if(model){const step=cw/rows.length;rows.forEach((r,i)=>{const px=L+step*i+step*.2,py=y(r.median_ms);svg+=`<rect x="${px}" y="${py}" width="${step*.6}" height="${T+ch-py}" rx="3" fill="${colors[r.plan]||'#777'}"><title>${esc(r.label)}: ${fmt(r.median_ms)} ms</title></rect><text x="${px+step*.3}" text-anchor="end" transform="rotate(-30 ${px+step*.3} ${H-37})" y="${H-37}">${esc(r.plan)}</text>`})}
  else{
    const lengths=[...new Set(all.map(r=>r.context))].sort((a,b)=>a-b),xmin=Math.log2(lengths[0]),xmax=Math.log2(lengths.at(-1));
    const x=v=>L+(Math.log2(v)-xmin)/(xmax-xmin||1)*cw;
    lengths.forEach(n=>{svg+=`<text x="${x(n)}" y="${H-12}" text-anchor="middle">${n>=1024?`${n/1024}k`:n}</text>`});
    [...new Set(rows.map(r=>r.plan))].forEach(p=>{const series=rows.filter(r=>r.plan===p).sort((a,b)=>a.context-b.context);svg+=`<path class="series" stroke="${colors[p]}" d="${series.map((r,i)=>`${i?'L':'M'}${x(r.context)},${y(r.median_ms)}`).join(' ')}"/>`;series.forEach(r=>{const px=x(r.context),py=y(r.median_ms),title=`<title>${esc(r.label)}, ${r.context} tokens: ${fmt(r.median_ms)} ms</title>`;svg+=r.phase==='validation'?`<path d="M${px},${py-4}l4,4 -4,4 -4,-4Z" fill="white" stroke="${colors[p]}" stroke-width="1.8">${title}</path>`:`<circle cx="${px}" cy="${py}" r="3" fill="${colors[p]}">${title}</circle>`})});
  }
  $('#chart').innerHTML=svg+'</svg>';
}

function inspect(row){
  $('#inspector-title').textContent=`${row.label} / ${row.context.toLocaleString()} tokens`;
  const samples=row.samples_ms||[],max=Math.max(...samples,1e-9);
  $('#inspector-content').innerHTML=`<div class="detail-grid"><div><span>MEDIAN</span><strong>${fmt(row.median_ms)} ms</strong></div><div><span>P90</span><strong>${fmt(row.p90_ms)} ms</strong></div><div><span>ALLOCATED PEAK</span><strong>${fmt(row.allocated_peak_mib,1)} MiB</strong></div></div><div class="inspector-section-title">RAW SYNCHRONIZED SAMPLES / ${samples.length} OBSERVATIONS</div><svg viewBox="0 0 550 100" width="100%" height="100" role="img" aria-label="Raw latency samples">${samples.map((v,i)=>`<rect x="${i*550/samples.length}" y="${100-v/max*92}" width="${Math.max(1,550/samples.length-3)}" height="${v/max*92}" fill="#71956a"><title>Sample ${i+1}: ${fmt(v,4)} ms</title></rect>`).join('')}</svg><div class="inspector-section-title">NUMERICAL CHECK</div><pre class="detail-json">${esc(JSON.stringify(row.correctness,null,2))}</pre><details><summary>All measurement fields</summary><pre class="detail-json">${esc(JSON.stringify(row,null,2))}</pre></details>`;
  $('#inspector').showModal();
}

function renderModel(doc){
  if(doc.continuation)$('#continuation').textContent=doc.continuation.text;
  $('#model-comparison').innerHTML=(doc.rows||[]).map(r=>`<div class="mini-plan"><span>${esc(r.label)}</span><span>${fmt(r.tokens_per_s,1)} <small>tok/s</small></span><span class="numerics ${r.correctness?.passed?'':'fail'}">${r.correctness?.passed?'✓':'×'} logits</span></div>`).join('');
}

async function start(config){
  clearNotice();
  const result=await api('/api/runs',{method:'POST',body:JSON.stringify(config)});
  state.kind=config.kind;state.selected=result.id;
  $$('.segmented button').forEach(b=>b.classList.toggle('active',b.dataset.kind===state.kind));
  await refresh(result.id);return result;
}

function view(){
  const requested=location.hash.slice(1)||'experiments';
  state.view=['experiments','model','architecture','research'].includes(requested)?requested:'experiments';
  $$('.view').forEach(el=>el.hidden=el.id!==`view-${state.view}`);
  $$('nav a').forEach(a=>{a.classList.toggle('active',a.dataset.view===state.view);a.setAttribute('aria-current',a.dataset.view===state.view?'page':'false')});
  const titles={experiments:['Find the crossover.','Explore where attention belongs: GPU, CPU, or somewhere between.'],model:['Follow every token.','Run a pretrained model through the same execution paths.'],architecture:['Built to be inspected.','The runtime, the evidence, and the decisions connecting them.'],research:['Start with the papers.','Source material, connections, and the boundaries of this study.']};
  $('#page-title').textContent=titles[state.view][0];$('#page-subtitle').textContent=titles[state.view][1];$('#breadcrumb').textContent=state.view==='model'?'Model playground':state.view==='architecture'?'Architecture':state.view==='research'?'Research library':'Attention Lab';
}

$('#new-run').onclick=()=>$('#new-dialog').showModal();$('#empty-run').onclick=()=>$('#new-dialog').showModal();
$$('.close-dialog').forEach(b=>b.onclick=()=>$('#new-dialog').close());$('.close-inspector').onclick=()=>$('#inspector').close();
$('#experiment-form').onsubmit=async e=>{e.preventDefault();const button=e.submitter;button.disabled=true;try{await start({kind:'benchmark',profile:$('#study-profile').value,threads:+$('#study-threads').value,repeats:+$('#study-repeats').value,kv_budget_mib:+$('#study-budget').value});$('#new-dialog').close();location.hash='experiments'}catch(e){notice(e.message)}finally{button.disabled=false}};
$('#run-model').onclick=async()=>{const b=$('#run-model');b.disabled=true;try{await start({kind:'model',threads:+$('#model-threads').value,new_tokens:+$('#model-tokens').value,prompt:$('#prompt').value})}catch(e){notice(e.message);b.disabled=false}};
$('#cancel-job').onclick=async()=>{try{await api(`/api/runs/${state.active}/cancel`,{method:'POST'});notice('Cancellation requested. Waiting for worker cleanup.')}catch(e){notice(e.message)}};
$('#refresh').onclick=()=>refresh().catch(e=>notice(e.message));
$('#run-select').onchange=e=>loadRun(e.target.value).catch(e=>notice(e.message));
$$('.segmented button').forEach(b=>b.onclick=()=>{state.kind=b.dataset.kind;state.selected=null;$$('.segmented button').forEach(x=>x.classList.toggle('active',x===b));refresh().catch(e=>notice(e.message))});
$('#phase-filter').onchange=renderTable;$('#plan-filter').onchange=renderTable;$('#log-scale').onchange=renderChart;
$$('[data-sort]').forEach(b=>b.onclick=()=>{state.ascending=state.sort===b.dataset.sort?!state.ascending:true;state.sort=b.dataset.sort;renderTable()});
$('#export').onclick=()=>{if(state.selected){const a=document.createElement('a');a.href=`/api/runs/${state.selected}/export`;a.download=state.selected+'.json';a.click()}};
window.addEventListener('hashchange',view);
const papers=[['hgca','HGCA: Hybrid GPU-CPU Attention for Long Context LLM Inference','2025 · arXiv · 14 pages','Recent KV entries on the GPU, selected older entries processed on the CPU. The closest starting point for this runtime.'],['beyond','Scaling Attention Beyond GPUs for LLM Inference','HPDC 2026 · 12 pages','BEYOND combines KV management and hybrid attention into an inference runtime. Atlas starts with dense attention to isolate execution costs.'],['nomad','NOMAD: Non-Exclusive Memory Tiering via Transactional Page Migration','OSDI 2024 · 18 pages','Transactional migration and shadow copies motivate measuring the cost of moving data, alongside placement.'],['miku','Architectural and System Implications of CXL-enabled Tiered Memory','2025 preprint · 14 pages','MIKU studies interference between CXL requests and local DDR. This PC does not supply a CXL validation environment.'],['dsa','DSA-2LM: A CPU-Free Tiered Memory Architecture with Intel DSA','USENIX ATC 2025 · 10 pages','Accelerated page migration highlights how copying overhead changes the value of tiering decisions.']];
$('#paper-list').innerHTML=papers.map(([id,title,meta,description],i)=>`<article class="paper"><span class="paper-number">0${i+1}</span><div><h2><a href="/papers/${id}" target="_blank" rel="noopener">${esc(title)}</a></h2><p>${esc(description)}</p><small>${esc(meta)}</small></div><a href="/papers/${id}" target="_blank" rel="noopener" aria-label="Open ${esc(title)} PDF">↗</a></article>`).join('');
view();
await Promise.all([refresh().catch(e=>notice(`Could not connect to the local workspace: ${e.message}`)),api('/api/architecture').then(d=>$('#architecture-text').textContent=d.markdown).catch(e=>notice(e.message))]);
setInterval(()=>{if(state.active)refresh().catch(e=>notice(`Connection interrupted: ${e.message}`))},1500);

window.addEventListener('resize',()=>{if(state.doc)renderChart()});
