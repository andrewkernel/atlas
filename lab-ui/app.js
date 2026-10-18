const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors={gpu_dense:'#428e8c',resident:'#326b49',native:'#326b49',copy:'#bc8b48',cpu:'#7c8ca0',hybrid25:'#c2b4d1',hybrid50:'#9c85b5',hybrid75:'#68517f',adaptive:'#b06852'};
const labels={gpu_dense:'GPU dense reference',resident:'Native GPU',copy:'Transfer to GPU',cpu:'CPU attention',hybrid25:'Hybrid · 25% GPU',hybrid50:'Hybrid · 50% GPU',hybrid75:'Hybrid · 75% GPU',native:'Native model SDPA',adaptive:'Atlas adaptive'};
const studyLabels={...labels,hybrid50:'Hybrid · 256-token GPU window'};
Object.assign(colors,{shift:'#bc8b48',ring:'#326b49'});
Object.assign(labels,{shift:'Shift cache',ring:'Ring cache'});
const displayLabel=(r,doc=state.doc)=>doc?.config?.kind==='research'&&['repeated','pressure','pilot'].includes(doc.config.suite)&&r.plan==='hybrid50'?`Hybrid · ${doc.config.window||256}-token GPU window`:r.label;
const state={token:'',runs:[],doc:null,selected:null,kind:'benchmark',view:'experiments',hidden:new Set(),sort:'context',ascending:true,active:null,referenceRuns:new Set()};
const planLabel=p=>{const row=state.doc?.rows?.find(r=>r.plan===p);return row?displayLabel(row):labels[p]||p};
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
  const [session,runs,reference]=await Promise.all([api('/api/session'),api('/api/runs'),api('/data/reference-history.json').catch(()=>({runs:[]}))]);
  state.referenceRuns=new Set(reference.runs||[]);
  state.token=session.token;state.runs=runs;state.active=session.active_run;
  $('#run-model').disabled=!session.model_ready||!!state.active;
  $('#new-run').disabled=!!state.active;
  const filtered=runs.filter(r=>r.config?.kind===state.kind);
  const selection=prefer||state.selected;
  state.selected=filtered.some(r=>r.id===selection)?selection:(filtered.find(r=>r.status==='completed')||filtered[0])?.id;
  $('#run-select').innerHTML=filtered.map(r=>`<option value="${esc(r.id)}">${state.referenceRuns.has(r.id)?'Reference study':'Local run'} · ${esc(date(r.created_at))} · ${esc(r.config.kind==='long-validation'?`${r.config.case_id} / ${r.config.context.toLocaleString()} tokens`:r.config.suite||r.config.profile||'model')} · ${esc(r.id.slice(-6))} · ${esc(r.status)}</option>`).join('');
  $('#run-select').value=state.selected||'';
  $('#export').hidden=!state.selected;
  $('#active-job').hidden=!state.active;
  const active=runs.find(r=>r.id===state.active);
  if(active){$('#job-title').textContent=active.config.kind==='long-validation'?'Long-context validation':active.config.kind==='model'?'Model comparison':'Attention sweep';$('#job-detail').textContent=active.progress?.message||'Preparing';$('#job-progress').max=active.progress?.total||1;$('#job-progress').value=active.progress?.done||0}
  if(state.selected){await loadRun(state.selected)}else{state.doc=null;$('#results').hidden=true;$('#empty').hidden=false}
  const modelRun=runs.find(r=>r.config?.kind==='model'&&r.status==='completed');
  if(modelRun){const modelDoc=state.doc?.id===modelRun.id?state.doc:await api(`/api/runs/${modelRun.id}`);renderModel(modelDoc)}
}

async function loadRun(id){
  const previous=state.doc?.id;state.selected=id;state.doc=await api(`/api/runs/${id}`);
  if(previous!==id){clearNotice();state.hidden.clear();$('#phase-filter').value='all';$('#plan-filter').value='all'}
  const doc=state.doc,rows=doc.rows||[];
  $('#export').href=`/api/runs/${encodeURIComponent(id)}/export`;$('#export').download=id+'.json';
  $('#empty').hidden=rows.length>0||state.active===doc.id;$('#results').hidden=!rows.length;
  if(doc.error)notice(`${doc.status}: ${doc.error}`);
  if(!rows.length)return;
  const timeKey=doc.config.kind==='long-validation'?'generation_ms':'median_ms';
  const eligible=rows.filter(r=>r.correctness?.passed===true&&r.eligibility?.eligible!==false&&r[timeKey]!=null);
  const fastest=eligible.length?eligible.reduce((a,b)=>a[timeKey]<b[timeKey]?a:b):null;
  $('#fastest-label').textContent=doc.config.kind==='long-validation'?'LOWEST ELIGIBLE GENERATION TIME':'LOWEST ELIGIBLE LATENCY';
  $('#fastest').textContent=(fastest?displayLabel(fastest):null)||'No eligible result';$('#fastest-detail').textContent=fastest?`${fmt(fastest[timeKey])} ms at ${fastest.context.toLocaleString()} tokens`:'Failed and unknown checks block recommendation';
  const passed=rows.filter(r=>r.correctness?.passed).length;
  $('#checks').textContent=`${passed} / ${rows.length} passed`;
  $('#checks').classList.toggle('numerics',passed===rows.length);
  $('#scope').textContent=['research','long-validation'].includes(doc.config.kind)&&doc.model?'Qwen3 · 0.6B':doc.config.kind==='model'?'GPT-2 · 124M':`${new Set(rows.map(r=>r.context)).size} context lengths`;
  $('#scope-detail').textContent=doc.config.kind==='long-validation'?`${doc.config.context.toLocaleString()} tokens · FP32 · ${doc.config.cohort}`:doc.config.kind==='research'?`${doc.config.dtype||'float32'} · ${doc.config.suite} · block ${doc.config.block??'pilot'}`:doc.config.kind==='model'?`${doc.protocol?.decode_steps} decoding steps · float32`:`12 heads × 64 dimensions · float32`;
  $('#run-state').outerHTML=`<span id="run-state" class="pill ${doc.status==='completed'?'':doc.status==='failed'?'fail':'warn'}">${esc(doc.status)}</span>`;
  $('#run-date').textContent=date(doc.created_at);
  $('#scope-detail').textContent+=state.referenceRuns.has(doc.id)?' · reference record':' · local run';
  const ids=[...new Set(rows.map(r=>r.plan))];
  const prior=$('#plan-filter').value;
  $('#plan-filter').innerHTML='<option value="all">All execution plans</option>'+ids.map(p=>`<option value="${p}">${esc(planLabel(p))}</option>`).join('');
  $('#plan-filter').value=ids.includes(prior)?prior:'all';
  $('#legend').innerHTML=ids.map(p=>`<button data-series="${p}" aria-pressed="${!state.hidden.has(p)}" class="${state.hidden.has(p)?'off':''}"><svg width="8" height="8" aria-hidden="true"><circle cx="4" cy="4" r="3.5" fill="${colors[p]||'#777'}"/></svg>${esc(planLabel(p))}</button>`).join('');
  $$('#legend button').forEach(b=>b.onclick=()=>{const id=b.dataset.series;state.hidden.has(id)?state.hidden.delete(id):state.hidden.add(id);b.classList.toggle('off',state.hidden.has(id));b.setAttribute('aria-pressed',!state.hidden.has(id));renderChart()});
  $('#evidence').innerHTML=`<p>${(doc.limitations||[]).map(esc).join('<br>')}</p><pre>${esc(JSON.stringify({environment:doc.environment,protocol:doc.protocol,calibration_elapsed_s:doc.calibration_elapsed_s,policy_source:doc.policy_source},null,2))}</pre>`;
  renderPolicy();renderTable();renderChart();
}

function renderPolicy(){
  const doc=state.doc,evaluation=doc.evaluation||[];
  $('#policy-panel').hidden=!evaluation.length;
  if(!evaluation.length)return;
  if(doc.config.kind==='research'){$('#policy-panel').innerHTML='<strong>Calibration observations</strong><p>These raw decisions use the historical persistent-cache budget. The Research study page includes the declared correction for temporary GPU KV copies and precision promotion, plus rule/fixed baselines and calibration cost. Layer calibration does not certify model inference.</p>';return}
  $('#policy-panel').innerHTML=`<strong>Adaptive selection / held-out lengths</strong> <span>· calibrated in ${fmt(doc.calibration_elapsed_s,1)} s · fixed comparator: ${esc(labels[doc.fixed_plan])}</span><div class="decisions">${evaluation.map(e=>`<div class="decision"><span>${e.context.toLocaleString()} TOKENS</span><b>${esc(labels[e.selected])}</b> · ${fmt(e.selected_ms)} ms<br><small>${fmt(e.regret_percent,1)}% slower than measured optimum</small></div>`).join('')}</div>`;
}

function renderTable(){
  if(!state.doc)return;
  const long=state.doc.config.kind==='long-validation';
  $('#time-sort').textContent=long?'Generation total ↕':'Median ↕';$('#time-sort').dataset.sort=long?'generation_ms':'median_ms';
  $('#secondary-time-heading').textContent=long?'Decode median':'p90';$('#memory-heading').textContent=long?'GPU allocated peak':'GPU KV';
  if(long&&state.sort==='median_ms')state.sort='generation_ms';else if(!long&&state.sort==='generation_ms')state.sort='median_ms';
  const phase=$('#phase-filter').value,plan=$('#plan-filter').value;
  const rows=state.doc.rows.map((r,i)=>({...r,index:i})).filter(r=>(phase==='all'||r.phase===phase)&&(plan==='all'||r.plan===plan)).sort((a,b)=>(a[state.sort]-b[state.sort])*(state.ascending?1:-1));
  $('#row-count').textContent=rows.length;
  $('#measurements').innerHTML=rows.map(r=>`<tr data-row="${r.index}"><td>${r.context.toLocaleString()}</td><td><span class="plan-name"><svg width="8" height="8" aria-hidden="true"><circle cx="4" cy="4" r="3.5" fill="${colors[r.plan]||'#777'}"/></svg>${esc(displayLabel(r))}</span></td><td class="phase">${r.phase==='validation'?'Held out':esc(r.phase)}</td><td>${fmt(long?r.generation_ms:r.median_ms)} <span class="muted">ms</span></td><td>${fmt(long?r.median_ms:r.p90_ms)} <span class="muted">ms</span></td><td>${fmt(long?r.allocated_peak_mib:r.gpu_kv_mib,1)} <span class="muted">MiB</span></td><td class="numerics ${r.correctness?.passed?'':'fail'}">${r.correctness?.passed===true?'✓ Within bounds':r.correctness?.passed===false?'× Outside bounds':'? Unknown'}</td><td><button class="inspect-row" aria-label="Inspect ${esc(displayLabel(r))}, ${r.context} tokens">↗</button></td></tr>`).join('');
  $$('#measurements tr').forEach(tr=>{tr.onclick=()=>inspect(state.doc.rows[Number(tr.dataset.row)]);tr.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();tr.click()}}});
}

function renderChart(){
  if(!state.doc)return;
  const all=state.doc.rows,rows=all.filter(r=>!state.hidden.has(r.plan)&&r.median_ms!=null);
  const model=['model','long-validation'].includes(state.doc.config.kind)||['repeated','pressure','pilot'].includes(state.doc.config.suite);
  $('#chart-title').textContent=model?'Full-model decode latency':'Latency across context lengths';
  $('#chart-subtitle').textContent=state.doc.config.kind==='long-validation'?`Median of ${state.doc.config.generation_tokens-1} decode forwards from one generation window · total generation in the table`:'Median synchronized wall time · lower is faster';
  $('#chart-note').textContent=state.doc.config.suite==='cache'?'Complete append, eviction, transfers and attention · setup separate':['research','long-validation'].includes(state.doc.config.kind)&&model?'Free greedy timing · separate corpus validation · cache updates included':model?'Teacher-forced native token history · cache updates included':'Calibration points ● · held-out lengths ◇';
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
  if(model){const step=cw/rows.length;rows.forEach((r,i)=>{const px=L+step*i+step*.2,py=y(r.median_ms);svg+=`<rect x="${px}" y="${py}" width="${step*.6}" height="${T+ch-py}" rx="3" fill="${colors[r.plan]||'#777'}"><title>${esc(displayLabel(r))}: ${fmt(r.median_ms)} ms</title></rect><text x="${px+step*.3}" text-anchor="end" transform="rotate(-30 ${px+step*.3} ${H-37})" y="${H-37}">${esc(r.plan)}</text>`})}
  else{
    const lengths=[...new Set(all.map(r=>r.context))].sort((a,b)=>a-b),xmin=Math.log2(lengths[0]),xmax=Math.log2(lengths.at(-1));
    const x=v=>L+(Math.log2(v)-xmin)/(xmax-xmin||1)*cw;
    lengths.forEach(n=>{svg+=`<text x="${x(n)}" y="${H-12}" text-anchor="middle">${n>=1024?`${n/1024}k`:n}</text>`});
    [...new Set(rows.map(r=>r.plan))].forEach(p=>{const series=rows.filter(r=>r.plan===p).sort((a,b)=>a.context-b.context);svg+=`<path class="series" stroke="${colors[p]}" d="${series.map((r,i)=>`${i?'L':'M'}${x(r.context)},${y(r.median_ms)}`).join(' ')}"/>`;series.forEach(r=>{const px=x(r.context),py=y(r.median_ms),title=`<title>${esc(displayLabel(r))}, ${r.context} tokens: ${fmt(r.median_ms)} ms</title>`;svg+=r.phase==='validation'?`<path d="M${px},${py-4}l4,4 -4,4 -4,-4Z" fill="white" stroke="${colors[p]}" stroke-width="1.8">${title}</path>`:`<circle cx="${px}" cy="${py}" r="3" fill="${colors[p]}">${title}</circle>`})});
  }
  $('#chart').innerHTML=svg+'</svg>';
}

function inspect(row){
  $('#inspector-title').textContent=`${displayLabel(row)} / ${row.context.toLocaleString()} tokens`;
  const samples=row.samples_ms||[],max=Math.max(...samples,1e-9);
  $('#inspector-content').innerHTML=`<div class="detail-grid"><div><span>MEDIAN</span><strong>${fmt(row.median_ms)} ms</strong></div><div><span>P90</span><strong>${fmt(row.p90_ms)} ms</strong></div><div><span>ALLOCATED PEAK</span><strong>${fmt(row.allocated_peak_mib,1)} MiB</strong></div></div><div class="inspector-section-title">RAW SYNCHRONIZED SAMPLES / ${samples.length} OBSERVATIONS</div><svg viewBox="0 0 550 100" width="100%" height="100" role="img" aria-label="Raw latency samples">${samples.map((v,i)=>`<rect x="${i*550/samples.length}" y="${100-v/max*92}" width="${Math.max(1,550/samples.length-3)}" height="${v/max*92}" fill="#71956a"><title>Sample ${i+1}: ${fmt(v,4)} ms</title></rect>`).join('')}</svg><div class="inspector-section-title">NUMERICAL CHECK</div><pre class="detail-json">${esc(JSON.stringify(row.correctness,null,2))}</pre><details><summary>All measurement fields</summary><pre class="detail-json">${esc(JSON.stringify(row,null,2))}</pre></details>`;
  $('#inspector').showModal();
}

function renderModel(doc){
  if(doc.continuation)$('#continuation').textContent=doc.continuation.text;
  $('#model-comparison').innerHTML=(doc.rows||[]).map(r=>`<div class="mini-plan"><span>${esc(displayLabel(r,doc))}</span><span>${fmt(r.tokens_per_s,1)} <small>tok/s</small></span><span class="numerics ${r.correctness?.passed?'':'fail'}">${r.correctness?.passed?'✓':'×'} logits</span></div>`).join('');
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
  state.view=['experiments','study','validation','model','architecture','research'].includes(requested)?requested:'experiments';
  $$('.view').forEach(el=>el.hidden=el.id!==`view-${state.view}`);
  $$('nav a').forEach(a=>{a.classList.toggle('active',a.dataset.view===state.view);a.setAttribute('aria-current',a.dataset.view===state.view?'page':'false')});
  const titles={experiments:['Find the crossover.','Explore where attention belongs: GPU, CPU, or somewhere between.'],model:['Follow every token.','Run a pretrained model through the same execution paths.'],architecture:['Built to be inspected.','The runtime, the evidence, and the decisions connecting them.'],research:['Start with the papers.','Source material, connections, and the boundaries of this study.']};
  titles.study=['Test the claim.','Frozen cases, independent process blocks, and honest boundaries.'];
  titles.validation=['Show the evidence.','Long-context memory tradeoffs, full-model checks, and a path to reproduce them.'];
  $('#page-title').textContent=titles[state.view][0];$('#page-subtitle').textContent=titles[state.view][1];$('#breadcrumb').textContent=state.view==='model'?'Model playground':state.view==='architecture'?'Architecture':state.view==='research'?'Research library':'Attention Lab';
  if(state.view==='study')loadStudy().catch(e=>notice(e.message));
  if(state.view==='validation')loadValidation().catch(e=>notice(e.message));
  window.scrollTo(0,0);
}

let studyData=null;
async function loadStudy(){
  studyData=await api('/api/study');const {protocol,execution,summary}=studyData;
  if(!protocol){$('#study-progress').textContent='No frozen protocol on this machine.';return}
  $('#study-progress').textContent=`${protocol.id} · ${execution?.cases?.length||0} / ${protocol.cases.length} cases recorded · ${summary?.complete?'analysis complete':'study in progress'}. ${summary?`${summary.numerical_failed_rows} failed numerical rows · ${summary.numerical_unknown_rows} unknown rows. `:''}Execution completion does not imply numerical validity.`;
  const prior=$('#research-case').value;
  $('#research-case').innerHTML=protocol.cases.map(c=>`<option value="${esc(c.case_id)}">${esc(c.case_id)} · ${esc(c.suite)} · ${esc(c.dtype||'FP32')} · ${c.context||'layer sweep'} · block ${c.block}${c.methods?.length===1?' · '+esc(c.methods[0]):''}</option>`).join('');
  if(prior)$('#research-case').value=prior;
  $('#study-protocol').textContent=JSON.stringify({protocol,execution,cache_ablation:summary?.cache_ablation,selector_comparisons:summary?.selectors,backend:summary?.backend_operations},null,2);
  $('#study-validity').innerHTML=`<p>${esc(summary?.uncertainty||protocol.analysis.small_n_warning)}</p><ul>${(summary?.bias_limits||protocol.prohibited_claims).map(x=>`<li>${esc(x)}</li>`).join('')}</ul><pre>${esc(JSON.stringify(summary?.diagnosis||{},null,2))}</pre>`;
  $('#pressure-rows').innerHTML=(summary?.pressure||[]).map(r=>`<tr><td>${r.context.toLocaleString()}</td><td>${r.block}</td><td>${esc(r.label||r.plan)}</td><td>${pill(r.outcome)}</td><td>${fmt(r.generation_ms,0)} ms</td><td>${fmt(r.allocated_peak_mib,0)} MiB</td><td>${fmt(r.board_observed_peak_mib,0)} MiB</td></tr>`).join('')||'<tr><td colspan="7">Pressure records will appear after analysis. OOM and timeout outcomes remain in the denominator.</td></tr>';
  $('#selector-rows').innerHTML=(summary?.selector_comparisons||[]).map(r=>`<tr><td>${esc(r.dtype)}</td><td>${r.budget_mib} MiB</td><td>${fmt(r.vs_rule?.estimate,3)}×</td><td>${fmt(r.vs_fixed?.estimate,3)}×</td><td>${fmt(r.calibration_s?.estimate,1)} s</td></tr>`).join('')||'<tr><td colspan="5">Calibration comparisons are awaiting analysis.</td></tr>';
  renderStudy();
}
function renderStudy(){
  const summary=studyData?.summary,rows=(summary?.model_comparisons||[]).filter(r=>r.dtype===$('#research-dtype').value&&r.context===+$('#research-context').value);
  $('#study-comparisons').innerHTML=rows.length?`<div class="table-wrap"><table><thead><tr><th>Candidate</th><th>Generation ratio</th><th>95% interval</th><th>Eligibility</th><th>NLL delta</th></tr></thead><tbody>${rows.map(r=>{const n=r.generation_speedup?.n_process_blocks||0;return `<tr><td>${esc(studyLabels[r.plan])}</td><td>${fmt(r.generation_speedup?.estimate,3)}×</td><td>${r.generation_speedup?.low!=null?`${fmt(r.generation_speedup.low,3)}–${fmt(r.generation_speedup.high,3)}`:'Pending 3 blocks'}</td><td><span class="pill ${r.eligible_blocks<n?'fail':n<3?'warn':''}">${r.eligible_blocks}/${n} passed${n<3?' · 3 planned':''}</span></td><td>${fmt(r.nll_delta?.estimate,6)}</td></tr>`}).join('')}</tbody></table></div>`:'<p>Full generation comparisons will appear after the planned runs and analysis finish.</p>';
  if(!rows.length){$('#study-chart').innerHTML='';return}
  const max=Math.max(1.1,...rows.map(r=>r.generation_speedup?.high||0))*1.05,W=680,H=150,L=165,R=35,cw=W-L-R,x=v=>L+v/max*cw;
  $('#study-chart').innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Full generation ratios and process-block uncertainty"><line x1="${x(1)}" x2="${x(1)}" y1="5" y2="125" stroke="#9caa98" stroke-dasharray="4 4"/><text x="${x(1)}" y="145" text-anchor="middle">native 1×</text>${rows.map((r,i)=>{const c=r.generation_speedup,y=25+i*40;if(!c)return '';return `<text x="150" y="${y+4}" text-anchor="end">${esc(r.plan==='hybrid50'?'hybrid / 256 GPU':r.plan)}</text>${c.low!=null?`<line x1="${x(c.low)}" x2="${x(c.high)}" y1="${y}" y2="${y}" stroke="${colors[r.plan]}" stroke-width="3"/>`:''}<circle cx="${x(c.estimate)}" cy="${y}" r="5" fill="${r.all_blocks_eligible?colors[r.plan]:'#b86857'}"><title>${esc(r.plan)} ${fmt(c.estimate,3)}×; ${r.eligible_blocks}/3 eligible blocks</title></circle>`}).join('')}</svg>`;
}
$('#research-dtype').onchange=renderStudy;$('#research-context').onchange=renderStudy;
$('#repeat-case').onclick=async()=>{const b=$('#repeat-case');b.disabled=true;try{const result=await api('/api/study/repeat',{method:'POST',body:JSON.stringify({case_id:$('#research-case').value})});$('#research-job').textContent=`Started ${result.id}. Inspect progress and raw evidence in Research runs.`;state.kind='research';$$('.segmented button').forEach(x=>x.classList.toggle('active',x.dataset.kind==='research'));await refresh(result.id)}catch(e){$('#research-job').textContent=e.message}finally{b.disabled=false}};

function numericalState(arm){return !arm?'Pending':arm.status==='oom'?'Out of memory':arm.passed===true?(arm.eligible?'Within bounds':'Within bounds · blocked'):arm.passed===false?'Outside bounds':'Unknown'}
async function loadValidation(){
  const [{protocol,execution,summary},reference]=await Promise.all([api('/api/validation'),api('/data/reference-history.json').catch(()=>null)]);
  const unavailable=new Set(reference?.unavailable_runs||[]);
  if(!protocol){$('#validation-progress').textContent='The long-context protocol has not been frozen on this machine.';$('#validation-repeat').disabled=true;return}
  const entries=execution?.cases||[],terminal=entries.filter(e=>e.status!=='running'),pending=entries.find(e=>e.status==='running');
  let activeMessage='';
  if(pending){try{const live=await api(`/api/runs/${pending.run_id}`);activeMessage=` ${pending.case_id}: ${live.progress?.message||live.status}.`}catch{activeMessage=' Worker progress is unavailable; the recorded attempt remains pending.'}}
  $('#validation-progress').textContent=`${terminal.length} / ${protocol.cases.length} registered outcomes collected.${activeMessage} ${summary?`${summary.recorded_cases} outcomes independently checked against saved arrays.`:'Full-array verification is pending.'} Execution completion alone does not qualify a result.`;
  $('#validation-export').hidden=!summary;
  if(summary?.all_cases_recorded){
    try{
      const kit=await api('/downloads/manifest.json');
      const ready=kit.all_cases_recorded&&kit.protocol_sha256===summary.protocol_sha256&&kit.recorded_cases===summary.planned_cases;
      $('#validation-kit').hidden=!ready;
      $('#validation-kit-status').textContent=ready?`Complete local kit · ${fmt(kit.bytes/2**20,0)} MiB · ${kit.files} files · ZIP SHA-256: ${kit.sha256}. External reproduction has not been established.`:'The researcher bundle does not match this study.';
    }catch{$('#validation-kit').hidden=true;$('#validation-kit-status').textContent='Use the included reproduction guide to verify the local evidence or run a fresh replication. A download appears here when an archive is generated.'}
  }
  const prior=$('#validation-case').value;
  $('#validation-case').innerHTML=protocol.cases.map(c=>`<option value="${esc(c.case_id)}">${esc(c.case_id)} · ${c.context.toLocaleString()} tokens · ${c.book_id?'PG-19 book '+esc(c.book_id):'earlier WikiText input '+c.block}</option>`).join('');
  if(prior)$('#validation-case').value=prior;
  $('#validation-repeat').disabled=!!pending||!!state.active;
  $('#validation-rows').innerHTML=protocol.cases.map(c=>{
    const entry=entries.find(e=>e.case_id===c.case_id),verified=summary?.cases.find(r=>r.case_id===c.case_id),arm=p=>verified?.arms.find(a=>a.plan===p);
    return `<tr><td>${esc(c.case_id)}</td><td>${c.book_id?'PG-19 / '+esc(c.book_id):'WikiText replay / '+c.block}</td><td>${c.context.toLocaleString()}</td><td>${entry?pill(entry.status):'Not run'}</td>${['native','resident','hybrid50'].map(p=>`<td class="${arm(p)?.passed===false?'fail':''}">${numericalState(arm(p))}</td>`).join('')}<td>${entry?(unavailable.has(entry.run_id)?'Full evidence kit required':`<a href="/api/runs/${esc(entry.run_id)}/export" target="_blank" rel="noopener">Raw JSON ↗</a>`):'—'}</td></tr>`;
  }).join('');
  $('#validation-comparisons').innerHTML=summary?`<div class="table-wrap"><table><thead><tr><th>Input cohort</th><th>Context</th><th>Qualified pairs</th><th>Generation ratio</th><th>Allocator reduction</th></tr></thead><tbody>${summary.comparisons.map(c=>`<tr><td>${c.cohort==='v2-replay'?'Earlier inputs · retrospective':'Six separate books'}</td><td>${c.context.toLocaleString()}</td><td>${c.eligible_pairs} / ${c.planned}</td><td>${fmt(c.eligible_ratio_median,3)}×</td><td>${c.eligible_allocator_reduction_percent.length?fmt(c.eligible_allocator_reduction_percent.reduce((a,b)=>a+b,0)/c.eligible_allocator_reduction_percent.length,1)+'% mean':'—'}</td></tr>`).join('')}</tbody></table></div><p class="fine-print">These are descriptive paired measurements. A partial set of qualifying pairs cannot establish a benefit across all registered inputs. Allocator measurements do not establish physical GPU residency or an out-of-memory rescue.</p>`:'<p>No correctness-qualified long-context claim yet. The earlier 32k/40k memory measurements remain feasibility evidence while this study runs.</p>';
  $('#validation-protocol').textContent=JSON.stringify({protocol,execution,verification:summary?.verification},null,2);
  try{
    const isolated=await api('/data/isolation-summary.json');
    if(isolated.all_cases_recorded&&isolated.base_protocol_sha256===summary?.protocol_sha256){
      $('#isolation-results').innerHTML=`<div class="table-wrap"><table><thead><tr><th>Context</th><th>Arm</th><th>Outcome</th><th>Numerics</th><th>Generation</th><th>Host pool after cleanup</th></tr></thead><tbody>${isolated.cases.map(r=>{const host=r.host_allocator?.after_generation_cleanup?.['allocated_bytes.current'];return `<tr><td>${r.context.toLocaleString()}</td><td>${esc(r.method==='hybrid50'?'Hybrid / 256 GPU tokens':r.method)}</td><td>${esc(r.arm_status)}</td><td>${r.passed===true?(r.eligible?'Within bounds':'Within bounds · blocked'):r.passed===false?'Outside bounds':'Unknown'}</td><td>${fmt(r.generation_ms==null?null:r.generation_ms/1000,1)} s</td><td>${fmt(host==null?null:host/2**30,2)} GiB</td></tr>`}).join('')}</tbody></table></div><p class="fine-print">${esc(isolated.verification)} The case was chosen after its original failures; this is a post-hoc diagnostic. Host pool means allocator-owned pinned memory, including cached blocks; it is not process RSS or an exclusive physical-memory measurement.</p>`;
    }
  }catch{/* Verified follow-up data is published only after its complete analysis. */}
}
$('#validation-repeat').onclick=async()=>{
  const button=$('#validation-repeat');button.disabled=true;
  try{
    const result=await api('/api/validation/repeat',{method:'POST',body:JSON.stringify({case_id:$('#validation-case').value})});
    $('#validation-job').textContent=`Started ${result.id}. Live progress and Cancel appear above. This fresh repeat does not alter the registered study ledger.`;
    state.kind='long-validation';$$('.segmented button').forEach(x=>x.classList.toggle('active',x.dataset.kind===state.kind));
    await refresh(result.id);
  }catch(e){$('#validation-job').textContent=e.message}finally{await loadValidation()}
};

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
$('#export').onclick=e=>{if(!state.selected)e.preventDefault()};
window.addEventListener('hashchange',view);
const papers=[['hgca','HGCA: Hybrid GPU-CPU Attention for Long Context LLM Inference','2025 · arXiv · 14 pages','Recent KV entries on the GPU, selected older entries processed on the CPU. The closest starting point for this runtime.'],['beyond','Scaling Attention Beyond GPUs for LLM Inference','HPDC 2026 · 12 pages','BEYOND combines KV management and hybrid attention into an inference runtime. Atlas starts with dense attention to isolate execution costs.'],['nomad','NOMAD: Non-Exclusive Memory Tiering via Transactional Page Migration','OSDI 2024 · 18 pages','Transactional migration and shadow copies motivate measuring the cost of moving data, alongside placement.'],['miku','Architectural and System Implications of CXL-enabled Tiered Memory','2025 preprint · 14 pages','MIKU studies interference between CXL requests and local DDR. This PC does not supply a CXL validation environment.'],['dsa','DSA-2LM: A CPU-Free Tiered Memory Architecture with Intel DSA','USENIX ATC 2025 · 10 pages','Accelerated page migration highlights how copying overhead changes the value of tiering decisions.']];
const paperSources={hgca:'https://arxiv.org/pdf/2507.03153',beyond:'https://huilucs.github.io/pubs/hpdc2026.pdf',nomad:'https://www.usenix.org/system/files/osdi24-xiang.pdf',miku:'https://arxiv.org/pdf/2503.17864v2',dsa:'https://ranger.uta.edu/~jrao/papers/atc25.pdf'};
$('#paper-list').innerHTML=papers.map(([id,title,meta,description],i)=>`<article class="paper"><span class="paper-number">0${i+1}</span><div><h2><a href="${paperSources[id]}" target="_blank" rel="noopener">${esc(title)}</a></h2><p>${esc(description)}</p><small>${esc(meta)}</small></div><a href="${paperSources[id]}" target="_blank" rel="noopener" aria-label="Open ${esc(title)} PDF">↗</a></article>`).join('');
view();
await Promise.all([refresh().catch(e=>notice(`Could not connect to the local workspace: ${e.message}`)),api('/api/architecture').then(d=>$('#architecture-text').textContent=d.markdown).catch(e=>notice(e.message))]);
setInterval(()=>{if(state.active)refresh().catch(e=>notice(`Connection interrupted: ${e.message}`))},1500);
setInterval(()=>{if(state.view==='study')loadStudy().catch(e=>notice(e.message))},5000);
setInterval(()=>{if(state.view==='validation')loadValidation().catch(e=>notice(e.message))},5000);

window.addEventListener('resize',()=>{if(state.doc)renderChart()});
