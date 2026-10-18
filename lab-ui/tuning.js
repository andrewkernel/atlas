const $=s=>document.querySelector(s);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(x,d=1)=>x==null?'—':Number(x).toLocaleString(undefined,{maximumFractionDigits:d});
const terminal=new Set(['completed','failed','cancelled','timed_out','resource_limited','interrupted','abstained','unknown_owner']);
const words=x=>String(x||'').replaceAll('_',' ');

export function initTuning(api){
  let current=null, overview=null, selected='', inference=null, loading=false;
  const feedback=text=>{const el=$('#tune-feedback');el.textContent=text;el.hidden=!text};
  async function refresh(){
    if(loading)return;loading=true;
    try{
      overview=await api('/api/tuning/state');
      const campaigns=[...overview.campaigns].sort((a,b)=>(b.created_at||'').localeCompare(a.created_at||''));
      if(!campaigns.some(c=>c.id===selected))selected=campaigns[0]?.id||'';
      $('#tune-campaign').innerHTML=campaigns.length?campaigns.map(c=>`<option value="${esc(c.id)}">${esc(c.id)} · ${esc(words(c.status))} · ${num(c.settings?.context,0)} tokens</option>`).join(''):'<option value="">No campaigns yet</option>';
      $('#tune-campaign').value=selected;
      $('#tune-readiness').textContent=overview.readiness.model&&overview.readiness.corpus?'Pinned model and public text are available.':'Local model or public text is missing. Follow the setup guide.';
      $('#tune-start').disabled=!!overview.active_job||!overview.readiness.model||!overview.readiness.corpus;
      $('#tune-generate').disabled=!!overview.active_job;
      $('#tune-cancel').hidden=!overview.active_job;
      $('#tune-history').textContent=overview.events.length?JSON.stringify(overview.events,null,2):'No changes recorded.';
      if(selected){current=await api(`/api/tuning/campaigns/${selected}`);render()}
      if(inference)await pollInference();
    }catch(e){feedback(e.message)}finally{loading=false}
  }
  function render(){
    const {status:s,protocol:p,nomination:n,verified:v,execution:e,active_state:a}=current;
    $('#tune-status').textContent=`${words(s.status)}${s.progress?` · ${s.progress.done} / ${s.progress.total} trials currently registered · ${s.progress.message}`:''}${s.error?` · ${s.error}`:''}`;
    $('#tune-progress').hidden=terminal.has(s.status);$('#tune-progress').max=s.progress?.total||1;$('#tune-progress').value=s.progress?.done||0;
    $('#tune-export').hidden=s.status!=='completed';$('#tune-export').href=`/api/tuning/campaigns/${selected}/export`;
    const d=v?.decision;
    $('#tune-decision').innerHTML=d?`<h3>${esc(words(d.outcome))}</h3><p>${esc(d.reason)}</p><p class="fine-print">Total tuning cost: ${num(s.tuning_cost_s)} s. ${s.speed_break_even_requests?`Generation-only break-even: ${num(s.speed_break_even_requests,0)} compatible requests.`:'No measured speed break-even.'} Warm generation timings exclude startup and checking. ${esc(s.activation_conflict||'')}</p>`:n?`<p>Nominee: ${esc(n.profile||'none')}. ${esc(n.reason)}.</p>`:'<p class="fine-print">A result can be slower, inconclusive, or leave settings unchanged. No change is applied before full verification.</p>';
    const saved=s.profile&&overview.profiles.find(x=>x.id===s.profile);
    const active=overview.profiles.find(x=>x.id===a?.profile);
    const activeName=active?.execution.id||(p?.settings.goal==='budget'?'None (request will abstain)':'Default native');
    $('#tune-profile').innerHTML=p?`<p><strong>Active setting:</strong> ${esc(activeName)}</p>${saved?`<p class="fine-print">Saved: ${esc(saved.execution.id)} · ${esc(saved.id)}. ${current.source_compatible?'Changes affect compatible future Atlas requests.':'Historical source differs from the current runtime; a fresh campaign is required.'}</p>`:''}<div class="tune-actions">${saved?`<button id="tune-apply" class="button small" ${a?.profile===s.profile||!current.source_compatible?'disabled':''}>Apply saved setting</button>`:''}<button id="tune-revert" class="button small" ${!a?.revision?'disabled':''}>Revert last change</button></div>`:'';
    if(saved){
      $('#tune-apply').onclick=()=>change(`/api/tuning/profiles/${s.profile}/apply`,a,'Saved setting applied.');
    }
    if(p)$('#tune-revert').onclick=()=>change(`/api/tuning/domains/${current.domain}/revert`,a,'Previous setting restored.');
    const rows=v?.calibration||n?.calibration;
    $('#tune-measurements').hidden=!rows;
    if(rows)$('#tune-rows').innerHTML=rows.map(r=>`<tr><td>${esc(r.profile.id)}</td><td>${num(r.generation_geomean_ms)} ms</td><td>${num(r.peak_allocated_mib)} MiB</td><td>${r.passed_windows} / ${r.planned_windows} ${r.numerical?'within bounds':'blocked or failed'}</td><td>${r.eligible?'Yes':'No'}</td></tr>`).join('');
    $('#tune-pairs').innerHTML=d?.pair_ratios?.length?`<div class="section-heading table-heading"><h2>Separate confirmation pairs</h2><span>Above 1 is faster</span></div><div class="table-wrap"><table><thead><tr><th>Input</th><th>Repeat</th><th>Current</th><th>Nominee</th><th>Ratio</th></tr></thead><tbody>${d.pair_ratios.map(r=>`<tr><td>${esc(r.input)}</td><td>${r.repetition+1}</td><td>${num(r.baseline_ms)} ms</td><td>${num(r.candidate_ms)} ms</td><td>${num(r.ratio,3)}×</td></tr>`).join('')}</tbody></table></div>`:'';
    $('#tune-generation').hidden=!p||s.status!=='completed';
    $('#tune-evidence').textContent=JSON.stringify({protocol:p,nomination:n,execution:e,verification:v},null,2);
  }
  async function change(path,expected,message){
    try{await api(path,{method:'POST',body:JSON.stringify(expected)});feedback(message);await refresh()}catch(e){feedback(e.message)}
  }
  async function pollInference(){
    const r=await api(`/api/tuning/inference/${inference}`);
    $('#tune-inference-state').textContent=`${words(r.status)} · ${r.execution?.id||'Checking compatibility'} · Current request check: ${r.request_numerical_check||'unknown'}${r.error?` · ${r.error}`:''}`;
    $('#tune-text').textContent=r.text||'';
    if(terminal.has(r.status)){
      $('#tune-inference-detail').hidden=false;
      $('#tune-inference-detail').textContent=JSON.stringify({applied_profile:r.applied_profile,native_fallback:r.native_fallback,fallback_reason:r.fallback_reason,profile_evidence:r.profile_numerical_evidence,request_check:r.request_numerical_check,check_scope:r.check_scope,generation_ms:r.measurement?.generation_ms,model_load_s:r.model_load_s,warmup_ms:r.warmup?.generation_ms,native_check_s:r.native_check_s,total_request_s:r.total_request_s},null,2);
      inference=null;
    }
  }
  $('#tune-goal').onchange=()=>{$('#tune-budget-label').hidden=$('#tune-goal').value!=='budget'};
  $('#tune-custom').onchange=()=>{$('#tune-prompt-label').hidden=!$('#tune-custom').checked};
  $('#tune-campaign').onchange=async e=>{selected=e.target.value;await refresh()};
  $('#tune-refresh').onclick=()=>refresh();
  $('#tune-form').onsubmit=async e=>{
    e.preventDefault();$('#tune-start').disabled=true;feedback('');
    try{
      const goal=$('#tune-goal').value;
      const r=await api('/api/tuning/start',{method:'POST',body:JSON.stringify({context:+$('#tune-context').value,dtype:$('#tune-dtype').value,generation_tokens:+$('#tune-output').value,goal,budget_mib:goal==='budget'?+$('#tune-budget').value:null,auto_apply:$('#tune-auto').checked,max_wall_s:14400})});
      selected=r.id;await refresh();
    }catch(err){feedback(err.message);$('#tune-start').disabled=false}
  };
  $('#tune-cancel').onclick=async()=>{
    try{await api(`/api/tuning/jobs/${overview.active_job}/cancel`,{method:'POST'});feedback('Cancellation requested. Waiting for owned processes to exit.');await refresh()}catch(e){feedback(e.message)}
  };
  $('#tune-generate-form').onsubmit=async e=>{
    e.preventDefault();$('#tune-generate').disabled=true;
    try{
      const r=await api('/api/tuning/generate',{method:'POST',body:JSON.stringify({campaign:selected,prompt:$('#tune-custom').checked?$('#tune-prompt').value:null,check_native:$('#tune-native-check').checked})});
      inference=r.id;$('#tune-text').textContent='';$('#tune-inference-detail').hidden=true;await refresh();
    }catch(err){feedback(err.message);$('#tune-generate').disabled=false}
  };
  window.addEventListener('hashchange',()=>{if(location.hash==='#tuning')refresh()});
  setInterval(()=>{if(location.hash==='#tuning'||inference)refresh()},2500);
  if(location.hash==='#tuning')refresh();
}
