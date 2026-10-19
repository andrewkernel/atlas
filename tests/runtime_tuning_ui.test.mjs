// Authored asynchronous UI controls. These contain no model measurements.
import test from 'node:test';
import assert from 'node:assert/strict';
import {initTuning} from '../lab-ui/tuning.js';

const tick=()=>new Promise(resolve=>setImmediate(resolve));
const event={preventDefault(){}};
const record=(id,status='completed')=>({
  status:{id,status,progress:{done:1,total:1,message:id}},
  protocol:{settings:{goal:'fastest',dtype:'float32',context:id==='case-b'?32768:2048,generation_tokens:16}},nomination:null,execution:{cases:[]},
  verified:{decision:{outcome:'keep_current',reason:`Authored ${id}`,pair_ratios:[]},calibration:[]},
  active_state:{profile:null,revision:0},source_compatible:true,domain:`domain-${id}`
});
const overview=(ids=['case-a','case-b'])=>({
  campaigns:ids.map((id,i)=>({id,status:'completed',created_at:String(ids.length-i),settings:{context:id==='case-b'?32768:2048}})),
  active_job:null,active_kind:null,profiles:[],events:[],readiness:{model:true,corpus:true}
});

function harness(t){
  const previous=new Map(['document','window','location','setInterval'].map(k=>[k,
    {exists:Object.hasOwn(globalThis,k),value:globalThis[k]}]));
  const elements=new Map(), requests=[];
  let interval;
  function element(id){
    if(!elements.has(id)){
      let html='';
      elements.set(id,{value:'',checked:false,disabled:false,hidden:false,textContent:'',writes:0,
        setAttribute(name,value){this[name]=name==='disabled'?true:value},
        get innerHTML(){return html},set innerHTML(value){html=value;this.writes++}});
    }
    return elements.get(id);
  }
  globalThis.document={querySelector:element};
  globalThis.window={addEventListener(){}};
  globalThis.location={hash:'#tuning'};
  globalThis.setInterval=callback=>{interval=callback;return 0};
  for(const [key,value] of Object.entries({'#tune-context':'2048','#tune-output':'16',
    '#tune-dtype':'float32','#tune-goal':'fastest','#tune-budget':'2560'}))element(key).value=value;
  element('#tune-auto').checked=true;
  element('#tune-native-check').checked=true;
  const api=(path,options={})=>new Promise((resolve,reject)=>requests.push({path,options,resolve,reject,answered:false}));
  initTuning(api);
  t.after(()=>{for(const [key,old] of previous){if(old.exists)globalThis[key]=old.value;else delete globalThis[key]}});
  return {
    element,requests,
    async answer(path,value){
      const request=requests.find(r=>r.path===path&&!r.answered);
      assert.ok(request,`Expected pending request ${path}`);
      request.answered=true;request.resolve(structuredClone(value));await tick();return request;
    },
    async select(id){element('#tune-campaign').value=id;void element('#tune-campaign').onchange({target:{value:id}});await tick()},
    interval(){interval()},
  };
}

async function initial(h){
  await h.answer('/api/tuning/state',overview());
  await h.answer('/api/tuning/campaigns/case-a',record('case-a'));
}

test('late campaign response cannot overwrite a newer selection',async t=>{
  const h=harness(t);
  await h.answer('/api/tuning/state',overview());
  await h.select('case-b');
  await h.answer('/api/tuning/campaigns/case-a',record('case-a'));
  assert.match(h.element('#tune-status').textContent,/Loading/);
  assert.equal(h.element('#tune-generation').hidden,true);
  await h.answer('/api/tuning/state',overview());
  await h.answer('/api/tuning/campaigns/case-b',record('case-b'));
  assert.match(h.element('#tune-status').textContent,/case-b/);
  assert.ok(h.element('#tune-workload').textContent.includes(Number(32768).toLocaleString()));
  assert.equal(h.element('#tune-export').href,'/api/tuning/campaigns/case-b/export');
  assert.equal(h.element('#tune-campaign').value,'case-b');
});

test('starting a campaign during an old refresh preserves the newly returned ID',async t=>{
  const h=harness(t);await initial(h);
  h.element('#tune-refresh').onclick();
  const submitted=h.element('#tune-form').onsubmit(event);
  await h.answer('/api/tuning/start',{id:'case-new'});await submitted;
  await h.answer('/api/tuning/state',overview());
  assert.match(h.element('#tune-status').textContent,/Loading/);
  await h.answer('/api/tuning/state',overview(['case-new','case-a','case-b']));
  await h.answer('/api/tuning/campaigns/case-new',record('case-new','queued'));
  assert.equal(h.element('#tune-campaign').value,'case-new');
  assert.match(h.element('#tune-status').textContent,/case-new/);
  assert.equal(h.element('#tune-generation').hidden,true);
});

test('unchanged polling preserves action and selection elements',async t=>{
  const h=harness(t);await initial(h);
  const profileWrites=h.element('#tune-profile').writes, selectionWrites=h.element('#tune-campaign').writes;
  h.interval();
  await h.answer('/api/tuning/state',overview());
  await h.answer('/api/tuning/campaigns/case-a',record('case-a'));
  assert.equal(h.element('#tune-profile').writes,profileWrites);
  assert.equal(h.element('#tune-campaign').writes,selectionWrites);
});

test('generation from an earlier campaign is not displayed under another campaign',async t=>{
  const h=harness(t);await initial(h);
  const generated=h.element('#tune-generate-form').onsubmit(event);
  await h.select('case-b');
  const request=await h.answer('/api/tuning/generate',{id:'inference-a'});await generated;
  assert.equal(JSON.parse(request.options.body).campaign,'case-a');
  await h.answer('/api/tuning/state',overview());
  await h.answer('/api/tuning/campaigns/case-b',record('case-b'));
  await h.answer('/api/tuning/inference/inference-a',{status:'completed',text:'Authored private output for A'});
  assert.equal(h.element('#tune-text').textContent,'');
  assert.equal(h.element('#tune-inference-state').textContent,'');
  assert.match(h.element('#tune-status').textContent,/case-b/);
});

test('viewing history explicitly identifies the different campaign that Cancel will stop',async t=>{
  const h=harness(t), state=overview();
  state.active_job='case-b';state.active_kind='campaign';
  await h.answer('/api/tuning/state',state);
  await h.answer('/api/tuning/campaigns/case-a',record('case-a'));
  assert.equal(h.element('#tune-cancel').textContent,'Cancel active campaign');
  assert.match(h.element('#tune-running-note').textContent,/case-b/);
  assert.equal(h.element('#tune-running-note').hidden,false);
  void h.element('#tune-cancel').onclick();
  assert.ok(h.requests.some(r=>r.path==='/api/tuning/jobs/case-b/cancel'));
  assert.ok(!h.requests.some(r=>r.path==='/api/tuning/jobs/case-a/cancel'));
});

test('imported evidence is presented as a recorded reference rather than a new execution',async t=>{
  const h=harness(t), imported=record('case-a');
  imported.status.record_origin='imported_reference';
  await h.answer('/api/tuning/state',overview());
  await h.answer('/api/tuning/campaigns/case-a',imported);
  assert.match(h.element('#tune-workload').textContent,/^Imported recorded result/);
});

test('generation waits for owned cleanup and distinguishes all request clocks',async t=>{
  const h=harness(t);await initial(h);
  const submitted=h.element('#tune-generate-form').onsubmit(event);
  await h.answer('/api/tuning/generate',{id:'inference-clocks'});
  await h.answer('/api/tuning/state',overview());
  await h.answer('/api/tuning/campaigns/case-a',record('case-a'));
  await h.answer('/api/tuning/inference/inference-clocks',{status:'completed',owned_job_active:true,text:'Authored output'});
  await submitted;
  assert.match(h.element('#tune-inference-state').textContent,/Finishing owned job cleanup/);
  assert.equal(h.element('#tune-inference-detail').hidden,true);
  h.interval();
  await h.answer('/api/tuning/state',overview());
  await h.answer('/api/tuning/campaigns/case-a',record('case-a'));
  await h.answer('/api/tuning/inference/inference-clocks',{status:'completed',owned_job_active:false,
    total_request_s:1,owned_process_total_s:2,service_job_total_s:3,service_timing_scope:'Authored cleanup scope'});
  const detail=JSON.parse(h.element('#tune-inference-detail').textContent);
  assert.equal(detail.worker_request_s,1);assert.equal(detail.owned_process_total_s,2);
  assert.equal(detail.service_job_total_s,3);assert.ok(detail.browser_observed_s>=0);
  assert.match(detail.browser_timing_scope,/polling/);
  assert.equal(h.element('#tune-inference-detail').hidden,false);
  assert.equal(h.element('#tune-inference-cost').hidden,false);
  assert.equal(h.element('#tune-request-details').hidden,false);
  assert.match(h.element('#tune-inference-cost').textContent,/Including launch and cleanup: 3 s/);
});
