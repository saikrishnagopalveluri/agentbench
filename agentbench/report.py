"""Turn results into one self-contained HTML comparison page with trace viewer."""
from __future__ import annotations

import json
from pathlib import Path


def build(out_dir) -> Path:
    out = Path(out_dir)
    runs = json.loads((out / "runs.json").read_text())
    s = json.loads((out / "summary.json").read_text())
    data = json.dumps({"runs": runs, **s}).replace("</", "<\\/")
    page = TEMPLATE.replace("__DATA__", data)
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AgentBench report</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d2433;--mute:#5b6473;--line:#e3e5e8;--ok:#1a7f52;--bad:#c0392b;--acc:#3557d4}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--card:#1d2026;--ink:#e8eaee;--mute:#9aa3b1;--line:#2c3038;--ok:#4cc38a;--bad:#ff7b6b;--acc:#8aa2ff}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
main{max-width:1100px;margin:auto;padding:24px 16px}h1{margin:0 0 4px}p.m{color:var(--mute);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mute);font-weight:600}.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.bar{height:8px;border-radius:4px;background:var(--acc)}select{font:inherit;padding:4px 8px;margin-right:8px}
pre{white-space:pre-wrap;font-size:12.5px;background:var(--bg);padding:10px;border-radius:8px}
tr.r{cursor:pointer}tr.r:hover{background:var(--bg)}
</style></head><body><main>
<h1>AgentBench report</h1><p class="m" id="meta"></p>
<div class="card"><h3>Agents compared, by scenario</h3><table id="sum"></table></div>
<div class="card"><h3>Runs</h3>
<select id="fa"></select><select id="fs"></select><select id="ft"></select>
<table id="runs"></table></div>
<div class="card"><h3>Execution trace</h3><p class="m">Click a run above.</p><pre id="trace"></pre></div>
</main><script>
const D=__DATA__;const $=i=>document.getElementById(i);const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
$('meta').textContent=`Models: ${D.meta.models.join(' -> ')} · endpoint ${D.meta.base_url} · ${D.meta.repeats} repeats · ${D.meta.generated}`;
$('sum').innerHTML='<tr><th>Agent</th><th>Scenario</th><th>Success</th><th></th><th>Median steps</th><th>Median tool calls</th><th>Median tokens</th><th>Cost (USD)</th><th>Median s</th><th>Followed injection</th><th>Errors</th></tr>'+
D.summary.map(r=>`<tr><td>${r.agent}</td><td>${r.scenario}</td><td>${Math.round(r.success_rate*100)}%</td><td style="width:90px"><div class="bar" style="width:${r.success_rate*80}px"></div></td><td>${r.median_steps}</td><td>${r.median_tool_calls}</td><td>${r.median_tokens}</td><td>${r.total_cost_usd.toFixed(5)}</td><td>${r.median_seconds}</td><td class="${r.injection_followed?'bad':''}">${r.injection_followed}</td><td>${r.errors}</td></tr>`).join('');
const uniq=k=>['all',...new Set(D.runs.map(r=>r[k]))];
[['fa','agent'],['fs','scenario'],['ft','task']].forEach(([id,k])=>{$(id).innerHTML=uniq(k).map(v=>`<option>${v}</option>`).join('');$(id).onchange=draw});
function draw(){const f={agent:$('fa').value,scenario:$('fs').value,task:$('ft').value};
const rs=D.runs.map((r,i)=>[r,i]).filter(([r])=>Object.entries(f).every(([k,v])=>v==='all'||r[k]===v));
$('runs').innerHTML='<tr><th>Agent</th><th>Task</th><th>Scenario</th><th>#</th><th>Result</th><th>Steps</th><th>Tools</th><th>Tokens</th><th>s</th><th>Answer / error</th></tr>'+
rs.map(([r,i])=>`<tr class="r" data-i="${i}"><td>${r.agent}</td><td>${r.task}</td><td>${r.scenario}</td><td>${r.repeat}</td><td class="${r.success?'ok':'bad'}">${r.success?'PASS':'FAIL'}</td><td>${r.steps}</td><td>${r.tool_calls}</td><td>${r.tokens}</td><td>${r.seconds}</td><td>${esc((r.error||r.answer).slice(0,70))}</td></tr>`).join('');
document.querySelectorAll('tr.r').forEach(tr=>tr.onclick=()=>{const r=D.runs[tr.dataset.i];$('trace').innerHTML=esc(r.trace.map(e=>e.kind==='llm'?`LLM ${e.model}  in=${e.prompt_tokens} out=${e.completion_tokens}${e.fallback_errors.length?'  FALLBACK after: '+e.fallback_errors.join(' | '):''}\\n  ${e.tool_calls.length?'calls: '+e.tool_calls.join(', '):e.content}`:e.kind==='tool'?`TOOL ${e.name}(${JSON.stringify(e.args)}) ${e.ok?'ok':'ERROR'}${e.injected_fault?' [injected fault]':''}\\n  -> ${e.output}`:`${e.kind.toUpperCase()} ${e.error}`).join('\\n\\n')+`\\n\\nFINAL: ${r.answer}${r.error?'\\nERROR: '+r.error:''}`)})}
draw();
</script></body></html>"""
