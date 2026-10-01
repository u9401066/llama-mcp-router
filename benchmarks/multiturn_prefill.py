"""Prefill per turn of a 5-turn conversation: all tools vs router (default vs --sticky).

Start two routers in front of llama-server on :8001 (8090 with --sticky, 8091 default), then run this file.
"""
import json, sys, time, urllib.request
U=["Find recent RCTs on remimazolam for ICU sedation.","Export them to BibTeX.","Which papers cite PMID 31452104?","Get the full text of PMC7654321.","Look up the BRCA1 gene."]
def run(port, nonce, bypass=False):
    msgs=[{"role":"system","content":"You are a biomedical research assistant. Nonce %s"%nonce}]
    out=[]
    for i,u in enumerate(U):
        msgs.append({"role":"user","content":u})
        body={"model":"m","messages":msgs,"max_tokens":1,"temperature":0}
        h={"content-type":"application/json"}
        if bypass:
            h["x-router-bypass"]="1"; body["tools"]=json.load(open('/tmp/all_tools.json'))
        r=urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(body).encode(),h),timeout=600)
        d=json.load(r); tm=d["timings"]
        out.append((tm["prompt_ms"],tm["prompt_n"],tm["cache_n"],r.headers.get("x-router-tools","-").count(",")+1 if not bypass else 41))
        msgs.append({"role":"assistant","content":"Done."})
    return out
if __name__=="__main__":
    json.dump(json.load(urllib.request.urlopen("http://127.0.0.1:8001/tools")) and [t["definition"] for t in json.load(urllib.request.urlopen("http://127.0.0.1:8001/tools"))],open('/tmp/all_tools.json','w'))
    for name,port,bp in [("all tools (no router)",8090,True),("router, --sticky",8090,False),("router, default (not sticky)",8091,False)]:
        res=run(port,name+str(time.time()),bp)
        print(name)
        for i,(ms,n,c,k) in enumerate(res,1): print(f"  turn {i}: {k:2d} tools  prefill {ms:7.0f} ms  (read {n:5d} tokens, reused {c:5d})")
        print(f"  total prefill: {sum(r[0] for r in res)/1000:.1f} s")
