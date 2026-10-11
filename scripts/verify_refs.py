"""Verify paper/refs.bib against publisher-deposited metadata (Crossref REST + arXiv API).

Run before every submission. Prints a per-key verdict; apply corrections by hand so each fix
lands in the bib with a dated VERIFIED/CORRECTED note that a reviewer can audit.

    python scripts/verify_refs.py /tmp/refs_verified.json

WHAT IT CAUGHT on 2026-10-07: robertson2009bm25 was cited as Found. Trends IR 3(4):333--389.
The correct reference is 4(1--2):1--174. That is a fabricated-looking citation produced by
nothing worse than confident recall, and it is the class of error the repo's citation policy
exists to stop.

KNOWN GAPS, both flagged inline in refs.bib rather than silently accepted:
  * ICML 2011 and earlier PMLR volumes are absent from Crossref (welling2011sgld).
  * OpenReview-only documents are in neither source -- and a title search for lecun2022path
    returns a DIFFERENT paper (Dawid & LeCun 2023), which is exactly how a mis-citation
    enters a bibliography.

Uses curl rather than urllib: this machine has no CA bundle available to Python.
"""
import json, re, time, urllib.parse, urllib.request, sys

ENTRIES = [
 ("hopfield1982","Neural networks and physical systems with emergent collective computational abilities"),
 ("krotov2016dense","Dense Associative Memory for Pattern Recognition"),
 ("ramsauer2021hopfield","Hopfield Networks is All You Need"),
 ("millidge2022universal","Universal Hopfield Networks: A General Framework for Single-Shot Associative Memory Models"),
 ("ackley1985boltzmann","A Learning Algorithm for Boltzmann Machines"),
 ("welling2011sgld","Bayesian Learning via Stochastic Gradient Langevin Dynamics"),
 ("lecun2022path","A Path Towards Autonomous Machine Intelligence"),
 ("assran2023ijepa","Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"),
 ("packer2023memgpt","MemGPT: Towards LLMs as Operating Systems"),
 ("zhong2024memorybank","MemoryBank: Enhancing Large Language Models with Long-Term Memory"),
 ("xu2025amem","A-MEM: Agentic Memory for LLM Agents"),
 ("gutierrez2024hipporag","HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models"),
 ("maharana2024locomo","Evaluating Very Long-Term Conversational Memory of LLM Agents"),
 ("wu2025longmemeval","LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory"),
 ("reimers2019sbert","Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"),
 ("he2016resnet","Deep Residual Learning for Image Recognition"),
 ("robertson2009bm25","The Probabilistic Relevance Framework: BM25 and Beyond"),
 ("cormack2009rrf","Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"),
 ("yang2018softmax","Breaking the Softmax Bottleneck: A High-Rank RNN Language Model"),
 ("mallat1993matching","Matching Pursuits with Time-Frequency Dictionaries"),
 ("liu2016svgd","Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm"),
 ("cameron2008bootstrap","Bootstrap-Based Improvements for Inference with Clustered Errors"),
 ("ethayarajh2019contextual","How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings"),
 ("gao2019representation","Representation Degeneration Problem in Training Natural Language Generation Models"),
 # Added 2026-10-11 with the six-corpus study. Four of these are distributed under a name
 # that does NOT appear in their title (2WikiMultihopQA, QASPER, MSC, BGE), so a title
 # search for the dataset name finds nothing and a DOI lookup is the only reliable route.
 ("yang2018hotpotqa","HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering"),
 ("ho2020twowiki","Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps"),
 ("dasigi2021qasper","A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers"),
 ("thakur2021beir","BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models"),
 ("xu2022msc","Beyond Goldfish Memory: Long-Term Open-Domain Conversation"),
 ("radford2021clip","Learning Transferable Visual Models From Natural Language Supervision"),
 ("xiao2024bge","C-Pack: Packed Resources For General Chinese Embeddings"),
]

import subprocess
def get(url):
    r=subprocess.run(["curl","-sfL","--max-time","30","-A","bib-verify/1.0",url],
                     capture_output=True)
    if r.returncode!=0: raise RuntimeError(f"curl {r.returncode}")
    return r.stdout.decode("utf-8","replace")

def crossref(title):
    q=urllib.parse.quote(title)
    try:
        d=json.loads(get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=3&select=title,author,issued,container-title,volume,issue,page,DOI,type"))
    except Exception as e:
        return {"err":str(e)}
    best=None
    tl=re.sub(r'[^a-z0-9]','',title.lower())
    for it in d.get("message",{}).get("items",[]):
        t=(it.get("title") or [""])[0]
        if re.sub(r'[^a-z0-9]','',t.lower())[:60]==tl[:60]:
            best=it; break
    if best is None: return {"nomatch":True}
    au=best.get("author",[])
    return {
      "title":(best.get("title") or [""])[0],
      "authors":[f"{a.get('family','')}, {a.get('given','')}".strip(", ") for a in au],
      "n_authors":len(au),
      "year":(best.get("issued",{}).get("date-parts") or [[None]])[0][0],
      "venue":(best.get("container-title") or [""])[0],
      "volume":best.get("volume"),"issue":best.get("issue"),"pages":best.get("page"),
      "doi":best.get("DOI"),"type":best.get("type"),
    }

def arxiv(title):
    q=urllib.parse.quote(f'ti:"{title}"')
    try:
        x=get(f"http://export.arxiv.org/api/query?search_query={q}&max_results=1")
    except Exception as e:
        return {"err":str(e)}
    t=re.search(r"<entry>.*?<title>(.*?)</title>",x,re.S)
    i=re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>",x.split("<entry>")[-1] if "<entry>" in x else "")
    names=re.findall(r"<name>([^<]+)</name>", x.split("<entry>")[-1]) if "<entry>" in x else []
    pub=re.search(r"<published>(\d{4})",x.split("<entry>")[-1]) if "<entry>" in x else None
    if not t: return {"nomatch":True}
    return {"title":" ".join(t.group(1).split()),"arxiv":i.group(1) if i else None,
            "n_authors":len(names),"authors":names,"year":pub.group(1) if pub else None}

out={}
for key,title in ENTRIES:
    c=crossref(title); time.sleep(0.7)
    a=arxiv(title) if (c.get("nomatch") or c.get("err")) else None
    if a: time.sleep(2.5)
    out[key]={"crossref":c,"arxiv":a}
    src = "crossref" if not (c.get("nomatch") or c.get("err")) else ("arxiv" if a and not (a.get("nomatch") or a.get("err")) else "NONE")
    print(f"{key:26} {src}")
json.dump(out,open(sys.argv[1],"w"),indent=1)
print("saved")
