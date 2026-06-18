import sys, pickle, glob, os, statistics as st
base = os.path.expanduser("~/fedbn-experiment/flower/baselines/fedbn/results")
order = [("fbn_nr","#2 FedBN, no recompute (stale local)"),
         ("fedavg","#1 FedAvg (shared global stats)"),
         ("shared","#4 Shared-global-stats (Candidate-3 ceiling)"),
         ("fz5","#3 frozen@5  (ideal stats frozen early)"),
         ("fz15","#3 frozen@15 (ideal stats frozen mid)"),
         ("fz30","#3 frozen@30 (ideal stats @final = no staleness)")]
seeds=[int(x) for x in (sys.argv[1].split(",") if len(sys.argv)>1 else ["0","1","2","3","4"])]
def final(s): return s[-1][1] if s else None
def load(seed,label):
    g=glob.glob(f"{base}/s{seed}_{label}/*/history.pkl")
    if not g: return None
    md=pickle.load(open(g[0],"rb"))["history"].metrics_distributed
    return final(md.get("overall_accuracy",[])), final(md.get("min_client_accuracy",[]))
def ms(x): return (st.mean(x), st.pstdev(x) if len(x)>1 else 0.0)
print(f"seeds={seeds}\n{'config':46} {'overall':>15} {'worst-client':>15}  n")
agg={}
for label,name in order:
    ov=[];wc=[]
    for s in seeds:
        r=load(s,label)
        if r and r[0] is not None: ov.append(r[0]*100); wc.append(r[1]*100)
    if not ov: print(f"{name:46} {'(no data)':>15}"); continue
    om,osd=ms(ov); wm,wsd=ms(wc); agg[label]=(wm,wsd)
    print(f"{name:46} {om:6.2f} ± {osd:4.2f}  {wm:6.2f} ± {wsd:4.2f}  {len(ov)}")
print()
if "fedavg" in agg:
    fwm,fwsd=agg["fedavg"]
    print(f"FedAvg worst-client bar = {fwm:.2f} ± {fwsd:.2f}")
    for lab in ["shared","fz5","fz15","fz30"]:
        if lab in agg:
            wm,wsd=agg[lab]; print(f"  {lab:7} - FedAvg = {wm-fwm:+.2f}  (worst-client {wm:.2f} ± {wsd:.2f})")
