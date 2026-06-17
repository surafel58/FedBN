import sys, pickle, glob, os, statistics as st
base = os.path.expanduser("~/fedbn-experiment/flower/baselines/fedbn/results")
order = [("base","FedBN baseline"),("fedavg","FedAvg"),
         ("tt1","TestTime-BN (1 batch=32)"),("tt4","TestTime-BN (4 batch=128)"),
         ("oracle","Oracle (full recompute)")]
seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv)>1 else ["0","1","2","3","4"])]
def final(s): return s[-1][1] if s else None
def load(seed,label):
    g = glob.glob(f"{base}/s{seed}_{label}/*/history.pkl")
    if not g: return None
    md = pickle.load(open(g[0],"rb"))["history"].metrics_distributed
    return final(md.get("overall_accuracy",[])), final(md.get("min_client_accuracy",[]))
def ms(x): return (st.mean(x), st.pstdev(x) if len(x)>1 else 0.0)
print(f"seeds={seeds}\n{'config':28} {'overall':>16} {'worst-client':>16}  n")
agg={}
for label,name in order:
    ov=[]; wc=[]
    for s in seeds:
        r=load(s,label)
        if r and r[0] is not None: ov.append(r[0]*100); wc.append(r[1]*100)
    if not ov: print(f"{name:28} {'(no data)':>16}"); continue
    om,osd=ms(ov); wm,wsd=ms(wc); agg[label]=(wm,wsd,om,osd)
    print(f"{name:28} {om:6.2f} ± {osd:4.2f}   {wm:6.2f} ± {wsd:4.2f}  {len(ov)}")
print()
if "fedavg" in agg:
    fwm,fwsd,_,_=agg["fedavg"]
    for lab in ["tt1","tt4","oracle"]:
        if lab in agg:
            wm,wsd,_,_=agg[lab]
            print(f"{lab:8} worst-client - FedAvg = {wm-fwm:+.2f}  (vs FedAvg {fwm:.2f}±{fwsd:.2f})")
if "oracle" in agg:
    owm,_,_,_=agg["oracle"]
    for lab in ["tt1","tt4"]:
        if lab in agg:
            wm,_,_,_=agg[lab]; print(f"{lab:8} worst-client - Oracle = {wm-owm:+.2f}  (recovery vs full)")
