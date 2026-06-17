import sys, pickle, glob, os, statistics as st
base = os.path.expanduser("~/fedbn-experiment/flower/baselines/fedbn/results")
labels = {"base":"FedBN baseline","stats":"Stats-shared","oracle":"Oracle(pers-fresh)","fedavg":"FedAvg"}
def final(metric_series):
    return metric_series[-1][1] if metric_series else None
def load(seed,label):
    g = glob.glob(f"{base}/s{seed}_{label}/*/history.pkl")
    if not g: return None
    h = pickle.load(open(g[0],"rb"))["history"]
    md = h.metrics_distributed
    return final(md.get("overall_accuracy",[])), final(md.get("min_client_accuracy",[]))
print(f"{'config':22} {'overall mean±std':>20} {'worst-client mean±std':>24}   n")
agg={}
for label,name in labels.items():
    ov=[]; wc=[]
    for seed in range(5):
        r=load(seed,label)
        if r: ov.append(r[0]*100); wc.append(r[1]*100)
    if not ov: 
        print(f"{name:22} {'(no data)':>20}"); continue
    def ms(x): return (st.mean(x), st.pstdev(x) if len(x)>1 else 0.0)
    om,osd=ms(ov); wm,wsd=ms(wc)
    agg[label]=(om,osd,wm,wsd,len(ov))
    print(f"{name:22} {om:6.2f} ± {osd:4.2f} ({len(ov)})   {wm:6.2f} ± {wsd:4.2f}")
print()
if "oracle" in agg and "fedavg" in agg:
    om,osd,owm,owsd,_=agg["oracle"]; fm,fsd,fwm,fwsd,_=agg["fedavg"]
    print(f"Oracle - FedAvg : overall {om-fm:+.2f}  worst-client {owm-fwm:+.2f}")
if "stats" in agg and "base" in agg:
    sm,ssd,swm,swsd,_=agg["stats"]; bm,bsd,bwm,bwsd,_=agg["base"]
    print(f"Stats-shared - FedBN baseline : overall {sm-bm:+.2f}  worst-client {swm-bwm:+.2f}")
