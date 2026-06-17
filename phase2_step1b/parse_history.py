import sys, pickle
from pathlib import Path
h = pickle.load(open(sys.argv[1],"rb"))["history"]
md = h.metrics_distributed  # evaluate metrics
def last(series): return series[-1] if series else None
overall = md.get("overall_accuracy", [])
minc = md.get("min_client_accuracy", [])
nrep = md.get("num_clients_reported", [])
acc = md.get("accuracy", [])  # per-domain dict series
print("final round (overall):", last(overall))
print("final round (min-client):", last(minc))
print("final round (num reported):", last(nrep))
if acc:
    rnd, perdom = acc[-1]
    print(f"per-domain @round {rnd}:")
    for k,v in perdom.items(): print(f"  {k:12s} {v*100:.2f}")
