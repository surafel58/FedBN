"""Path B Direction-B headroom gate: can a SHRINKAGE estimator (interpolate a
client's scarce/biased local BN stats with a global BN prior) beat FedAvg's
worst-client?  Offline λ-sweep ceiling (λ* hand-picked = best the family could do).

DESIGN 1 (confirmed): base model = FedAvg model (partial-participation trained).
  global prior μ_global,σ²_global = the FedAvg model's OWN aggregated running stats
    (real, available — downloaded with the model).
  local μ_local,σ²_local = recompute BN stats on k of the client's OWN samples.
  stats(λ) = λ·local + (1−λ)·global, applied to the FedAvg model, then score.
  => λ=0 ≡ FedAvg exactly (bar); λ=1 ≡ naive test-time recompute at k.

NO TEST-LABEL LEAKAGE: local stats use the client's TRAIN split only (inputs only;
  labels used solely to select the biased subset in Regime B, never fed to BN).
  The TEST split is used ONLY to score accuracy.
Regimes: A=SCARCE (k random samples); B=BIASED (k samples from only 2 classes).

PERF: per client we cache the TRANSFORMED train tensor (CPU, for sampling) and the
  TRANSFORMED test tensor (GPU, capped) ONCE, so the 8k×11λ×2regime eval sweep is
  pure GPU forward passes (no repeated CPU image transforms).
"""

import argparse, copy, json, os, random
import numpy as np
import torch
import torch.nn as nn

from fedbn.model import CNNModel
from fedbn.dataset import load_partition

DOMAINS = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST_M"]
BN = [f"bn{i}" for i in range(1, 6)]


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_clients(npd, seed, dev, batch=32, test_cap=2000):
    """20 clients (npd/domain), each a distinct 1/10 partition. Caches
    transformed train tensor (CPU) + transformed test tensor (GPU, capped)."""
    clients = []
    for di, dom in enumerate(DOMAINS):
        parts = list(range(10)); random.Random(1000 * seed + di).shuffle(parts)
        for j in range(npd):
            tr, te = load_partition(dom, str(_DATA), [parts[j]], batch)
            trds, teds = tr.dataset, te.dataset
            trX = torch.stack([trds[i][0] for i in range(len(trds))])      # CPU
            trY = np.asarray(trds.labels).reshape(-1)
            n = min(test_cap, len(teds))
            teX = torch.stack([teds[i][0] for i in range(n)]).to(dev)      # GPU
            teY = torch.tensor([int(teds[i][1]) for i in range(n)]).to(dev)
            clients.append(dict(dom=dom, train=tr, trX=trX, trY=trY,
                                teX=teX, teY=teY, ntest=len(teds)))
    return clients


def agg(sds, keys):
    return {k: sum(sd[k].float() for sd in sds) / len(sds) for k in keys}


def train_federated(clients, mode, rounds, seed, dev, lr=0.01, frac=0.1):
    seed_all(seed)
    g = CNNModel().to(dev)
    nbn = [k for k in g.state_dict() if "bn" not in k]
    allk = list(g.state_dict().keys())
    cstate = {i: None for i in range(len(clients))}
    n_samp = max(1, int(frac * len(clients)))
    for _ in range(rounds):
        sampled = random.sample(range(len(clients)), n_samp)
        sds = []
        for i in sampled:
            m = copy.deepcopy(g).to(dev)
            if mode == "fedbn" and cstate[i] is not None:
                m.load_state_dict(cstate[i], strict=False)
            opt = torch.optim.SGD(m.parameters(), lr=lr); m.train()
            for x, y in clients[i]["train"]:
                opt.zero_grad()
                loss = nn.functional.cross_entropy(m(x.to(dev).float()), y.to(dev).long())
                loss.backward(); opt.step()
            if mode == "fedbn":
                cstate[i] = {k: v.detach().clone() for k, v in m.state_dict().items() if "bn" in k}
            sds.append({k: v.detach() for k, v in m.state_dict().items()})
        keys = nbn if mode == "fedbn" else allk
        new = agg(sds, keys); gsd = g.state_dict()
        for k in keys: gsd[k].copy_(new[k].to(gsd[k].dtype))
        g.load_state_dict(gsd)
    return g, cstate


def capture_stats(model):
    sd = model.state_dict()
    return {b: (sd[f"{b}.running_mean"].clone(), sd[f"{b}.running_var"].clone()) for b in BN}


@torch.no_grad()
def recompute_stats(model, x, dev):
    saved = {}
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.reset_running_stats(); saved[mod] = mod.momentum; mod.momentum = None
    model.train()
    model(x.to(dev).float())
    for mod, mm in saved.items(): mod.momentum = mm
    model.eval()
    return capture_stats(model)


def set_mix(model, local, glob, lam):
    sd = model.state_dict()
    for b in BN:
        sd[f"{b}.running_mean"].copy_(lam * local[b][0] + (1 - lam) * glob[b][0])
        sd[f"{b}.running_var"].copy_(lam * local[b][1] + (1 - lam) * glob[b][1])


@torch.no_grad()
def score(model, teX, teY, bs=512):
    model.eval(); correct = 0
    for i in range(0, teX.size(0), bs):
        out = model(teX[i:i + bs].float())
        correct += (out.argmax(1) == teY[i:i + bs]).sum().item()
    return correct / teX.size(0)


def sample_k(c, k, regime, seed):
    labels = c["trY"]; n = len(labels); rng = random.Random(seed)
    if k == 0 or k >= n:
        idx = list(range(n))
    elif regime == "A":
        idx = rng.sample(range(n), k)
    else:
        classes = sorted(set(labels.tolist())); two = rng.sample(classes, min(2, len(classes)))
        pool = [i for i in range(n) if labels[i] in two]
        idx = pool if k >= len(pool) else rng.sample(pool, k)
    return c["trX"][idx]


def worst_overall(accs, clients):
    worst = min(accs) * 100
    nt = [c["ntest"] for c in clients]
    overall = sum(a * w for a, w in zip(accs, nt)) / sum(nt) * 100
    return worst, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--ks", default="4,8,16,32,64,128,256,full")
    ap.add_argument("--lams", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--test_cap", type=int, default=2000)
    ap.add_argument("--out", default=os.path.expanduser("~/fedbn-experiment/mix_probe_raw.json"))
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    ks = [None if x == "full" else int(x) for x in args.ks.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
    rec = []
    for seed in seeds:
        clients = build_clients(4, seed, dev, test_cap=args.test_cap)
        Mavg, _ = train_federated(clients, "fedavg", 30, seed, dev)
        Mbn, cstate = train_federated(clients, "fedbn", 30, seed, dev)
        glob = capture_stats(Mavg)
        # FedBN reference lines
        for tag, mode in [("fedbn_norecompute", "stale"), ("fedbn_fullrecompute", "full")]:
            accs = []
            for i, c in enumerate(clients):
                if cstate[i] is not None:
                    Mbn.load_state_dict(cstate[i], strict=False)
                if mode == "full":
                    st = recompute_stats(Mbn, sample_k(c, 0, "A", seed), dev)
                    set_mix(Mbn, st, st, 1.0)
                accs.append(score(Mbn, c["teX"], c["teY"]))
            w, o = worst_overall(accs, clients)
            rec.append(dict(seed=seed, regime="-", k="-", lam="-", tag=tag, worst=w, overall=o))
        # Part 1+2 on FedAvg model: stats(λ)=λ·local_k+(1-λ)·global (reuse Mavg; BN-only mutate)
        for regime in ["A", "B"]:
            for k in ks:
                local = [recompute_stats(Mavg, sample_k(c, 0 if k is None else k, regime, seed), dev)
                         for c in clients]
                for lam in lams:
                    accs = []
                    for i, c in enumerate(clients):
                        set_mix(Mavg, local[i], glob, lam)
                        accs.append(score(Mavg, c["teX"], c["teY"]))
                    w, o = worst_overall(accs, clients)
                    rec.append(dict(seed=seed, regime=regime, k=("full" if k is None else k),
                                    lam=round(lam, 2), tag="mix", worst=w, overall=o))
        print(f"seed {seed} done", flush=True)
    json.dump(rec, open(args.out, "w"))
    print("WROTE", args.out, len(rec), "records")


if __name__ == "__main__":
    from fedbn.dataset import DATA_DIRECTORY as _D
    globals()["_DATA"] = _D
    main()
