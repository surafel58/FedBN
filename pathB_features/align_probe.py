"""Path B features-lever ceiling gate: can an AGGRESSIVE feature-alignment objective
reshape the global representation so that SHARED global BN stats beat FedAvg's
worst-client under partial participation (strict eval: no inference-time recompute)?

Alignment loss (added to local training): pull each client's per-channel BN-INPUT
activation stats (batch mean/var of the inputs to bn1..bn5) toward a COMMON target =
the GLOBAL BN running stats broadcast with the model at the start of the round (a real,
available quantity; snapshotted/frozen for the round, no oracle, no test peeking).
  L = CE + α · Σ_b [ mean((μ_batch,b − μ_global,b)^2) + mean((v_batch,b − v_global,b)^2) ]
This makes clients' feature distributions match the shared target -> "normalization-
friendly" features -> one shared stat set fits everyone.

EVAL (strict): score each client's test set with the trained global model using its
SHARED aggregated BN running stats. NO per-client recompute, NO inference data, labels
only for scoring. α=0 ≡ FedAvg (the bar). This is a generous CEILING (aggressive α).

Configs: FedAvg(α=0) | align α∈{0.1,1,10} (+shared stats) | FedBN-no-recompute (collapse
ref) | Step-1c oracle (recompute, unreachable upper bound, context only).
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
    clients = []
    for di, dom in enumerate(DOMAINS):
        parts = list(range(10)); random.Random(1000 * seed + di).shuffle(parts)
        for j in range(npd):
            tr, te = load_partition(dom, str(_DATA), [parts[j]], batch)
            trds, teds = tr.dataset, te.dataset
            trX = torch.stack([trds[i][0] for i in range(len(trds))])
            trY = np.asarray(trds.labels).reshape(-1)
            n = min(test_cap, len(teds))
            teX = torch.stack([teds[i][0] for i in range(n)]).to(dev)
            teY = torch.tensor([int(teds[i][1]) for i in range(n)]).to(dev)
            clients.append(dict(dom=dom, train=tr, trX=trX, trY=trY,
                                teX=teX, teY=teY, ntest=len(teds)))
    return clients


def agg(sds, keys):
    return {k: sum(sd[k].float() for sd in sds) / len(sds) for k in keys}


def capture_stats(model):
    sd = model.state_dict()
    return {b: (sd[f"{b}.running_mean"].clone(), sd[f"{b}.running_var"].clone()) for b in BN}


def add_bn_input_hooks(model):
    """Capture per-channel mean/var of the INPUT to each BN layer (forward hooks)."""
    caps, handles = {}, []
    def mk(name):
        def hook(mod, inp, out):
            x = inp[0]
            if x.dim() == 4:
                caps[name] = (x.mean([0, 2, 3]), x.var([0, 2, 3], unbiased=False))
            else:
                caps[name] = (x.mean(0), x.var(0, unbiased=False))
        return hook
    for n, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            handles.append(m.register_forward_hook(mk(n)))
    return caps, handles


def train_align(clients, rounds, seed, dev, alpha, lr=0.01, frac=0.1):
    """FedAvg training (aggregates BN too) + feature-alignment penalty toward the
    broadcast global BN running stats (snapshotted per round)."""
    seed_all(seed)
    g = CNNModel().to(dev)
    allk = list(g.state_dict().keys())
    n_samp = max(1, int(frac * len(clients)))
    for _ in range(rounds):
        gsd = g.state_dict()
        target = {b: (gsd[f"{b}.running_mean"].clone(), gsd[f"{b}.running_var"].clone()) for b in BN}
        sampled = random.sample(range(len(clients)), n_samp)
        sds = []
        for i in sampled:
            m = copy.deepcopy(g).to(dev)
            caps, handles = add_bn_input_hooks(m)
            opt = torch.optim.SGD(m.parameters(), lr=lr); m.train()
            for x, y in clients[i]["train"]:
                opt.zero_grad()
                out = m(x.to(dev).float())
                loss = nn.functional.cross_entropy(out, y.to(dev).long())
                if alpha > 0:
                    al = out.new_zeros(())
                    for b in BN:
                        cm, cv = caps[b]; tm, tv = target[b]
                        al = al + ((cm - tm) ** 2).mean() + ((cv - tv) ** 2).mean()
                    loss = loss + alpha * al
                loss.backward(); opt.step()
            for h in handles: h.remove()
            sds.append({k: v.detach() for k, v in m.state_dict().items()})
        new = agg(sds, allk); gsd = g.state_dict()
        for k in allk: gsd[k].copy_(new[k].to(gsd[k].dtype))
        g.load_state_dict(gsd)
    return g


def train_fedbn(clients, rounds, seed, dev, lr=0.01, frac=0.1):
    seed_all(seed)
    g = CNNModel().to(dev)
    nbn = [k for k in g.state_dict() if "bn" not in k]
    cstate = {i: None for i in range(len(clients))}
    n_samp = max(1, int(frac * len(clients)))
    for _ in range(rounds):
        sampled = random.sample(range(len(clients)), n_samp); sds = []
        for i in sampled:
            m = copy.deepcopy(g).to(dev)
            if cstate[i] is not None: m.load_state_dict(cstate[i], strict=False)
            opt = torch.optim.SGD(m.parameters(), lr=lr); m.train()
            for x, y in clients[i]["train"]:
                opt.zero_grad()
                loss = nn.functional.cross_entropy(m(x.to(dev).float()), y.to(dev).long())
                loss.backward(); opt.step()
            cstate[i] = {k: v.detach().clone() for k, v in m.state_dict().items() if "bn" in k}
            sds.append({k: v.detach() for k, v in m.state_dict().items()})
        new = agg(sds, nbn); gsd = g.state_dict()
        for k in nbn: gsd[k].copy_(new[k].to(gsd[k].dtype))
        g.load_state_dict(gsd)
    return g, cstate


@torch.no_grad()
def recompute_stats(model, x, dev):
    saved = {}
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.reset_running_stats(); saved[mod] = mod.momentum; mod.momentum = None
    model.train(); model(x.to(dev).float())
    for mod, mm in saved.items(): mod.momentum = mm
    model.eval(); return capture_stats(model)


def set_stats(model, stats):
    sd = model.state_dict()
    for b in BN:
        sd[f"{b}.running_mean"].copy_(stats[b][0]); sd[f"{b}.running_var"].copy_(stats[b][1])


@torch.no_grad()
def score(model, teX, teY, bs=512):
    model.eval(); correct = 0
    for i in range(0, teX.size(0), bs):
        correct += (model(teX[i:i + bs].float()).argmax(1) == teY[i:i + bs]).sum().item()
    return correct / teX.size(0)


def worst_overall(accs, clients):
    nt = [c["ntest"] for c in clients]
    return min(accs) * 100, sum(a * w for a, w in zip(accs, nt)) / sum(nt) * 100


def eval_shared(model, clients):           # strict: global model + its shared stats, no recompute
    return worst_overall([score(model, c["teX"], c["teY"]) for c in clients], clients)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--alphas", default="0.1,1.0,10.0")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--test_cap", type=int, default=2000)
    ap.add_argument("--out", default=os.path.expanduser("~/fedbn-experiment/align_raw.json"))
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    rec = []
    for seed in seeds:
        clients = build_clients(4, seed, dev, test_cap=args.test_cap)
        # FedAvg bar = alignment with alpha=0
        g0 = train_align(clients, args.rounds, seed, dev, 0.0)
        w, o = eval_shared(g0, clients)
        rec.append(dict(seed=seed, cfg="FedAvg(a0)", alpha=0.0, worst=w, overall=o))
        # alignment sweep
        for a in alphas:
            g = train_align(clients, args.rounds, seed, dev, a)
            w, o = eval_shared(g, clients)
            rec.append(dict(seed=seed, cfg=f"align(a{a})", alpha=a, worst=w, overall=o))
        # FedBN references
        gb, cstate = train_fedbn(clients, args.rounds, seed, dev)
        accs = []
        for i, c in enumerate(clients):
            if cstate[i] is not None: gb.load_state_dict(cstate[i], strict=False)
            accs.append(score(gb, c["teX"], c["teY"]))
        w, o = worst_overall(accs, clients)
        rec.append(dict(seed=seed, cfg="FedBN_norecompute", alpha="-", worst=w, overall=o))
        accs = []
        for i, c in enumerate(clients):
            if cstate[i] is not None: gb.load_state_dict(cstate[i], strict=False)
            st = recompute_stats(gb, c["trX"], dev); set_stats(gb, st)
            accs.append(score(gb, c["teX"], c["teY"]))
        w, o = worst_overall(accs, clients)
        rec.append(dict(seed=seed, cfg="oracle_recompute", alpha="-", worst=w, overall=o))
        print(f"seed {seed} done", flush=True)
    json.dump(rec, open(args.out, "w"))
    print("WROTE", args.out, len(rec), "records")


if __name__ == "__main__":
    from fedbn.dataset import DATA_DIRECTORY as _D
    globals()["_DATA"] = _D
    main()
