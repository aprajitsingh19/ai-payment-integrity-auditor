import numpy as np, pandas as pd
rng = np.random.default_rng(42)
N = 3000
providers = ["Hospital A","Hospital B","Hospital C","Hospital D","Hospital E","Hospital F","Hospital G","Hospital H"]
bad = {"Hospital E","Hospital H"}
proc = {
 "Knee Replacement": dict(dx="Osteoarthritis", cost=(250000,350000), los=(3,4), age=(55,75), w=0.18),
 "Angioplasty": dict(dx="Coronary Artery Disease", cost=(150000,300000), los=(2,3), age=(50,75), w=0.20),
 "C-Section": dict(dx="Obstructed Labour", cost=(60000,120000), los=(3,4), age=(22,38), w=0.22),
 "Appendectomy": dict(dx="Acute Appendicitis", cost=(50000,90000), los=(2,3), age=(12,60), w=0.20),
 "Cataract Surgery": dict(dx="Cataract", cost=(30000,60000), los=(0,1), age=(50,80), w=0.20),
}
names = list(proc)
pw = np.array([proc[n]["w"] for n in names]); pw = pw/pw.sum()
rows = []
for i in range(N):
    p = rng.choice(names, p=pw)
    cfg = proc[p]
    prov = rng.choice(providers)
    cost = rng.uniform(*cfg["cost"])
    if prov in bad:  # bad actors bill 20-40% higher on average
        cost *= rng.uniform(1.2, 1.4)
    los = int(rng.integers(cfg["los"][0], cfg["los"][1]+1))
    age = int(rng.integers(*cfg["age"]))
    rows.append(dict(
        patient_id=f"P{rng.integers(10000,99999)}",
        claim_date=pd.Timestamp("2026-01-01")+pd.Timedelta(days=int(rng.integers(0,270))),
        provider=prov, procedure=p, diagnosis=cfg["dx"], patient_age=age,
        los_days=los, billed_amount=round(cost,-2),
        prior_claims=int(rng.poisson(2)), anomaly_type="None"))
df = pd.DataFrame(rows)

# plant ~8% anomalies
idx = rng.permutation(N)
n_anom = int(0.08*N)  # 240
split = {"Inflated Billing":0.35,"Duplicate Claim":0.20,"Abnormal Length of Stay":0.20,"Diagnosis Mismatch":0.15,"Provider Outlier":0.10}
pos = 0
dups = []
for k, frac in split.items():
    n = int(round(n_anom*frac))
    for j in idx[pos:pos+n]:
        if k=="Inflated Billing":
            df.loc[j,"billed_amount"] = round(df.loc[j,"billed_amount"]*rng.uniform(2.0,3.0),-2)
        elif k=="Abnormal Length of Stay":
            p = df.loc[j,"procedure"]
            if rng.random()<0.5: df.loc[j,"los_days"] = proc[p]["los"][1]+int(rng.integers(5,10))
            else: df.loc[j,"los_days"] = 0 if proc[p]["los"][0]>0 else 7
        elif k=="Diagnosis Mismatch":
            p = df.loc[j,"procedure"]
            other = rng.choice([x for x in names if x!=p])
            df.loc[j,"diagnosis"] = proc[other]["dx"]
        elif k=="Provider Outlier":
            df.loc[j,"provider"] = rng.choice(list(bad))
            df.loc[j,"billed_amount"] = round(df.loc[j,"billed_amount"]*rng.uniform(1.6,1.9),-2)
            df.loc[j,"prior_claims"] = int(rng.integers(6,10))
        if k!="Duplicate Claim":
            df.loc[j,"anomaly_type"] = k
        else:
            dups.append(j)
    pos += n
# duplicates: add extra rows copying existing claims
dup_rows = df.loc[dups].copy()
dup_rows["claim_date"] = dup_rows["claim_date"] + pd.to_timedelta(rng.integers(1,6,len(dup_rows)), unit="D")
dup_rows["anomaly_type"] = "Duplicate Claim"
df = pd.concat([df, dup_rows], ignore_index=True)
df = df.sort_values("claim_date").reset_index(drop=True)
df.insert(0,"claim_id",[f"C{i+1:04d}" for i in range(len(df))])
df["claim_date"] = df["claim_date"].dt.strftime("%Y-%m-%d")
df["is_anomaly"] = (df["anomaly_type"]!="None").astype(int)
df.to_csv("synthetic_claims.csv", index=False)

print("Total claims:", len(df))
print("Anomalies:", df.is_anomaly.sum(), f"({df.is_anomaly.mean():.1%})")
print(df.anomaly_type.value_counts().to_string())
print()
clean = df[df.is_anomaly==0]
print(clean.groupby("procedure").agg(n=("claim_id","count"),min_bill=("billed_amount","min"),median_bill=("billed_amount","median"),max_bill=("billed_amount","max"),avg_los=("los_days","mean")).round(0).to_string())
print()
print(df.groupby("provider").agg(claims=("claim_id","count"),avg_bill=("billed_amount","mean"),anomalies=("is_anomaly","sum")).round(0).to_string())
print()
print(df.head(5).to_string())
