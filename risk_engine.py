import pandas as pd, numpy as np

# ---------- THE RULEBOOK (every number here is a business decision, easy to change) ----------
EXPECTED = {  # procedure -> (expected diagnosis, normal stay range in days)
 "Knee Replacement": ("Osteoarthritis", (3,4)),
 "Angioplasty": ("Coronary Artery Disease", (2,3)),
 "C-Section": ("Obstructed Labour", (3,4)),
 "Appendectomy": ("Acute Appendicitis", (2,3)),
 "Cataract Surgery": ("Cataract", (0,1)),
}
POINTS = dict(dup=60, dx=60, bill_high=50, bill_med=20, los=40, provider=20, prior=10)
BILL_HIGH, BILL_MED = 1.9, 1.5       # x peer median
HIGH_CUT, MED_CUT = 50, 25           # risk-score cut-offs
PROVIDER_ALERT = 1.15                # hospital's avg bill vs peer median
# --------------------------------------------------------------------------------------------

def score(df):
    d = df.copy()
    d["claim_date"] = pd.to_datetime(d["claim_date"])
    med = d.groupby("procedure")["billed_amount"].transform("median")
    d["bill_ratio"] = d["billed_amount"]/med
    prov_ratio = d.groupby("provider")["bill_ratio"].transform("mean")
    d["provider_ratio"] = prov_ratio

    # duplicate: same patient+procedure+provider seen earlier within 7 days
    d = d.sort_values(["patient_id","procedure","provider","claim_date"])
    g = d.groupby(["patient_id","procedure","provider"])["claim_date"]
    gap = g.diff().dt.days
    d["is_dup"] = gap.le(7).fillna(False)
    d = d.sort_index()

    d["dx_mismatch"] = d.apply(lambda r: r["diagnosis"]!=EXPECTED[r["procedure"]][0], axis=1)
    def los_bad(r):
        lo,hi = EXPECTED[r["procedure"]][1]
        return (r["los_days"]>hi+2) or (lo>0 and r["los_days"]<lo-1) or (lo==0 and r["los_days"]>hi+2)
    d["los_abnormal"] = d.apply(los_bad, axis=1)

    pts = np.zeros(len(d))
    reasons = [[] for _ in range(len(d))]
    for i,(_,r) in enumerate(d.iterrows()):
        if r["is_dup"]: pts[i]+=POINTS["dup"]; reasons[i].append("Possible duplicate claim")
        if r["dx_mismatch"]: pts[i]+=POINTS["dx"]; reasons[i].append("Diagnosis does not match procedure")
        if r["bill_ratio"]>=BILL_HIGH: pts[i]+=POINTS["bill_high"]; reasons[i].append(f"Billed {r['bill_ratio']:.1f}x the peer median")
        elif r["bill_ratio"]>=BILL_MED: pts[i]+=POINTS["bill_med"]; reasons[i].append(f"Billed {r['bill_ratio']:.1f}x the peer median")
        if r["los_abnormal"]: pts[i]+=POINTS["los"]; reasons[i].append(f"Length of stay ({r['los_days']} days) unusual for procedure")
        if r["provider_ratio"]>=PROVIDER_ALERT and r["bill_ratio"]>=1.4: pts[i]+=POINTS["provider"]; reasons[i].append("Hospital bills consistently above peers")
        if r["prior_claims"]>=6: pts[i]+=POINTS["prior"]; reasons[i].append("Unusually high prior-claim frequency")
    d["risk_score"]=pts
    d["risk_level"]=np.where(pts>=HIGH_CUT,"High",np.where(pts>=MED_CUT,"Medium","Low"))
    d["reasons"]=["; ".join(r) for r in reasons]
    return d

if __name__=="__main__":
    df = pd.read_csv("synthetic_claims.csv")
    s = score(df)
    print("Provider avg bill ratio:\n", s.groupby("provider")["provider_ratio"].first().round(2).to_string(), "\n")
    print(s["risk_level"].value_counts().to_string(), "\n")
    for label,flag in [("High only", s.risk_level=="High"), ("High + Medium", s.risk_level.isin(["High","Medium"]))]:
        tp=((flag)&(s.is_anomaly==1)).sum(); fp=((flag)&(s.is_anomaly==0)).sum(); fn=((~flag)&(s.is_anomaly==1)).sum()
        print(f"{label}: flagged={flag.sum()}, caught={tp}/{s.is_anomaly.sum()} ({tp/s.is_anomaly.sum():.0%} recall), precision={tp/flag.sum():.0%}, false alarms={fp}")
    print("\nCatch rate by problem type (High+Medium):")
    f = s.risk_level.isin(["High","Medium"])
    print(s[s.is_anomaly==1].assign(caught=f).groupby("anomaly_type")["caught"].agg(["sum","count"]).assign(rate=lambda x:(x["sum"]/x["count"]).round(2)).to_string())
    print("\nWhat are the false alarms (High+Medium)?")
    print(s[f&(s.is_anomaly==0)].groupby("provider").size().to_string())
    print(s[f&(s.is_anomaly==0)]["reasons"].str.split("; ").explode().value_counts().head(6).to_string())
    s.to_csv("scored_claims.csv",index=False)
