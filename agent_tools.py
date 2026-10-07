import pandas as pd, numpy as np, json
from risk_engine import EXPECTED
S = pd.read_csv("scored_claims.csv", parse_dates=["claim_date"])

def peer_benchmark(cid):
    r = S[S.claim_id==cid].iloc[0]; peers = S[(S.procedure==r.procedure)]
    med = peers.billed_amount.median()
    return dict(procedure=r.procedure, billed=int(r.billed_amount), peer_median=int(med),
        peer_p25=int(peers.billed_amount.quantile(.25)), peer_p75=int(peers.billed_amount.quantile(.75)),
        ratio_to_median=round(r.billed_amount/med,2),
        percentile=int((peers.billed_amount<r.billed_amount).mean()*100),
        excess_over_median=int(max(0,r.billed_amount-med)))

def duplicate_search(cid):
    r = S[S.claim_id==cid].iloc[0]
    m = S[(S.claim_id!=cid)&(S.patient_id==r.patient_id)&(S.procedure==r.procedure)&(S.provider==r.provider)
          &((S.claim_date-r.claim_date).abs().dt.days<=14)]
    return dict(matches=[dict(claim_id=x.claim_id, date=str(x.claim_date.date()), billed=int(x.billed_amount)) for x in m.itertuples()])

def provider_history(prov):
    p = S[S.provider==prov]; allm = S.groupby("provider").bill_ratio.mean().sort_values(ascending=False)
    return dict(provider=prov, claims=len(p), avg_ratio_to_peer_median=round(p.bill_ratio.mean(),2),
        high_risk_claims=int((p.risk_level=="High").sum()), share_flagged=round((p.risk_level!="Low").mean(),3),
        rank_by_billing=int(list(allm.index).index(prov)+1), of=len(allm))

def clinical_consistency(cid):
    r = S[S.claim_id==cid].iloc[0]; dx,(lo,hi)=EXPECTED[r.procedure]
    return dict(diagnosis=r.diagnosis, expected_diagnosis=dx, diagnosis_matches=bool(r.diagnosis==dx),
                los_days=int(r.los_days), expected_los=f"{lo}-{hi}", los_normal=bool(lo-1<=r.los_days<=hi+2))

def policy_recommendation(r):
    """Fallback / guardrail: deterministic recommendation the AI must justify or flag disagreement with."""
    if r.risk_level=="Low": return "Approve"
    if r.is_dup or r.dx_mismatch: return "Escalate to senior investigator"
    if r.risk_score>=70: return "Escalate to senior investigator"
    return "Request itemised bill / documents"

if __name__=="__main__":
    F = S[S.risk_level!="Low"].copy()
    F["policy"]=F.apply(policy_recommendation,axis=1)
    print("Recommendation mix on flagged claims:\n", F.policy.value_counts().to_string(), "\n")
    for t in ["Inflated Billing","Duplicate Claim","Provider Outlier"]:
        cid = F[(F.anomaly_type==t)&(F.risk_level=="High")].claim_id.iloc[0]
        r = S[S.claim_id==cid].iloc[0]
        print("="*70, f"\n{cid} | {r.provider} | {r.procedure} | score {int(r.risk_score)} {r.risk_level} | reasons: {r.reasons}")
        print("peer_benchmark:", json.dumps(peer_benchmark(cid)))
        print("duplicate_search:", json.dumps(duplicate_search(cid)))
        print("provider_history:", json.dumps(provider_history(r.provider)))
        print("clinical_consistency:", json.dumps(clinical_consistency(cid)))
        print("policy ->", policy_recommendation(r))
