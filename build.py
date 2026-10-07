import pandas as pd, json, re
df = pd.read_csv("synthetic_claims.csv", keep_default_na=False)
rows = [[r.claim_id, r.claim_date, r.provider, r.procedure, r.diagnosis, int(r.patient_age), int(r.los_days), int(r.billed_amount), int(r.prior_claims), r.patient_id, r.anomaly_type] for r in df.itertuples()]
t = open("template.html").read()
assert "/*DATA*/[]" in t
open("index.html","w").write(t.replace("/*DATA*/[]", json.dumps(rows, separators=(",",":"))))
