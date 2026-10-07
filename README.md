# AI Payment Integrity Command Center

A prototype that scores hospital insurance claims for payment-integrity risk, lets an AI investigator agent build a case file for flagged claims, and shows the business value on a dashboard.

**Important:** all data is synthetic (3,048 generated claims, about 8% with planted problems). Savings figures rest on illustrative assumptions. Detection rates are measured against problems I planted, so they are not real-world accuracy.

## What it does
1. **Risk engine:** a transparent points-based checklist (duplicate claim, diagnosis/procedure mismatch, bill vs peer median, length of stay, hospital billing pattern, prior-claim frequency) gives each claim Low, Medium or High risk with plain-English reasons.
2. **Investigator Agent:** an LLM with four tools (peer benchmark, duplicate search, provider history, clinical consistency) chooses its checks, gathers evidence and writes a case file recommending approve, request documents or escalate. A fixed policy runs alongside it and flags disagreements. A human makes the final decision.
3. **Strategy dashboard:** claims flagged, suspicious amount, potential recovery, review cost, hours saved vs random sampling, with a threshold slider to show the cost/coverage trade-off.

## Files
- `index.html`: the finished app (dashboard and claims screens). Open it in a browser.
- `synthetic_claims.csv`: the generated dataset, including a hidden answer key (`anomaly_type`, `is_anomaly`).
- `make_claims.py`: generates the synthetic dataset.
- `risk_engine.py`: the scoring rules, with a test against the answer key.
- `agent_tools.py`: Python prototype of the agent's four tools.
- `template.html` and `build.py`: build the app by injecting the data into the template.

## Run it
```
pip install pandas numpy
python make_claims.py
python risk_engine.py
python build.py
```

## Note on the AI agent
The Investigator Agent uses Claude's built-in "ask Claude" feature, which only works when the app is opened inside claude.ai. Elsewhere (for example GitHub Pages) the app still works and falls back to a rule-based case file, labelled as such.

## Built with
Designed by me (workflow, rules, policy, assumptions); code written with Claude as a coding partner.
