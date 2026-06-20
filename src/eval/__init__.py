"""
eval/ — P5 proxy evaluation harness (EXECUTION_PLAN §5).

Local proxy evaluation: metrics (NDCG@10, NDCG@50, MAP, P@10,
composite), proxy labels (~50 hand-labeled tiers 0–5 plus adversarial
near-miss decoys for the §5 independence guard), and the
calibration driver.

Used to (a) regression-test every change, (b) lightly calibrate ≤4
macro knobs, and (c) decide what to submit. There is no leaderboard
and only 3 blind submissions — the proxy is the only signal we have.
"""
