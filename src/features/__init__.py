"""
features/ — P3 feature extractors.

Pure, unit-tested functions that each return a normalized score in [0, 1]
for a single candidate, reading only the candidate dict + cfg. No side
effects, no I/O. Assembled in P4 by scoring.py into fit_score =
w_role*s_role + w_skill*s_skill + w_exp*s_exp + w_edu*s_edu + w_loc*s_loc.
"""
