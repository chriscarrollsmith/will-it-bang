/workspace/.venv/lib/python3.12/site-packages/scipy/stats/_stats_py.py:5445: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  warnings.warn(stats.ConstantInputWarning(warn_msg))
/workspace/.venv/lib/python3.12/site-packages/scipy/stats/_stats_py.py:5445: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  warnings.warn(stats.ConstantInputWarning(warn_msg))
/workspace/.venv/lib/python3.12/site-packages/scipy/stats/_stats_py.py:5445: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  warnings.warn(stats.ConstantInputWarning(warn_msg))
/workspace/.venv/lib/python3.12/site-packages/scipy/stats/_stats_py.py:5445: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  warnings.warn(stats.ConstantInputWarning(warn_msg))
/workspace/.venv/lib/python3.12/site-packages/scipy/stats/_stats_py.py:5445: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  warnings.warn(stats.ConstantInputWarning(warn_msg))
/workspace/.venv/lib/python3.12/site-packages/scipy/stats/_stats_py.py:5445: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  warnings.warn(stats.ConstantInputWarning(warn_msg))
/workspace/.venv/lib/python3.12/site-packages/scipy/stats/_stats_py.py:5445: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  warnings.warn(stats.ConstantInputWarning(warn_msg))
/workspace/.venv/lib/python3.12/site-packages/scipy/stats/_stats_py.py:5445: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  warnings.warn(stats.ConstantInputWarning(warn_msg))
# v49 parent-chain smoke

rows=110,625  reply_chain=25.0%  quote_chain=2.8%
among_reply reply_chain=31.1%
among_quote quote_chain=19.7%

| feature | global ρ(rate) | among-reply ρ | among-NR ρ | nonzero |
|---|---:|---:|---:|---:|
| has_reply_chain_parent | -0.080 | -0.016 | nan | 25.0% |
| reply_chain_emb_cos | -0.075 | -0.015 | nan | 25.0% |
| reply_chain_parent_cos | -0.079 | -0.017 | nan | 25.0% |
| reply_chain_replies | -0.078 | -0.023 | nan | 19.7% |
| reply_chain_age_hours | -0.079 | -0.014 | nan | 25.0% |
| reply_chain_same_author | -0.058 | -0.019 | nan | 11.6% |
| reply_chain_via_reply | -0.095 | -0.045 | nan | 19.2% |
| reply_chain_via_quote | 0.011 | 0.047 | nan | 5.8% |
| has_quote_chain_parent | 0.091 | 0.019 | 0.076 | 2.8% |
| quote_chain_emb_cos | 0.084 | 0.016 | 0.073 | 2.8% |
| quote_chain_parent_cos | 0.088 | 0.017 | 0.072 | 2.8% |
| quote_chain_replies | 0.079 | 0.018 | 0.065 | 2.3% |
| quote_chain_age_hours | 0.091 | 0.019 | 0.077 | 2.8% |

Among reply+chain rows: 27,624
  reply_chain_emb_cos: ρ=-0.009
  reply_chain_parent_cos: ρ=-0.010
  reply_chain_replies: ρ=0.013
  reply_chain_age_hours: ρ=0.008
