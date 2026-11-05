# Test Suite

This directory contains development-time `pytest` checks. It is deliberately
outside `src/sparse_st_fgw/`, so tests are not imported by the model and are
not included in the installed Python package.

```text
test_config.py             fixed best-configuration checks
test_graphsage.py          H5AD-loaded spatial graph checks
test_sparse_ot.py          sparse support and GW cost checks
test_evaluation.py         saved-transport accuracy checks
test_workflow_contract.py  public script and label-isolation checks
```

Run the suite in the validated environment:

```bash
conda run -n go_lt_uot env PYTHONPATH=src pytest -q
```
