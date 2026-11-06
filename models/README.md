# Experimental weights

There is currently **no accepted model** at `models/checkpoint.npz`.

Each directory in `candidates/` contains an actual trained checkpoint and its matching evaluation report. The `all_gates_passed` value is deliberately preserved, including failures. Use `--allow-unvalidated` only for explicitly provisional inspection.

| Candidate | Step | Encoder | Accepted |
|---|---:|---|---|
| gaussian | 50,000 | pooled | false |
| oseen | 100,000 | pooled | false |
| moments | 30,000 | pooled plus moments | false |
| sensor | 5,000 | direct sensors | false |
| pressure | 5,000 | separate scalar pressure/stream operators, direct sensors | false |
| pressure_long | 94,000 selected from completed 100,000-step run | separate scalar pressure/stream operators, direct sensors | false |

Both the last 100,000-step weights and the selected 94,000-step weights are preserved in the [complete original output](../reports/experiments/pressure_long/output/). See the [full review](../reports/experiments/pressure_long/REVIEW.md).

See [measured comparisons](../reports/experiments/README.md) and [the completed-run review](../reports/COMPLETED_RUN_REVIEW.md). A passing report is bound to both the checkpoint hash and the implementation signature; none of these weights has a universal accuracy guarantee.
