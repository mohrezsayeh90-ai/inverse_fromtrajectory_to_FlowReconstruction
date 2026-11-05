# Experimental weights

There is currently **no accepted model** at `models/checkpoint.npz`.

Each directory in `candidates/` contains an actual trained checkpoint and its matching evaluation report. The `all_gates_passed` value is deliberately preserved, including failures. Use `--allow-unvalidated` only for explicitly provisional inspection.

| Candidate | Step | Encoder | Accepted |
|---|---:|---|---|
| gaussian | 50,000 | pooled | false |
| oseen | 100,000 | pooled | false |
| moments | 30,000 | pooled plus moments | false |
| sensor | 5,000 | direct sensors | false |

See [measured comparisons](../reports/experiments/README.md) and [the completed-run review](../reports/COMPLETED_RUN_REVIEW.md). A passing report is bound to both the checkpoint hash and the implementation signature; none of these weights has a universal accuracy guarantee.
