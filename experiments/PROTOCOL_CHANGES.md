# Protocol Deviations

The protocol in `docs/04_EXPERIMENT_PROTOCOL.md` is pre-registered. Any change after
seeing results gets logged here with a date and a reason. Reviewers respect a logged
deviation and punish a silent one.

| Date | Change | Reason |
|---|---|---|
| 2026-08-11 | Stage A moves from synthetic to real sentence embeddings in Week 1 | Day-0 finding F6: cosine scores 1.000 on every synthetic condition, so there is no headroom to measure anything |
| 2026-08-11 | DegradeBench D5 is no longer the G1 discriminator | Day-0 finding F4: single-state descent is structurally *worse* than top-k for multi-target queries. D5 now motivates residual peeling instead |
