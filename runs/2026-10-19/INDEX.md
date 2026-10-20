# 19 October 2026

- [BF16 override service check](uncensored-bf16/service-check.json): two DGX Spark nodes, the existing CUDA image with a local dense-override source patch, 1,024-token prefill chunks, 1M configured context, four slots, five MTP drafts, and a 1.5 GiB reserve. Real 8K/64K requests and four concurrent replies passed. Driver allocation warnings remained. This is not a speed A/B, a full-1M test, or qualification of the later PR head.
