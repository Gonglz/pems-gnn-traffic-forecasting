# Training And Acceleration

Summarizes the model-training workflow, GraphSAGE/NeighborSampler path, DDP sharding, and acceleration decisions.

## Notes

- The main workflow is data cleaning, graph construction, GraphSAGE training, DDP evaluation, and evidence-pack reporting.
- Large raw/intermediate PeMS arrays, checkpoints, and generated experiment outputs are intentionally excluded.
