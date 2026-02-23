# presto-benchmarking
[`presto`](https://github.com/cole-group/presto/tree/devel) is a tool for fitting bespoke SMIRNOFF force fields for your molecule(s) of interest. This repo contains a `Snakemake` workflow for benchmarking `presto`.

To rerun the workflow, [install `pixi`](https://pixi.prefix.dev/latest/installation/) and run:
```bash
git clone https://github.com/fjclark/presto-benchmarking.git
cd presto-benchmarking
pixi run snakemake --cores all
```
