# presto-benchmarking
[`presto`](https://github.com/cole-group/presto/tree/devel) is a tool for fitting bespoke SMIRNOFF force fields for your molecule(s) of interest. This repo contains a `Snakemake` workflow for benchmarking `presto`.

To rerun the workflow, [install `pixi`](https://pixi.prefix.dev/latest/installation/) and run:
```bash
git clone https://github.com/fjclark/presto-benchmarking.git
cd presto-benchmarking
pixi run snakemake --cores all
```

## Running on a SLURM cluster

Install the SLURM executor plugin (already listed in `pyproject.toml`, so `pixi install` handles this):
```bash
pixi install
```

Create a workflow-specific profile (the `profiles/` directory is git-ignored):
```bash
mkdir -p profiles/default
cat > profiles/default/config.yaml << 'EOF'
executor: slurm
jobs: 100
default-resources:
  mem_mb: 4000
  runtime: 60          # minutes
  slurm_partition: ""  # set your partition, e.g. "gpu"
  slurm_account: ""    # set your account if required
latency-wait: 60
rerun-incomplete: true
EOF
```

Snakemake automatically picks up `profiles/default` as the default profile, so you can simply run:
```bash
pixi run snakemake
```

The lightweight rules (`split_tnet500_input`, `create_combined_force_field`) run on the head node.
