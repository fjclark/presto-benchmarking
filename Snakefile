from pathlib import Path
from typing import Any

from convenience_functions.get_qca_input import get_tnet_500_spice_lot_qca_input
from convenience_functions.split_qca_input import create_validation_and_test_sets
from convenience_functions.run_presto import run_presto
from convenience_functions.combine_ffs import combine_force_fields

RANDOM_SEED = 0
TNET_500_FRAC_TEST = 0.8  # 20 percent validation, 80 percent test


def smiles_dir_outputs(wildcards: Any, checkpoint_obj: Any, smiles_dir: str, output_pattern: str) -> list[str]:
    """Expand output_pattern over all .smi files in smiles_dir once checkpoint_obj is done."""
    checkpoint_obj.get()
    molecules = glob_wildcards(f"{smiles_dir}/{{molecule}}.smi").molecule
    return expand(output_pattern, molecule=molecules)


def validation_force_fields(wildcards: Any) -> list[str]:
    """Generic input function for create_combined_force_field.

    Infers the smiles directory from the dataset wildcard and resolves
    the per-molecule force field paths after the relevant checkpoint completes.
    """
    dataset = wildcards.dataset
    checkpoint_obj = checkpoints.split_tnet500_input if dataset == "tnet500" else None
    if checkpoint_obj is None:
        raise ValueError(f"No checkpoint registered for dataset '{dataset}'")
    return smiles_dir_outputs(
        wildcards,
        checkpoint_obj=checkpoint_obj,
        smiles_dir=f"benchmarking/{dataset}/input/{wildcards.dataset_type}/smiles",
        output_pattern=f"benchmarking/{dataset}/output/{wildcards.dataset_type}/{wildcards.config_name}/{{molecule}}/bespoke_force_field.offxml",
    )


rule all:
    input:
        "benchmarking/tnet500/output/validation/default/combined_force_field.offxml",


############ General Rules #############

rule run_presto:
    input:
        smiles_file="benchmarking/{dataset}/input/{dataset_type}/smiles/{molecule}.smi",
        config_file="configs/{config_name}.yaml",
    output:
        "benchmarking/{dataset}/output/{dataset_type}/{config_name}/{molecule}/bespoke_force_field.offxml",
    resources:
        mem_mb=8000,
        runtime=120,  # minutes
        slurm_partition="gpu-s_free",
        slurm_extra="--gpus-per-task=1",
    run:
        run_presto(
            config_path=Path(input.config_file),
            smiles_path=Path(input.smiles_file),
            output_dir=Path(output[0]),
        )

rule create_combined_force_field:
    input:
        force_fields=validation_force_fields,
    output:
        "benchmarking/{dataset}/output/{dataset_type}/{config_name}/combined_force_field.offxml",
    run:
        ff_to_combine_paths = {
            Path(ff_path).parent.name: Path(ff_path)
            for ff_path in input.force_fields
        }
        combine_force_fields(
            ff_to_combine_paths=ff_to_combine_paths,
            output_file=Path(output[0]),
        )

############ TNet 500 #############

rule get_tnet500_input:
    output:
        "benchmarking/tnet500/input/full_dataset.json"
    run:
        # get_qca_torsion_input(
        #     dataset_name="TorsionNet500 Re-optimization TorsionDrives v4.0",
        #     json_output_path=Path(output[0]),
        # )
        get_tnet_500_spice_lot_qca_input(json_output_path=Path(output[0]))

checkpoint split_tnet500_input:
    input:
        "benchmarking/tnet500/input/full_dataset.json"
    output:
        validation_set_dir=directory("benchmarking/tnet500/input/validation"),
        validation_set_json="benchmarking/tnet500/input/validation/validation.json",
        validation_set_smiles=directory("benchmarking/tnet500/input/validation/smiles"),
        test_set_dir=directory("benchmarking/tnet500/input/test"),
        test_set_json="benchmarking/tnet500/input/test/test.json",
        test_set_smiles=directory("benchmarking/tnet500/input/test/smiles"),
    run:
        create_validation_and_test_sets(
            input_json_path=Path(input[0]),
            frac_test=TNET_500_FRAC_TEST,
            seed=RANDOM_SEED,
            validation_output_path=Path(output.validation_set_dir),
            test_output_path=Path(output.test_set_dir),
        )
