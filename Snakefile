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


rule all:
    input:
        "benchmarking/tnet500/output/default/combined_force_field.offxml",


############ General Rules #############

rule run_presto:
    input:
        smiles_file="benchmarking/{dataset}/input/{dataset_type}/smiles/{molecule}.smi",
        config_file="configs/{config_name}.yaml",
    output:
        directory("benchmarking/{dataset}/output/{config_name}/{molecule}/bespoke_force_field.offxml"),
    run:
        run_presto(
            config_path=Path(input.config_file),
            smiles_path=Path(input.smiles_file),
            output_dir=Path(output[0]),
        )

rule create_combined_force_field:
    input:
        force_fields=directory("benchmarking/{dataset}/output/{config_name}/*/bespoke_force_field.offxml"),
    output:
        "benchmarking/{dataset}/output/{config_name}/combined_force_field.offxml",
    run:
        ff_to_combine_paths = {
            ff_path.stem: ff_path for ff_path in Path(input.force_fields[0]).parent.glob("*/bespoke_force_field.offxml")
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
        validation_set_dir=directory("benchmarking/tnet500/input/validation_set"),
        validation_set_json="benchmarking/tnet500/input/validation_set/validation_set.json",
        validation_set_smiles=directory("benchmarking/tnet500/input/validation_set/smiles"),
        test_set_dir=directory("benchmarking/tnet500/input/test_set"),
        test_set_json="benchmarking/tnet500/input/test_set/test_set.json",
        test_set_smiles=directory("benchmarking/tnet500/input/test_set/smiles"),
    run:
        create_validation_and_test_sets(
            input_json_path=Path(input[0]),
            frac_test=TNET_500_FRAC_TEST,
            seed=RANDOM_SEED,
            validation_output_path=Path(output.validation_set_dir),
            test_output_path=Path(output.test_set_dir),
        )
