"""Functionality to split QCArchive inputs into validation and test sets."""

import json
from pathlib import Path

import deepchem as dc
import loguru
import pandas as pd
from openff.toolkit import Molecule
from rdkit.Chem import Draw

JSON_FILE = "TNet500_minimal_dataset.json"
SMILES_NAME = "smiles.csv"
FRAC_TEST = 0.80  # We'll use the 20 % = 100 molecules for validation
SEED = 0
VALIDATION_OUTPUT_PATH = Path("validation_set")
TEST_OUTPUT_PATH = Path("test_set")

logger = loguru.logger


def load_smiles(json_file: str) -> pd.DataFrame:
    with open(json_file, "r") as f:
        data = json.load(f)

    smiles_list = [entry["mapped_smiles"] for entry in data["qm_torsions"]]
    id_list = [entry["id"] for entry in data["qm_torsions"]]
    torsion_idxs = [entry["dihedral_indices"] for entry in data["qm_torsions"]]

    return pd.DataFrame(
        {"id": id_list, "smiles": smiles_list, "torsion_idx": torsion_idxs}
    )


def save_individual_smiles_files(
    ids: list[int], smiles_df: pd.DataFrame, output_dir: Path
) -> None:
    """Save an individual .smi file for each molecule in the subset."""
    output_dir.mkdir(exist_ok=True)
    subset_df = smiles_df.iloc[ids]
    for _, row in subset_df.iterrows():
        id_ = row["id"]
        smiles = row["smiles"]
        output_file = output_dir / f"{id_}.smi"
        with open(output_file, "w") as f:
            f.write(smiles + "\n")


def split_dataset_maxmin(
    smiles_df: pd.DataFrame, frac_train: float, seed: int
) -> tuple[list[int], list[int]]:
    splitter = dc.splits.MaxMinSplitter()

    dc_dataset = dc.data.DiskDataset.from_numpy(
        X=smiles_df.id,
        ids=smiles_df.smiles,
    )

    # We will use the small "test" set as our validation set
    # and test on the large "train" set as the test set
    test_dataset, valid_dataset = splitter.train_test_split(
        dc_dataset,
        frac_train=frac_train,
        seed=seed,
    )
    return test_dataset.X, valid_dataset.X


def save_torsion_img(ids: list[int], smiles_df: pd.DataFrame, filename: Path) -> None:
    subset_df = smiles_df.iloc[ids]
    rdkit_mols = [
        Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True).to_rdkit()
        for smiles in subset_df["smiles"]
    ]

    img = Draw.MolsToGridImage(  # type: ignore
        rdkit_mols,
        highlightAtomLists=subset_df["torsion_idx"].tolist(),
        legends=[f"ID: {id_}" for id_ in subset_df["id"]],
        molsPerRow=5,
        subImgSize=(300, 300),
    )

    # Save to png
    img.save(filename)


def save_smiles(ids: list[int], smiles_df: pd.DataFrame, filename: Path) -> None:
    subset_df = smiles_df.iloc[ids]
    subset_df.to_csv(filename, index=False)


def save_sub_dataset(ids: list[int], json_file: Path, output_file: Path) -> None:
    with open(json_file, "r") as f:
        data = json.load(f)

    subset_data = {"qm_torsions": [data["qm_torsions"][i] for i in ids]}

    with open(output_file, "w") as f:
        json.dump(subset_data, f, indent=2)


def create_validation_and_test_sets(
    input_json_path: Path,
    frac_test: float,
    seed: int,
    validation_output_path: Path,
    test_output_path: Path,
) -> None:
    """Split the input dataset into validation and test sets and save them.

    This produces:

    - test_output_path/test_set.json: the test set in the same format as the input JSON file
    - test_output_path/test_set_torsions.png: a grid image of the test set torsions
    - test_output_path/smiles.csv: a CSV file containing the SMILES strings and IDs for the test set
    - validation_output_path/validation_set.json: the validation set in the same format as the input JSON file
    - validation_output_path/validation_set_torsions.png: a grid image of the validation set torsions
    - validation_output_path/smiles.csv: a CSV file containing the SMILES strings and IDs for the validation set

    Args:
        input_json_path: Path to the input JSON file containing the dataset.
        frac_test: Fraction of the dataset to use as the test set.
        seed: Random seed for reproducibility.
        validation_output_path: Directory to save the validation set outputs.
        test_output_path: Directory to save the test set outputs.
    """
    smiles_df = load_smiles(json_file=str(input_json_path))
    test_inds, valid_inds = split_dataset_maxmin(smiles_df, frac_test, seed)

    validation_output_path.mkdir(exist_ok=True)
    test_output_path.mkdir(exist_ok=True)

    save_torsion_img(
        valid_inds, smiles_df, validation_output_path / "validation_set_torsions.png"
    )
    save_smiles(valid_inds, smiles_df, validation_output_path / "smiles.csv")
    save_individual_smiles_files(
        valid_inds, smiles_df, validation_output_path / "smiles"
    )
    save_sub_dataset(
        valid_inds, input_json_path, validation_output_path / "validation_set.json"
    )

    save_torsion_img(test_inds, smiles_df, test_output_path / "test_set_torsions.png")
    save_smiles(test_inds, smiles_df, test_output_path / SMILES_NAME)
    save_individual_smiles_files(test_inds, smiles_df, test_output_path / "smiles")
    save_sub_dataset(test_inds, input_json_path, test_output_path / "test_set.json")

    logger.info(f"Validation set size: {len(valid_inds)} molecules")
    logger.info(f"Test set size: {len(test_inds)} molecules")
