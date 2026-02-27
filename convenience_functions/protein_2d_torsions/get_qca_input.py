"""Get the input data in QCA format,
adapted from https://github.com/openforcefield/protein-param-fit/blob/sage-2.1/validation/torsiondrive/1-curate-validation-torsiondrive-datasets.py
"""

import json
from collections import defaultdict
from pathlib import Path

import click
from openff.qcsubmit.results import TorsionDriveResultCollection
from openff.qcsubmit.results.filters import RecordStatusFilter
from openff.toolkit import Molecule
from qcportal import PortalClient
from qcportal.record_models import RecordStatusEnum


def get_qca_input(
    dataset_name: str, data_output_path: Path, names_output_path: Path
) -> None:

    client = PortalClient("api.qcarchive.molssi.org:443")

    protein_datasets = [client.get_dataset("Torsiondrive", dataset_name)]

    torsiondrive_dataset = TorsionDriveResultCollection.from_datasets(
        datasets=protein_datasets,
        spec_name="default",
    )

    # Hack to avoid filtering incomplete datasets from
    # TorsionDriveResultCollection.from_datasets()
    # from openff.qcsubmit.results import TorsionDriveResult
    # result_records = defaultdict(dict)
    # for dataset in protein_datasets:
    #    dataset.fetch_entries()
    #    for entry_name, spec_name, record in dataset.iterate_records(
    #        specification_names="default",  # status=RecordStatusEnum.complete
    #    ):
    #        entry = dataset.get_entry(entry_name)
    #        cmiles = entry.attributes[
    #            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
    #        ]
    #        inchi_key = entry.attributes.get("fixed_hydrogen_inchi_key")
    #        if inchi_key is None:
    #            tmp_mol = Molecule.from_mapped_smiles(
    #                cmiles, allow_undefined_stereo=True
    #            )
    #            inchi_key = tmp_mol.to_inchikey(fixed_hydrogens=True)
    #        td_rec = TorsionDriveResult(
    #            record_id=record.id, cmiles=cmiles, inchi_key=inchi_key
    #        )
    #        result_records[dataset._client.address][record.id] = td_rec
    # torsiondrive_dataset = TorsionDriveResultCollection(
    #    entries={
    #        address: [*entries.values()]
    #        for address, entries in result_records.items()
    #    }
    # )

    # Filter incomplete TorsionDrive results
    torsiondrive_dataset = torsiondrive_dataset.filter(
        RecordStatusFilter(status=RecordStatusEnum.complete)
    )

    # Write dict of IDs to names to file
    dataset_names = dict()
    for dataset in protein_datasets:
        dataset.fetch_entries()
        for entry_name, spec_name, record in dataset.iterate_records(
            specification_names="default",
        ):
            dataset_names[record.id] = entry_name

    names_output_path.parent.mkdir(parents=True, exist_ok=True)
    names_output_path.write_text(json.dumps(dataset_names, indent=4))

    data_output_path.parent.mkdir(parents=True, exist_ok=True)
    data_output_path.write_text(torsiondrive_dataset.json(indent=4))
