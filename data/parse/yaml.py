from pathlib import Path

import yaml
from rdkit.Chem.rdchem import Mol

from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.types import Target


def parse_yaml(path: Path, ccd: dict[str, Mol]) -> Target:
    """Parse a Boltz input yaml / json.

    The input file should be a yaml file with the following format:

    sequences:
        - protein:
            id: A
            sequence: "MADQLTEEQIAEFKEAFSLF"
        - protein:
            id: [B, C]
            sequence: "AKLSILPWGHC"
        - rna:
            id: D
            sequence: "GCAUAGC"
        - ligand:
            id: E
            smiles: "CC1=CC=CC=C1"
        - ligand:
            id: [F, G]
            ccd: []
    constraints:
        - bond:
            atom1: [A, 1, CA]
            atom2: [A, 2, N]
        - pocket:
            binder: E
            contacts: [[B, 1], [B, 2]]
    version: 1

    Parameters
    ----------
    path : Path
        Path to the YAML input format.
    components : Dict
        Dictionary of CCD components.

    Returns
    -------
    Target
        The parsed target.

    """
    with path.open("r") as file:
        data = yaml.safe_load(file)

    name = path.stem
    return parse_boltz_schema(name, data, ccd)

def parse_yaml_update_seq(path: Path, ccd: dict[str, Mol], new_seq) -> Target:
    """Parse a Boltz input yaml / json.

    The input file should be a yaml file with the following format:

    sequences:
        - protein:
            id: A
            sequence: "MADQLTEEQIAEFKEAFSLF"
        - protein:
            id: [B, C]
            sequence: "AKLSILPWGHC"
        - rna:
            id: D
            sequence: "GCAUAGC"
        - ligand:
            id: E
            smiles: "CC1=CC=CC=C1"
        - ligand:
            id: [F, G]
            ccd: []
    constraints:
        - bond:
            atom1: [A, 1, CA]
            atom2: [A, 2, N]
        - pocket:
            binder: E
            contacts: [[B, 1], [B, 2]]
    version: 1

    Parameters
    ----------
    path : Path
        Path to the YAML input format.
    components : Dict
        Dictionary of CCD components.

    Returns
    -------
    Target
        The parsed target.

    """
    with path.open("r") as file:
        data = yaml.safe_load(file)

        # Replacing the last sequence with the new sequence
        entities = data["sequences"]
        prot_count = 0
        last_prot_entity_idx = 0

        # Find the correct cyclic peptide entity
        for idx, entity in enumerate(entities):
            entity_type = list(entity.keys())[0]
            if entity_type == "protein":
                prot_count += 1
                last_prot_entity_idx = idx
        
       # Replacing the last sequence with the new sequence
        data["sequences"][last_prot_entity_idx]["protein"]["sequence"] = new_seq

    name = path.stem
    return parse_boltz_schema(name, data, ccd)