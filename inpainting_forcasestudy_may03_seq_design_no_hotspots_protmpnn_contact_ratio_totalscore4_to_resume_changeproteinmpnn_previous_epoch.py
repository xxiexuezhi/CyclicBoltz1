
import random
import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import torch
from pytorch_lightning import Trainer, seed_everything
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.parse.fasta import parse_fasta, parse_fasta_update_seq
from boltz.data.parse.yaml import parse_yaml, parse_yaml_update_seq
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data import const
from boltz.data.types import MSA, Manifest, Record, Structure, Interface
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1
from dataclasses import asdict, replace
import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa
import sys
import copy
import json
import os
import subprocess




# Default Parameters
#DATA_PATH = Path("/Users/christinali/boltz_testing/mcmc/input_files/diff_initial_pep_sequence/1ssc_input_14_no_hotspots.yaml")
#OUT_DIR = Path("./design_results_1ssc_14_testing_evobind_dist_plddt_10_2")

# To run: python script.py 1ssc_input_14_no_hotspots.yaml 
# Note: Put the input folder in the same folder as the script --> just double check --> don't use absolute path

DATA_PATH = Path(sys.argv[1]) # default to be 1ssc_input_14_no_hotspots 
f_name = sys.argv[1].split(".")[0]

MAX_ITERATIONS = 1000  # Number of optimization rounds
OUT_DIR = Path(f"./case_study_for_mdm2_1ycr_head_to_tail_design/inpaintdesign_results_{f_name}_contact_loss_{MAX_ITERATIONS}_higher_temp_T_0.5")

CACHE_DIR = Path("~/.boltz").expanduser()
CCD_DIR = CACHE_DIR / "ccd.pkl"
CHECKPOINT_PATH = None  # If None, will use default model
DEVICES = 1
ACCELERATOR = "cpu"  # Choices: "gpu", "cpu"
MUTATION_RATE = 0.5  # Mutation probability per residue
SEED = 22 #12  # Random seed for reproducibility
PROTMPNN_SEED = 17  # Seed for ProteinMPNN
PROTMPNN_SAMPLING_TEMP = 0.5  # Sampling temperature for ProteinMPNN
all_chain_ids = []
use_msa_server = False,
msa_server_url = "https://api.colabfold.com"
msa_pairing_strategy = "greedy"
recycling_steps = 3
sampling_steps = 200
diffusion_samples = 2 # update here for diffusion_samples into 5 on Mar 14. try to see some clear trend.
write_full_pae = False
write_full_pde = False
step_scale = 1.638
new_sequence = None # New mutated sequence
update_seq = False  # Whether or not to update the sequence
output_format = "pdb"
#output_format: Literal["pdb", "mmcif"] = "mmcif"

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MODEL_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"

@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""
    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


def download(cache: Path) -> None:
    """Download the required model checkpoint if not present."""
    model_path = cache / "boltz1_conf.ckpt"
    if not model_path.exists():
        print(f"Downloading Boltz1 model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, str(model_path))


def mutate_sequence(sequence: str, mutation_rate: float) -> str:
    """Randomly mutates a protein sequence with a given mutation probability."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq_list = list(sequence)

    for i in range(len(seq_list)):
        if random.random() < mutation_rate:  # Apply mutation with given probability
            seq_list[i] = random.choice(amino_acids)  # Replace with random AA
    
    return "".join(seq_list)

# Copied from main.py (boltz)
# Both compute_msa and process_inputs 
def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.

    """
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))

def process_inputs(  # noqa: C901, PLR0912, PLR0915
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 4096,
    use_msa_server: bool = False,
    update_seq: bool = False,
    new_sequence: str = None
) -> None:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 4096.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.

    Returns
    -------
    BoltzProcessedInput
        The processed input data.
    
    target_id
        The target id.

    """
    #click.echo("Processing input data.")
    existing_records = None

    # Check if manifest exists at output path
    manifest_path = out_dir / "processed" / "manifest.json"
    if manifest_path.exists():
        print(f"Found a manifest file at output directory: {out_dir}")

        manifest: Manifest = Manifest.load(manifest_path)
        input_ids = [d.stem for d in data]
        existing_records, processed_ids = zip(
            *[
                (record, record.id)
                for record in manifest.records
                if record.id in input_ids
            ]
        )

        if isinstance(existing_records, tuple):
            existing_records = list(existing_records)

        # Check how many examples need to be processed
        missing = len(input_ids) - len(processed_ids)
        if not missing:
            print("All examples in data are processed. Updating the manifest")
            # Dump updated manifest
            updated_manifest = Manifest(existing_records)
            updated_manifest.dump(out_dir / "processed" / "manifest.json")
            return

        print(f"{missing} missing ids. Preprocessing these ids")
        missing_ids = list(set(input_ids).difference(set(processed_ids)))
        data = [d for d in data if d.stem in missing_ids]
        assert len(data) == len(missing_ids)

    # Create output directories
    msa_dir = out_dir / "msa"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    if existing_records is not None:
        print(f"Found {len(existing_records)} records. Adding them to records")

    # Parse input data
    records: list[Record] = existing_records if existing_records is not None else []


    path = data[0]
    try:
        # Parse data
        if path.suffix in (".fa", ".fas", ".fasta"):
            if update_seq:
                target = parse_fasta_update_seq(path, ccd, new_sequence)
            else:
                target = parse_fasta(path, ccd)
        elif path.suffix in (".yml", ".yaml"):
            if update_seq:
                print("update")
                target = parse_yaml_update_seq(path, ccd, new_sequence)
            else:
                print("not updating")
                target = parse_yaml(path, ccd)
        elif path.is_dir():
            msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
            raise RuntimeError(msg)
        else:
            msg = (
                f"Unable to parse filetype {path.suffix}, "
                "please provide a .fasta or .yaml file."
            )
            raise RuntimeError(msg)
        
        # Get the target receptor hotspot residues
        # Get all residue indices (for first chain) if no pocket is specified (same as evobind)
        if len(target.record.inference_options.pocket) > 0:
            receptor_hotspot_pos = [x[1] for x in target.record.inference_options.pocket]
        else:
            receptor_hotspot_pos = [res for res in range(target.record.chains[0].num_residues)]

        print("receptor_hotspot_pos: ", receptor_hotspot_pos)

        # Get target id
        target_id = target.record.id

        # Get all MSA ids and decide whether to generate MSA
        to_generate = {}
        prot_id = const.chain_type_ids["PROTEIN"]
        for chain in target.record.chains:
            # Add to generate list, assigning entity id
            if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                entity_id = chain.entity_id
                msa_id = f"{target_id}_{entity_id}"
                to_generate[msa_id] = target.sequences[entity_id]
                chain.msa_id = msa_dir / f"{msa_id}.csv"

            # We do not support msa generation for non-protein chains
            elif chain.msa_id == 0:
                chain.msa_id = -1

        # Getting chain ids
        all_chain_ids = [c.chain_name for c in target.record.chains]
        print("all_chain_ids: ", all_chain_ids)
        # Generate MSA
        if to_generate and not use_msa_server:
            msg = "Missing MSA's in input and --use_msa_server flag not set."
            raise RuntimeError(msg)

        if to_generate:
            msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
            print(msg)
            compute_msa(
                data=to_generate,
                target_id=target_id,
                msa_dir=msa_dir,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
            )

            # Parse MSA data
            msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
            msa_id_map = {}
            for msa_idx, msa_id in enumerate(msas):
                # Check that raw MSA exists
                msa_path = Path(msa_id)
                if not msa_path.exists():
                    msg = f"MSA file {msa_path} not found."
                    raise FileNotFoundError(msg)

                # Dump processed MSA
                processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
                msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
                if not processed.exists():
                    # Parse A3M
                    if msa_path.suffix == ".a3m":
                        msa: MSA = parse_a3m(
                            msa_path,
                            taxonomy=None,
                            max_seqs=max_msa_seqs,
                        )
                    elif msa_path.suffix == ".csv":
                        msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                    else:
                        msg = f"MSA file {msa_path} not supported, only a3m or csv."
                        raise RuntimeError(msg)

                    msa.dump(processed)

            # Modify records to point to processed MSA
            for c in target.record.chains:
                if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                    c.msa_id = msa_id_map[c.msa_id]

            # Keep record
            records.append(target.record)

            # Dump structure
            struct_path = structure_dir / f"{target.record.id}.npz"
            target.structure.dump(struct_path)

    except Exception as e:
        if len(data) > 1:
            print(f"Failed to process {path}. Skipping. Error: {e}.")
        else:
            raise e

    # Dump manifest
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")
    return target_id, receptor_hotspot_pos, all_chain_ids


# Simulated Annealing Parameters
INITIAL_TEMPERATURE = 2 #0.1
FINAL_TEMPERATURE = 0.00001
ANNEALING_RATE = 0.9 #0.99  # update to 0.5 on May03 # Decay factor per iteration

"""
# Previous parameters
INITIAL_TEMPERATURE = 5.0
FINAL_TEMPERATURE = 0.1
ANNEALING_RATE = 0.99  # Decay factor per iteration
"""

def acceptance_probability(old_score, new_score, temperature):
    """Compute acceptance probability using the Metropolis criterion."""
    if old_score < new_score:
        print("new score is higher than old score")
        return 1.0
    return np.exp((new_score - old_score) / temperature) #np.exp((old_score - new_score) / temperature)

def initialize_weights(peptide_length):
    '''Initialize sequence probabilities --> Taken from evobind2
    '''

    weights = np.random.gumbel(0,1,(peptide_length,20))
    weights = np.array([np.exp(weights[i])/np.sum(np.exp(weights[i])) for i in range(len(weights))])

    #Get the peptide sequence
    #Residue types
    restypes = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V' ])

    peptide_sequence = ''.join(restypes[[x for x in np.argmax(weights,axis=1)]])

    return weights, peptide_sequence

def mutate_sequence_wtih_hotspot(sequence: str, mutation_rate: float, ref_seq: str) -> str:
    """
    Randomly mutates a protein sequence with a given mutation probability.
    Only mutates positions corresponding to 'X' in the reference sequence.

    Args:
        sequence (str): The input protein sequence to be mutated.
        ref_seq (str): The reference sequence specifying positions to mutate ('X').
        mutation_rate (float): Probability of mutation for applicable positions.

    Returns:
        str: The mutated protein sequence.
    """
    if len(sequence) != len(ref_seq):
        raise ValueError("The sequence and ref_seq must have the same length.")

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq_list = list(sequence)

    for i in range(len(seq_list)):
        if ref_seq[i] == 'X':  # Only mutate positions marked as 'X' in ref_seq
            if random.random() < mutation_rate:  # Apply mutation with given probability
                seq_list[i] = random.choice(amino_acids)  # Replace with random AA
        

    return "".join(seq_list)

# Mutate sequence from evobind --> mutate one at a time
# https://github.com/patrickbryant1/EvoBind/blob/master/src/mutate_sequence.py 

def mutate_sequence_evo(peptide_sequence, sequence_scores):
    '''Mutate the amino acid sequence randomly
    '''

    restypes = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V' ])

    seqlen = len(peptide_sequence)
    searched_seqs = sequence_scores['sequence']
    #Mutate seq
    seeds = [peptide_sequence]
    #Go through a shuffled version of the positions and aas
    for seed in seeds:
        for pi in np.random.choice(np.arange(seqlen),seqlen, replace=False):
            for aa in np.random.choice(restypes,len(restypes), replace=False):
                new_seq = copy.deepcopy(seed)
                new_seq = new_seq[:pi]+aa+new_seq[pi+1:]
                if new_seq in searched_seqs:
                    continue
                else:
                    return new_seq

        seeds.append(new_seq)

def if_distance_loss_predict_file(peptide_coords, receptor_coords, receptor_if_pos):
    """
    From Evobind2: https://github.com/patrickbryant1/EvoBind/blob/4389c148a1d0852425b6786b39dd99eddd996a16/src/mc_design.py#L192 

    Args:
        peptide_coords (np.ndarray): The coordinates of the peptide atoms
        receptor_coords (np.ndarray): The coordinates of the receptor atoms
        receptor_if_pos (list): The indices of the receptor interface residues (target hotspot residues)
    
    Returns:
        float: The mean distance between the peptide atoms and the closest receptor interface atom
    """
    receptor_if_pos = np.array([int(res) for res in receptor_if_pos])

    #Calc 2-norm - distance between peptide and interface
    mat = np.append(peptide_coords,receptor_coords[receptor_if_pos],axis=0)
    a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    l1 = len(peptide_coords)
    #Get interface
    contact_dists = dists[:l1,l1:] #first dimension = peptide, second = receptor

    #Get the closest atom-atom distances across the receptor interface residues.
    closest_dists_peptide = contact_dists[np.arange(contact_dists.shape[0]),np.argmin(contact_dists,axis=1)]

    return closest_dists_peptide.mean()

def get_pep_coords_pdb_original(pdb_file):
    parser = PDBParser(QUIET=True)  # Suppress warnings
    structure = parser.get_structure("structure", pdb_file)

    chain_list = []
    model = structure[0]  # Only one model in the structure
    chain_list = [chain for chain in model]  # Collect all chains in a list

    if len(chain_list) < 1:
        raise ValueError("No chains or too few chains found in the structure")

    receptor_chain = chain_list[0]
    peptide_chain = chain_list[-1]

    receptor_coords = [atom.get_coord() for residue in receptor_chain for atom in residue]
    peptide_coords = [atom.get_coord() for residue in peptide_chain for atom in residue]

    receptor_coords = np.array(receptor_coords)
    peptide_coords = np.array(peptide_coords)

    return peptide_coords, receptor_coords

def get_pep_coords_cif_original(cif_file):
    parser = MMCIFParser(QUIET=True)  # Suppress warnings
    structure = parser.get_structure("structure", cif_file)

    chain_list = []
    model = structure[0]  # Only one model in the structure
    chain_list = [chain for chain in model]  # Collect all chains in a list

    if len(chain_list) < 1:
        raise ValueError("No chains or too few chains found in the structure")

    receptor_chain = chain_list[0]
    peptide_chain = chain_list[-1]

    receptor_coords = [atom.get_coord() for residue in receptor_chain for atom in residue]
    peptide_coords = [atom.get_coord() for residue in peptide_chain for atom in residue]

    receptor_coords = np.array(receptor_coords)
    peptide_coords = np.array(peptide_coords)

    return peptide_coords, receptor_coords

def get_pep_coords_pdb(pdb_file):
    # this should be the only part to change if doing pdb. you should only need to change cif path to pdb path. 
    parser = PDBParser(QUIET=True)  # Suppress warnings
    structure = parser.get_structure("structure", pdb_file)

    chain_list = []
    model = structure[0]  # Only one model in the structure
    chain_list = [chain for chain in model]  # Collect all chains in a list

    if len(chain_list) < 2:
        raise ValueError("No chains or too few chains found in the structure")

    peptide_chain = chain_list[-1]  # The peptide is always the last chain
    receptor_chains = chain_list[:-1]  # All preceding chains are receptor chains

    # Collect coordinates and residue identifiers for receptor and peptide
    receptor_coords = []
    receptor_residues = []
    for chain in receptor_chains:
        for residue in chain:
            if is_aa(residue, standard=True):
                residue_id = residue.get_id()
                for atom in residue:
                    receptor_coords.append(atom.get_coord())

    peptide_coords = []
    peptide_residues = []
    for residue in peptide_chain:
        if is_aa(residue, standard=True):
            residue_id = residue.get_id()
            peptide_residues.extend([residue_id] * len(residue))
            for atom in residue:
                peptide_coords.append(atom.get_coord())

    receptor_coords = np.array(receptor_coords)
    peptide_coords = np.array(peptide_coords)

    return peptide_coords, receptor_coords, peptide_residues

def get_pep_coords_cif(cif_file):
    # this should be the only part to change if doing pdb. you should only need to change cif path to pdb path. 
    parser = MMCIFParser(QUIET=True)  # Suppress warnings
    structure = parser.get_structure("structure", cif_file)

    chain_list = []
    model = structure[0]  # Only one model in the structure
    chain_list = [chain for chain in model]  # Collect all chains in a list

    if len(chain_list) < 2:
        raise ValueError("No chains or too few chains found in the structure")

    peptide_chain = chain_list[-1]  # The peptide is always the last chain
    receptor_chains = chain_list[:-1]  # All preceding chains are receptor chains

    # Collect coordinates and residue identifiers for receptor and peptide
    receptor_coords = []
    receptor_residues = []
    for chain in receptor_chains:
        for residue in chain:
            if is_aa(residue, standard=True):
                residue_id = residue.get_id()
                for atom in residue:
                    receptor_coords.append(atom.get_coord())

    peptide_coords = []
    peptide_residues = []
    for residue in peptide_chain:
        if is_aa(residue, standard=True):
            residue_id = residue.get_id()
            peptide_residues.extend([residue_id] * len(residue))
            for atom in residue:
                peptide_coords.append(atom.get_coord())

    receptor_coords = np.array(receptor_coords)
    peptide_coords = np.array(peptide_coords)

    return peptide_coords, receptor_coords, peptide_residues

# Functions to calculate the contact ratio
# *******************************************************************************************************************

def updated_loss_by_contact_score_times_iplddt(peptide_coords, receptor_coords, peptide_residues, cutoff=4):
    """
    Calculates the mean distance between peptide atoms and the closest receptor atoms,
    and returns the number of interacting peptide residues within the distance cutoff.

    Args:
        peptide_coords (np.ndarray): The coordinates of the peptide atoms.
        receptor_coords (np.ndarray): The coordinates of the receptor atoms.
        peptide_residues (list): The residue IDs of the peptide atoms.
        cutoff (float): Distance cutoff to define interactions (default is 6 Å).
    
    Returns:
        float: The mean distance between the peptide atoms and the closest receptor atoms.
        int: The number of unique interacting peptide residues within the cutoff.
    """

    # Calculate all pairwise distances between peptide and receptor atoms
    mat = np.concatenate([peptide_coords, receptor_coords])
    a_min_b = mat[:, np.newaxis, :] - mat[np.newaxis, :, :]
    dists = np.sqrt(np.sum(a_min_b ** 2, axis=-1))
    
    l1 = len(peptide_coords)
    contact_dists = dists[:l1, l1:]  # First dimension = peptide, second = receptor
    
    #print(contact_dists.shape)

    # Calculate the mean distance of the closest peptide-receptor atom pairs
    closest_dists_peptide = contact_dists[np.arange(contact_dists.shape[0]), np.argmin(contact_dists, axis=1)]
    #print(closest_dists_peptide)
    #mean_distance = closest_dists_peptide.mean()

    # Identify peptide residues with at least one atom within the distance cutoff
    interacting_residue_indices = np.where(contact_dists < cutoff)

    # Map atom indices to peptide residue IDs and get unique residues
    interacting_residues = {peptide_residues[index] for index in interacting_residue_indices[0]}
    
    #print(interacting_residues)
    num_interacting_residues = len(interacting_residues)
    # mean_distance,

    return num_interacting_residues


def cal_interating_residues_nums(pdb_file):
    peptide_coords, receptor_coords, peptide_residues = get_pep_coords_pdb(pdb_file)
    num_interacting_residues = updated_loss_by_contact_score_times_iplddt(peptide_coords, receptor_coords, peptide_residues, cutoff=4)
    if num_interacting_residues < 1: return 0.5
    return num_interacting_residues

def cal_interating_residues_ratios(pdb_file):
    peptide_coords, receptor_coords, peptide_residues = get_pep_coords_pdb(pdb_file)
    num_interacting_residues = updated_loss_by_contact_score_times_iplddt(peptide_coords, receptor_coords, peptide_residues, cutoff=4)
    peptide_length = len(set(peptide_residues))
    if num_interacting_residues < 1:
        num_interacting_residues = 0.5
    return num_interacting_residues / peptide_length

# *******************************************************************************************************************

# Mutate sequence with proteinmpnn
def mut_seq_proteinmpnn2(output_dir, pdb_path, peptide_chain, seed, cur_dir, prev_seq_list, sampling_temp):
    #print("mut_seq_proteinmpnn")
    #print("output dir", output_dir)
    #print("pdb path", pdb_path)
    #print("seed", seed)
    #print("sampling_temp", sampling_temp)

    # Run proteinmpnn to get the mutated sequence
    os.system(f"python ProteinMPNN/protein_mpnn_run.py --pdb_path {pdb_path} \
               --pdb_path_chains {peptide_chain} \
               --out_folder {output_dir} \
               --num_seq_per_target 10 \
               --sampling_temp {sampling_temp} \
               --seed {seed} \
               --batch_size 1")

    # Get the mutated sequence from the fasta output file and move to pdb dir
    fasta_file = output_dir / "seqs" / f"{f_name}_model_0.fa"
    new_fasta_file_name = cur_dir / f"{f_name}_model_0.fa"
    shutil.copy(fasta_file, new_fasta_file_name)
    #print("fasta file name: ", fasta_file)
    #print("new fasta file name: ", new_fasta_file_name)

    with open(new_fasta_file_name, "r") as f:
        all_lines = f.readlines()
        mutated_seq = all_lines[-1].strip()

    # Check if the mutated sequence has already been generated TODO need to check if this works
    if mutated_seq in prev_seq_list:
        seed += 1
        #if seed > 42 and sampling_temp < 0.5: # Note: sample function in protmpnn default temp is 1 --> but most use max of 0.5
        if seed > 42:
            sampling_temp += 0.1
        #print(f"Mutated sequence already generated. Trying again...\n New seed: {seed}, New sampling temp: {sampling_temp}")
        return mut_seq_proteinmpnn(output_dir, pdb_path, peptide_chain, seed, cur_dir, prev_seq_list, sampling_temp)

    #print("mutated seq", mutated_seq)
    return mutated_seq

# Create json1 file with chains to design with proteinmpnn --> store in predictions directory (note: not necessary rn)
def create_assigned_pdbs(output_file_name, all_chain_ids):
    print("create_assigned_pdbs")
    output_dir = OUT_DIR / "predictions" / 'assigned_pdbs.jsonl'

    # Create dictionary with the chain assignments
    chains_to_design = [[all_chain_ids[-1]], all_chain_ids[:-1]]
    chains_dict = {}
    chains_dict[output_file_name+"_model_0"] = chains_to_design

    # Save the dictionary as a json file
    with open(output_dir, 'w') as f:
        json.dump(chains_dict, f)


import sys
from pathlib import Path
import torch
import numpy as np
from ProteinMPNN.protein_mpnn_run import main as proteinmpnn_main
from argparse import Namespace

import shutil

#path = "proteinmpnn_test_pdb/"

#print(mut_seq_proteinmpnn(path,path+4gux_model_0.pdb,"B",42,path,[],50))
def mut_seq_proteinmpnn(output_dir, pdb_path, peptide_chain, seed, cur_dir, prev_seq_list, sampling_temp):
    output_dir = Path(output_dir)
    cur_dir = Path(cur_dir)

    args = Namespace(
        pdb_path=str(pdb_path),
        pdb_path_chains=peptide_chain,
        out_folder=str(output_dir),
        num_seq_per_target=10,
        sampling_temp=str(sampling_temp),
        seed=seed,
        batch_size=1,
        path_to_model_weights='',
        model_name="v_48_020",
        ca_only=False,
        use_soluble_model=False,
        save_score=0,
        save_probs=0,
        score_only=0,
        max_length=200000,
        omit_AAs='',
        suppress_print=1,
        pssm_multi=0.0,
        pssm_threshold=0.0,
        #pdb_path_chains=peptide_chain,
        jsonl_path='',
        chain_id_jsonl='',
        fixed_positions_jsonl='',
        #omit_AAs='',
        bias_AA_jsonl='',
        bias_by_res_jsonl='',
        omit_AA_jsonl='',
        tied_positions_jsonl='',
        #save_score=0,
        conditional_probs_only=0,
        unconditional_probs_only=0,
        #save_probs=0,
        path_to_fasta='',
        backbone_noise=0.0,
        pssm_jsonl='',
        pssm_log_odds_flag=0,
        pssm_bias_flag=0,
        #max_length=200000,
    )

    proteinmpnn_main(args)

    # Assume f_name is derived from pdb_path
    f_name = Path(pdb_path).stem

    fasta_file = output_dir / "seqs" / f"{f_name}.fa"
    new_fasta_file_name = Path(cur_dir) / f"{f_name}_model_0.fa"

    shutil.copy(fasta_file, new_fasta_file_name)

    with open(new_fasta_file_name, "r") as f:
        all_lines = f.readlines()
        mutated_seq = all_lines[-1].strip()

    # Check if the mutated sequence has already been generated
    if mutated_seq in prev_seq_list:
        seed += 1
        if seed > 42:
            sampling_temp = float(sampling_temp) + 0.1
        return mut_seq_proteinmpnn(output_dir, pdb_path, peptide_chain, seed, cur_dir, prev_seq_list, sampling_temp)

    return mutated_seq



def main(start_step=0,start_iter=0): ## given the nums if continues. else default 0
    """Main function to run the sequence optimization loop."""
    
    # Set seed for reproducibility
    seed_everything(SEED)

    # Ensure output directories exist
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download the model if not found
    download(CACHE_DIR)

    # Load model checkpoint
    checkpoint = CHECKPOINT_PATH if CHECKPOINT_PATH else CACHE_DIR / "boltz1_conf.ckpt"

    # Parse input sequence from FASTA
    fasta_path = DATA_PATH
    with open(CCD_DIR, "rb") as file:
        ccd_dict = pickle.load(file)
    records = parse_yaml(fasta_path, ccd=ccd_dict)
    cyc_pep_seq_idx = list(records.sequences.keys())[-1] # Getting the last sequence
    original_sequence = records.sequences[cyc_pep_seq_idx]
    best_sequence = original_sequence
    best_contact_ratio_loss = 0.0
    best_iteration = 1
    #print(original_sequence)

    
    temperature = INITIAL_TEMPERATURE
    
    for _ in range(start_iter):
        temperature *= ANNEALING_RATE
    
    #print(f"Starting sequence optimization for {MAX_ITERATIONS} iterations...")

    # storing the steps
    step = start_step
    accepted_prev = False
    scores = {'iteration': [], 'sequence': [], 'loss': [], 'confidence_score': [], 'iplddt,iptm,pde': [], 'best_sequence': "", 'best_iteration': 1, 'best_loss': 1000}

    #for iteration in range(MAX_ITERATIONS):
    for iteration in range(start_iter,MAX_ITERATIONS):
        # Create directory to store predictions
        cur_pred_dir = OUT_DIR / "predictions" / f"step_{step+1}_iteration_{iteration+1}"
        cur_pred_dir.mkdir(parents=True, exist_ok=True)

        #new_sequence = mutate_sequence(best_sequence, MUTATION_RATE)
        scores["iteration"].append(iteration+1)

        if iteration >= start_iter:
            update_seq = True # TODO can delete or change this later --> ensures that the sequence is updated

        # Mutate the sequence
        #new_sequence = mutate_sequence(best_sequence, MUTATION_RATE)
        #new_sequence = mutate_sequence_wtih_hotspot(best_sequence, MUTATION_RATE,ref_seq)
        if iteration == start_iter:
            #new_sequence = mutate_sequence(best_sequence, 1)
            new_sequence = initialize_weights(len(best_sequence))[1]
        else: 
            #new_sequence = mutate_sequence_evo(best_sequence, scores)
            if accepted_prev:
                prev_folder = f"step_{step}_iteration_{iteration}"
                PROTMPNN_SEED = 37
            else:
                prev_folder = f"step_{step}_iteration_{best_iteration}"
                PROTMPNN_SEED += 1

            #print("before protmpnn")
            #print("prev folder: ", prev_folder)
            #print("step: ", step+1)
            #print("iteration: ", iteration+1)
            protmpnn_results_dir = "." / OUT_DIR / "predictions"
            prev_pred_dir = "." / OUT_DIR / "predictions" / prev_folder / f"{f_name}" / f"{f_name}_model_0.pdb"
            new_sequence = mut_seq_proteinmpnn(protmpnn_results_dir, prev_pred_dir, all_chain_ids[-1], PROTMPNN_SEED, cur_pred_dir, scores["sequence"], PROTMPNN_SAMPLING_TEMP)
        #scores["sequence"].append(new_sequence)
        

        # same thing.mutate the sequence.
        new_sequence_lst = list(new_sequence)
        new_sequence_lst[7] = "L" # try 4 not working. change back to 7 # prev is 7
        new_sequence_lst[3] = "W"
        new_sequence_lst[17] = "F"
        new_sequence = ''.join(new_sequence_lst)
        scores["sequence"].append(new_sequence)


        #print("new_sequence: ", new_sequence)

        #print(f"Iteration {iteration+1}/{MAX_ITERATIONS}: Predicting contact_ratio_loss for mutated sequence")

        """******************************************************************************************************************************************"""
        # Process the inputs
        use_msa_server = True
        print(DATA_PATH, OUT_DIR,CCD_DIR,use_msa_server,msa_server_url,msa_pairing_strategy,update_seq,new_sequence)

        target_id_name, receptor_hotspot_pos, all_chain_ids = process_inputs(
            data=[DATA_PATH],
            out_dir=OUT_DIR,
            ccd_path=CCD_DIR,
            use_msa_server=use_msa_server,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=msa_pairing_strategy,
            update_seq=update_seq,
            new_sequence=new_sequence
        )

        # Create data module and load processed data
        manifest=Manifest.load(OUT_DIR/ "processed" / "manifest.json")
        data_module = BoltzInferenceDataModule(
            manifest=manifest,
            target_dir=OUT_DIR / "processed" / "structures",
            msa_dir=OUT_DIR / "processed" / "msa",
            num_workers=2,
        )

        # Create prediction writer
        pred_writer = BoltzWriter(
            data_dir=OUT_DIR / "processed" / "structures",
            output_dir=cur_pred_dir,
            output_format=output_format,
        )

        trainer = Trainer(
            accelerator=ACCELERATOR,
            devices=DEVICES,
            callbacks=[pred_writer],
            precision=32,
        )

        # Load Boltz1 model
        predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
        }
        diffusion_params = BoltzDiffusionParams()
        diffusion_params.step_scale = step_scale

        model = Boltz1.load_from_checkpoint(
            checkpoint, 
            strict=True,
            predict_args = predict_args, 
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params), 
            ema = False)
        model.eval()

        # Run prediction
        # TODO see if you can get gradients
        predictions = trainer.predict(model, datamodule=data_module, return_predictions=True)

        # Extract the coordinates
        #get_pep_rec_coords(predictions[0], target_id_name)

        if output_format == "mmcif":
            # Getting cif file name
            path = cur_pred_dir / f"{target_id_name}" / f"{target_id_name}_model_0.cif"
            #peptide_coords, receptor_coords, peptide_residues = get_pep_coords_cif(path)
        elif output_format == "pdb":
            # Getting cif file name
            path = cur_pred_dir / f"{target_id_name}" / f"{target_id_name}_model_0.pdb"
            #peptide_coords, receptor_coords, peptide_residues = get_pep_coords_pdb(path)

        else:
            raise ValueError("Output format must be either 'mmcif' or 'pdb'")

        # Calculate the contact ratio
        #contact_ratio = cal_interating_residues_ratios(path)

        # Extract confidence scores (iplddt, iptm, pde)
        confidence_score = predictions[0]["confidence_score"].mean().item()
        iplddt = predictions[0]["complex_iplddt"].mean().item()
        iptm = predictions[0]["iptm"].mean().item()
        ptm = predictions[0]["ptm"].mean().item()
        pde = predictions[0]["complex_pde"].mean().item()
        scores["confidence_score"].append(confidence_score)
        scores["iplddt,iptm,pde"].append((iplddt, iptm, pde))

        # Change the acceptance probability to use the contact ratio loss
        #contact_ratio_loss = contact_ratio * (iplddt * 0.8 + iptm * 0.1 + pde * 0.1)
        # update the loss below 
        contact_ratio_loss = (confidence_score * ptm) / pde 
        if iteration == start_iter:
            scores["best_sequence"] = new_sequence
            scores["best_loss"] = contact_ratio_loss
            scores["best_iteration"] = iteration + 1
            best_contact_ratio_loss = contact_ratio_loss
            best_iteration = iteration + 1
            best_sequence = new_sequence
            #print(f"Accepted new sequence with contact ratio loss {contact_ratio_loss:.2f} (T={temperature:.2f})")
            #print(f"New best iteration: {iteration+1}")
            step+=1
            accepted_prev = True

        else:
            acceptance_prob = acceptance_probability(best_contact_ratio_loss, contact_ratio_loss, temperature)
            
            generated_num = np.random.rand()
            #if np.random.rand() < acceptance_prob:
            #print("Generated num: ", generated_num)
            #print("Acceptance prob: ", acceptance_prob)
            if generated_num < acceptance_prob:
                #Create new best seq
                scores["best_sequence"] = new_sequence
                scores["best_loss"] = contact_ratio_loss
                scores["best_iteration"] = iteration + 1
                best_contact_ratio_loss = contact_ratio_loss
                best_iteration = iteration + 1
                best_sequence = new_sequence
                #print(f"Accepted new sequence with dist loss {contact_ratio_loss:.2f} (T={temperature:.2f})")
                #print(f"New best iteration: {iteration+1}")
                step+=1
                accepted_prev = True

            else:
                #print(f"Rejected new sequence (contact_ratio_loss {contact_ratio_loss:.2f}), keeping current best.")
                accepted_prev = False
                #shutil.rmtree(cur_pred_dir)
        

        scores["loss"].append(contact_ratio_loss)
        temperature *= ANNEALING_RATE  # Anneal temperature
        shutil.rmtree(OUT_DIR / "processed")

    best_sequence_path = OUT_DIR / "best_sequence.fasta"
    with open(best_sequence_path, "w") as f:
        f.write(f">Optimized_Sequence\n{best_sequence}\n")
        f.write(f"Final best contact_ratio_loss: {best_contact_ratio_loss:.2f}\n")
        f.write(f"Best iteration: iteration_{best_iteration}")
        f.write(f"\nAll scores: {scores}")

    print(f"Optimization complete! Best sequence saved to {best_sequence_path}")



import re
from pathlib import Path

def get_max_step_and_iteration(path):
    max_step = 0
    max_iteration = 0

    path = Path(path)  # Ensure it's a Path object
    if not path.is_dir():
        raise ValueError(f"Provided path {path} is not a directory")

    for folder in path.iterdir():
        if folder.is_dir():  # Ensure it's a directory
            match = re.match(r"step_(\d+)_iteration_(\d+)", folder.name)
            if match:
                step_num = int(match.group(1))
                iteration_num = int(match.group(2))
                max_step = max(max_step, step_num)
                max_iteration = max(max_iteration, iteration_num)

    return max_step, max_iteration


if __name__ == "__main__":

    predictions_dir = OUT_DIR / "predictions"
    try: 
        max_step, max_iteration = get_max_step_and_iteration(predictions_dir)
        print( max_step, max_iteration)
    except:
        max_step, max_iteration = 0,0
    main( max_step, max_iteration)
