# grep ca array from encoded dataset using 6d coords.
length = 128
import gemmi


def grep_bb_from_pdb(pdb_name):
    p = pdb_name

    st = gemmi.read_structure(p)
    st.setup_entities()
    try:
        polymer = st[0][-1].get_polymer()
        #print(polymer)
    except:
        print(f"{p.name} skipped - chain file corrupted")
        return
    if len(polymer) > 128:
        return
    sequence = gemmi.one_letter_code(polymer)
    if "X" in sequence:
        return
    backbone_crds = []
    all_atoms_crds = []
    missing=[]
    for idx,res in enumerate(polymer):
        all_atoms = {}
        for atom in res:
            if atom.name == "N":
                n_crd = atom.pos.tolist()
            elif atom.name == "CA":
                ca_crd = atom.pos.tolist()
            elif atom.name == "C":
                c_crd = atom.pos.tolist()
            elif atom.name == "O":
                o_crd = atom.pos.tolist()

            # For chi angle calculations
            all_atoms[atom.name] = atom.pos.tolist()

        # Check if backbone atoms are missing
        if all([i in all_atoms for i in ["N","CA","C","O"]]):
            backbone_crds.append([n_crd,ca_crd,c_crd,o_crd])
        else:
            missing.append(idx)
    return backbone_crds



import numpy as np


def grep_ca_array_from_datasets(d,i):
    bg_cords = d[i]["bb_coords"]
    lst = []
    for i in range(length):
        bg_one = bg_cords[i]
        bg_ca = bg_one[1]
        lst.append(bg_ca)
    return np.array(lst)


def grep_ca_array_from_bb_crds(bb_crds):
    bg_cords = bb_crds
    length = len(bb_crds)
    lst = []
    for i in range(length):
        bg_one = bg_cords[i]
        bg_ca = bg_one[1]
        lst.append(bg_ca)
    return np.array(lst)

def grep_ca_array_from_pdb(pdb_name):
    return grep_ca_array_from_bb_crds(grep_bb_from_pdb(pdb_name))




# align pdb.
# so the function take in sample, ref, and outputname.
import Bio.PDB
def align_for_docking(sample, ref,output_name):
    start_id = 1
    end_id = 128
    atoms_to_be_aligned = range(start_id, end_id + 1)
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)
    ref_structure = pdb_parser.get_structure("reference", ref)
    sample_structure = pdb_parser.get_structure("sample", sample)
    ref_model = ref_structure[0]
    sample_model = sample_structure[0]
    for ref_chain in ref_model:
        ref_atoms = []
        sample_atoms = []
        for ref_res in ref_chain:
        # Check if residue number ( .get_id() ) is in the list
            if ref_res.get_id()[1] in atoms_to_be_aligned:
        # Append CA atom to list
                ref_atoms.append(ref_res['CA'])

    for sample_chain in sample_model:
        for sample_res in sample_chain:
            if sample_res.get_id()[1] in atoms_to_be_aligned:
                sample_atoms.append(sample_res['CA'])

    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms, sample_atoms)
    super_imposer.apply(sample_model.get_atoms())
    io = Bio.PDB.PDBIO()
    io.set_structure(sample_structure)
    io.save(output_name)




def rmsd_numpy(loop1, loop2):
    """ Simple rmsd calculation for numpy arrays.
    """
    return np.sqrt(np.mean(((loop1 - loop2) ** 2).sum(-1)))



from Bio.PDB import Superimposer, PDBParser, PDBIO,MMCIFParser

def superpose_pdb2(pdb1, pdb2):
    # Load the two PDB files
    parser = PDBParser(QUIET=True)

    pdb1 = parser.get_structure('pdb1', pdb1)
    #pdb2 = parser.get_structure('pdb2', pdb2)

    if pdb2[-1] == "f":
        parser2 = MMCIFParser(QUIET=True)
        pdb2 = parser2.get_structure('pdb2', pdb2)
        #print("cif")

    else: # pdb
        pdb2 = parser.get_structure('pdb2', pdb2)

    # Extract the mainchain atoms (N, CA, C) from both structures, excluding the last chain
    atoms1 = []
    atoms2 = []

    for model1, model2 in zip(pdb1, pdb2):
        chains1 = list(model1.get_chains())[:-1]  # Exclude last chain
        chains2 = list(model2.get_chains())[:-1]  # Exclude last chain

        for chain1, chain2 in zip(chains1, chains2):
            for residue1, residue2 in zip(chain1, chain2):
                for atom1, atom2 in zip(residue1, residue2):
                    if atom1.name in ['N', 'CA', 'C']:
                        atoms1.append(atom1)
                    if atom2.name in ['N', 'CA', 'C']:
                        atoms2.append(atom2)

    # Superimpose the second structure onto the first one
    superimposer = Superimposer()
    superimposer.set_atoms(atoms1, atoms2)
    superimposer.apply(pdb2.get_atoms())

    # Save the superimposed structure as a new PDB file
    io = PDBIO()
    io.set_structure(pdb2)
    io.save('pdb2_superimposed.pdb')



def get_rmsd_for_loop(g_path,pdb_id):
    superpose_pdb2(f'ground_truth/{pdb_id}_model_0_pdb.pdb',g_path)
    a,b = grep_ca_array_from_pdb(f'ground_truth/{pdb_id}_model_0_pdb.pdb'),grep_ca_array_from_pdb("pdb2_superimposed.pdb")
    #aa, bb = grep_by_pos(a,peptide_hotspots_lst),grep_by_pos(b,peptide_hotspots_lst)
    #print(a.shape,b.shape)
    rmsd = rmsd_numpy(a, b)
    return rmsd



import json
def get_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as json_file:
        try:
            # Load the JSON file content
            content = json.load(json_file)
            # Use the file name without extension as the dictionary key
            return content

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file_name}: {e}")







import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa
import sys

def get_pep_coords_pdb(pdb_file):
    # this should be the only part to change if doing pdb. you should only need to change cif path to pdb path.

    if pdb_file[-1] == "f":
        parser = MMCIFParser(QUIET=True)
        #pdb2 = parser2.get_structure('pdb2', pdb2)
        #print("cif")

    else:
        parser = PDBParser(QUIET=True)# pdb
        #pdb2 = parser.get_structure('pdb2', pdb2)


    #parser = PDBParser(QUIET=True)  # Suppress warnings
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



def cal_interating_residues_ratios35(pdb_file):
    peptide_coords, receptor_coords, peptide_residues = get_pep_coords_pdb(pdb_file)
    num_interacting_residues = updated_loss_by_contact_score_times_iplddt(peptide_coords, receptor_coords, peptide_residues, 3.5)
    peptide_length = len(set(peptide_residues))
    if num_interacting_residues < 1:
        num_interacting_residues = 0.5
    return num_interacting_residues / peptide_length


def get_pep_length(pdb_file):
    peptide_coords, receptor_coords, peptide_residues = get_pep_coords_pdb(pdb_file)
    num_interacting_residues = updated_loss_by_contact_score_times_iplddt(peptide_coords, receptor_coords, peptide_residues, cutoff=4)
    peptide_length = len(set(peptide_residues))
    return peptide_length


f_name_regarding_pdb = '''design_results_4gux_contact_loss_1000_higher_temp_T_0.5  design_results_6bvh_contact_loss_1000_higher_temp_T_0.5  design_results_6jwm_contact_loss_1000_higher_temp_T_0.5
design_results_5lso_contact_loss_1000_higher_temp_T_0.5  design_results_6d3y_contact_loss_1000_higher_temp_T_0.5  design_results_6mrq_contact_loss_1000_higher_temp_T_0.5
design_results_5tu6_contact_loss_1000_higher_temp_T_0.5  design_results_6d3z_contact_loss_1000_higher_temp_T_0.5
design_results_5xn3_contact_loss_1000_higher_temp_T_0.5  design_results_6d40_contact_loss_1000_higher_temp_T_0.5
'''

pdb_f_names = ['5xn3',
 '6d3y',
 '6jwm',
 '6n87',
 '6bvh',
 '6d3z',
 '6mrq',
 '6q1u',
 '6d3x',
 '6d40',
 '6n7q']


pdb_f_names = ['4gux',
 '6bvh',
 '6jwm',
 '5lso',
 '6d3y',
 '6mrq',
 '5tu6',
 '6d3z',
 '5xn3',
 '6d40']


f_name_regarding_pdb = f_name_regarding_pdb.split()

af_name = "mar05" #"v4_all_results_on_higher_temp_T_0.5" #_all_17/design_results_fasta_to_yaml" #"mar05"

json_data = {}

def cal_min_distance_for_evobind2(peptide_coords, receptor_coords):
    """
    From Evobind2: https://github.com/patrickbryant1/EvoBind/blob/4389c148a1d0852425b6786b39dd99eddd996a16/src/mc_design.py#L192 

    Args:
        peptide_coords (np.ndarray): The coordinates of the peptide atoms
        receptor_coords (np.ndarray): The coordinates of the receptor atoms
        receptor_if_pos (list): The indices of the receptor interface residues (target hotspot residues)
    
    Returns:
        float: The mean distance between the peptide atoms and the closest receptor interface atom
    """
    #receptor_if_pos = np.array([int(res) for res in receptor_if_pos])

    #Calc 2-norm - distance between peptide and interface
    mat = np.append(peptide_coords,receptor_coords[:],axis=0)
    a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    l1 = len(peptide_coords)
    #Get interface
    contact_dists = dists[:l1,l1:] #first dimension = peptide, second = receptor

    #Get the closest atom-atom distances across the receptor interface residues.
    closest_dists_peptide = contact_dists[np.arange(contact_dists.shape[0]),np.argmin(contact_dists,axis=1)]

    return closest_dists_peptide.mean()



def get_evobind2_dis(pdb_file):
    peptide_coords, receptor_coords, peptide_residues = get_pep_coords_pdb(pdb_file)
    dis = cal_min_distance_for_evobind2(peptide_coords, receptor_coords)
    return dis



import numpy as np

import sys

#name = sys.argv[1]

def get_pep_plddt(name,pep_len):
    b = np.load(name)

    #print(b["plddt"])

    #print(b["plddt"].shape)

    #print(b["plddt"][-10:])

    return sum((b["plddt"][-pep_len:])/pep_len)

#plddt_4gux_model_0.npz


def get_proteinmpnn_scores(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    # Reverse search for the last metadata line
    for line in reversed(lines):
        if line.startswith(">"):
            parts = line.strip().split(", ")
            score = float([p.split("=")[1] for p in parts if "score=" in p][0])
            global_score = float([p.split("=")[1] for p in parts if "global_score=" in p][0])
            return score, global_score

    return None, None





#for k in range(1,81):
import os
for pdb_id in pdb_f_names:#f_name_regarding_pdb:
    #pdb_id = f_name.split("_")[2]
    inter_f_name = f"design_results_{pdb_id}_contact_loss_1000_higher_temp_T_0.5" #f"design_results_{pdb_id}_testing_evobind_dist_plddt_1000"  #"{pdb_id}_testing_evobind_dist_plddt_300" #"design_results_{pdb_id}_testing_evobind_dist_plddt_1000"

    path = f"{af_name}/"+inter_f_name+"/predictions/"
    
    for dir_name in os.listdir(path):
        try:
        #print(dir_name)
        #for dir_name in dirs:
            if dir_name!="seqs":
                path_g_json = path+dir_name+f"/{pdb_id}/confidence_{pdb_id}_model_0.json"
                json_key = pdb_id+"---"+dir_name
                #f = pre+d[k]+"/5lso/confidence_5lso_model_0.json"
            
                json_data[json_key] = get_json(path_g_json)
                k = json_key
                pdb_id,dir_name = k.split("---")
                json_data[k]["pdb_id"] = pdb_id
                path_g_pre = f"{af_name}/"+inter_f_name+"/predictions/"
                g_path = path_g_pre +dir_name+f"/{pdb_id}/{pdb_id}_model_0.pdb"
                g_npz_path  = path_g_pre +dir_name+f"/{pdb_id}/plddt_{pdb_id}_model_0.npz"
                
                g_fa_path = path_g_pre +dir_name+f"/{pdb_id}_model_0.fa"
                
                try:
                    score,global_score = get_proteinmpnn_scores(g_fa_path)
                except:
                    score,global_score = 0,0

                #print(g_path,pdb_id)
                rmsd = get_rmsd_for_loop(g_path,pdb_id)
                ratios = cal_interating_residues_ratios(g_path)
                ratios35 = cal_interating_residues_ratios35(g_path)
                #pep_plddt = get_pep_plddt(g_npz_path)

                pep_length = get_pep_length(g_path)
                pep_plddt = get_pep_plddt(g_npz_path,pep_length)
                json_data[k]["RMSD"] = rmsd
                json_data[k]["ratios"] = ratios
                json_data[k]["pep_length"] = pep_length
                dis =get_evobind2_dis(g_path)
                json_data[k]["min_dis_evo2"] = dis
                json_data[k]["ratios35"] = ratios35
                json_data[k]["pep_plddt"] = pep_plddt

                json_data[k]["proteinmpnn_score"] = score
                json_data[k]["proteinmpnn_global_score"] = global_score


                #print(json_data)

        except:
                #print(dir_name,path)
            pass
    #except:
        #pass


if "/" in af_name:
    af_name = af_name.split("/")[0]

import pandas as pd
df = pd.DataFrame.from_dict(json_data, orient='index')
df = df.applymap(lambda x: x if not isinstance(x, dict) else str(x))


df.to_csv(f"{af_name}_10testcases_res_updated_to_add_proteinmpnn_pep_plddt_ratios35.csv")
