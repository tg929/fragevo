from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import rdmolops
# import datamol as dm
import numpy as np
from rdkit.Chem import BRICS
import utils.sascorer as sascorer
from rdkit.Chem import QED


def get_qed(mol):
    return QED.qed(mol)


# def get_sa(mol):
#     return sascorer.calculateScore(mol)

def get_sa(mol):
    return (10 - sascorer.calculateScore(mol)) / 9


def get_morgan_fingerprint(mol, radius=2, nbits=2048):
    """
    获取分子的 Morgan 指纹。
    如果解析失败（mol=None），返回 None。
    """
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return fp


def is_too_similar_to_children(new_smiles_fp, children_smiles_fps, threshold=0.8):
    """
    判断 new_smiles 是否与 children_smiles 列表中的任何一个 SMILES
    在 Tanimoto 相似度上超过 threshold。

    - new_smiles:  新生成的 SMILES 字符串
    - children_smiles:  当前节点所有子节点对应的 SMILES 列表
    - threshold:   相似度阈值，超过此值则认为“过于相似”

    Return:
      True  -> 如果过于相似（或无效 SMILES），不接受
      False -> 如果不相似或可接受
    """
    new_fp = new_smiles_fp
    if new_fp is None:
        # 若解析失败，可视为不可用（或你想要的其他处理方式）
        return True

    for fp in children_smiles_fps:
        if fp is None:
            continue
        tanimoto = TanimotoSimilarity(new_fp, fp)
        if tanimoto >= threshold:
            return True
    return False


# def sentence2smiles(smi):
#     smi = smi.replace("[SEP]", "").replace("[BOS]", "").replace("[EOS]", "").replace(" ", "")
#
#     mol = dm.to_mol(smi)
#     du = dm.from_smarts("[$([#0]!-!:*);$([#0;D1])]")
#     out = Chem.ReplaceSubstructs(mol, du, dm.to_mol("C"), True)[0]
#     mol = dm.remove_dummies(out)
#     mol = dm.remove_hs(mol, update_explicit_count=True)
#     mol = dm.standardize_mol(mol)
#     mol = dm.canonical_tautomer(mol)
#     out_smi = dm.to_smiles(mol)
#     out_smi = dm.standardize_smiles(out_smi)
#     return mol, out_smi


dummy = Chem.MolFromSmiles('[*]')


def mol_from_smiles(smi):
    smi = canonicalize(smi)
    return Chem.MolFromSmiles(smi)


def strip_dummy_atoms(mol):
    try:
        hydrogen = mol_from_smiles('[H]')
        mols = Chem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=True)
        mol = Chem.RemoveHs(mols[0])
    except Exception:
        return None
    return mol


def break_on_bond(mol, bond, min_length=3):
    if mol.GetNumAtoms() - bond <= min_length:
        return [mol]

    broken = Chem.FragmentOnBonds(
        mol, bondIndices=[bond],
        dummyLabels=[(0, 0)])

    res = Chem.GetMolFrags(
        broken, asMols=True, sanitizeFrags=False)

    return res


def get_size(frag):
    dummies = count_dummies(frag)
    total_atoms = frag.GetNumAtoms()
    real_atoms = total_atoms - dummies
    return real_atoms


def count_dummies(mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            count += 1
    return count


def mol_to_smiles(mol):
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonicalize(smi)


def mols_to_smiles(mols):
    return [mol_to_smiles(m) for m in mols]


def canonicalize(smi, clear_stereo=False):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if clear_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def fragment_recursive(mol, frags):
    try:
        bonds = list(BRICS.FindBRICSBonds(mol))

        if bonds == []:
            frags.append(mol)
            return frags

        idxs, labs = list(zip(*bonds))

        bond_idxs = []
        for a1, a2 in idxs:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            bond_idxs.append(bond.GetIdx())

        order = np.argsort(bond_idxs).tolist()
        bond_idxs = [bond_idxs[i] for i in order]

        # 只会断开一根键，也就是说，如果某个片段可以切割两个断点，但是只会切割其中一个，另一个会跟该变短视作一个整体
        broken = Chem.FragmentOnBonds(mol,
                                      bondIndices=[bond_idxs[0]],
                                      dummyLabels=[(0, 0)])
        head, tail = Chem.GetMolFrags(broken, asMols=True)
        # print(mol_to_smiles(head), mol_to_smiles(tail))
        frags.append(head)
        return fragment_recursive(tail, frags)
    except Exception:
        pass


def join_molecules(molA, molB):
    marked, neigh = None, None
    for atom in molA.GetAtoms():
        if atom.GetAtomicNum() == 0:
            marked = atom.GetIdx()
            neigh = atom.GetNeighbors()[0]
            break
    neigh = 0 if neigh is None else neigh.GetIdx()

    if marked is not None:
        ed = Chem.EditableMol(molA)
        if neigh > marked:
            neigh = neigh - 1
        ed.RemoveAtom(marked)
        molA = ed.GetMol()

    joined = Chem.ReplaceSubstructs(
        molB, dummy, molA,
        replacementConnectionPoint=neigh,
        useChirality=False)[0]

    try:
        Chem.Kekulize(joined)
    except Exception:
        return None
    return joined


def reconstruct(frags, reverse=False):
    if len(frags) == 1:
        return strip_dummy_atoms(frags[0]), frags

    # if count_dummies(frags[0]) != 1:
    #     return None, None
    #
    # if count_dummies(frags[-1]) != 1:
    #     return None, None
    #
    # for frag in frags[1:-1]:
    #     if count_dummies(frag) != 2:
    #         return None, None

    mol = join_molecules(frags[0], frags[1])
    if mol is None:
        return None, frags
    for i, frag in enumerate(frags[2:]):
        # print(i, mol_to_smiles(frag), mol_to_smiles(mol))
        mol = join_molecules(mol, frag)
        if mol is None:
            break
        # print(i, mol_to_smiles(mol))
    if mol is None:
        return None, frags

    # 去除手性信息
    rdmolops.RemoveStereochemistry(mol)

    # see if there are kekulization/valence errors
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    smi = canonicalize(smi)
    if smi is None:
        return None, frags

    return mol, frags


def break_into_fragments(mol, smi):
    frags = []
    frags = fragment_recursive(mol, frags)

    if len(frags) == 0:
        return smi, np.nan, 0

    if len(frags) == 1:
        return smi, smi, 1

    rec, frags = reconstruct(frags)
    if rec and mol_to_smiles(rec) == smi:
        # fragments = [Chem.MolToSmiles(frag, isomericSmiles=True, canonical=False) for frag in frags]
        fragments = mols_to_smiles(frags)
        return smi, " ".join(fragments), len(frags)

    return smi, np.nan, 0


def sentence2mol(string):
    frag_list = string.replace(" ", "").replace("[BOS]", "").replace("[EOS]", "").split('[SEP]')
    frag_list = [frag for frag in frag_list if frag]  # 去除空字符串
    if len(frag_list[0]) <= 1:
        return None, None
    frag_mol = [Chem.MolFromSmiles(s) for s in frag_list]
    if None in frag_mol:
        return None, None
    mol = reconstruct(frag_mol)[0]
    mol = strip_dummy_atoms(mol)
    if mol is None:
        return None, None
    # 去除手性信息
    # rdmolops.RemoveStereochemistry(mol)
    smi = Chem.MolToSmiles(mol)
    return mol, smi

