from tdc import Oracle


def cal_QED(smiles):
    oracle = Oracle(name='QED')
    return oracle(smiles)


def cal_DRD2(smiles):
    oracle = Oracle(name='DRD2')
    return oracle(smiles)


def cal_GSK3b(smiles):
    oracle = Oracle(name='GSK3b')
    return oracle(smiles)


def cal_JNK3(smiles):
    oracle = Oracle(name='JNK3')
    return oracle(smiles)


def cal_Albuterol_Similarity(smiles):
    oracle = Oracle(name='Albuterol_Similarity')
    return oracle(smiles)


def cal_Amlodipine_MPO(smiles):
    oracle = Oracle(name='Amlodipine_MPO')
    return oracle(smiles)


def cal_Celecoxib_Rediscovery(smiles):
    oracle = Oracle(name='Celecoxib_Rediscovery')
    return oracle(smiles)


def cal_Deco_Hop(smiles):
    oracle = Oracle(name='Deco_Hop')
    return oracle(smiles)


def cal_Fexofenadine_MPO(smiles):
    oracle = Oracle(name='Fexofenadine_MPO')
    return oracle(smiles)


def cal_Isomers_C7H8N2O2(smiles):
    oracle = Oracle(name='Isomers_C7H8N2O2')
    return oracle(smiles)


def cal_Isomers_C9H10N2O2PF2Cl(smiles):
    oracle = Oracle(name='Isomers_C9H10N2O2PF2Cl')
    return oracle(smiles)


def cal_Median1(smiles):
    oracle = Oracle(name='Median1')
    return oracle(smiles)


def cal_Median2(smiles):
    oracle = Oracle(name='Median2')
    return oracle(smiles)


def cal_Mestranol_Similarity(smiles):
    oracle = Oracle(name='Mestranol_Similarity')
    return oracle(smiles)


def cal_Osimertinib_MPO(smiles):
    oracle = Oracle(name='Osimertinib_MPO')
    return oracle(smiles)


def cal_Perindopril_MPO(smiles):
    oracle = Oracle(name='Perindopril_MPO')
    return oracle(smiles)


def cal_Ranolazine_MPO(smiles):
    oracle = Oracle(name='Ranolazine_MPO')
    return oracle(smiles)


def cal_Scaffold_Hop(smiles):
    oracle = Oracle(name='Scaffold_Hop')
    return oracle(smiles)


def cal_Sitagliptin_MPO(smiles):
    oracle = Oracle(name='Sitagliptin_MPO')
    return oracle(smiles)


def cal_Thiothixene_Rediscovery(smiles):
    oracle = Oracle(name='Thiothixene_Rediscovery')
    return oracle(smiles)


def cal_Troglitazone_Rediscovery(smiles):
    oracle = Oracle(name='Troglitazone_Rediscovery')
    return oracle(smiles)


def cal_Valsartan_SMARTS(smiles):
    oracle = Oracle(name='Valsartan_SMARTS')
    return oracle(smiles)


def cal_Zaleplon_MPO(smiles):
    oracle = Oracle(name='Zaleplon_MPO')
    return oracle(smiles)


def cal_all_metrics(smiles):
    """
    Calculates the 23 metrics based on the provided SMILES string and scoring functions described in the images.

    Args:
        smiles (str): SMILES string of the molecule.

    Returns:
        dict: A dictionary containing the calculated values for each metric. Returns None if the input SMILES is invalid.
    """
    results = {}

    list = ['QED', 'DRD2', 'GSK3b', 'JNK3', 'Albuterol_Similarity', 'Amlodipine_MPO', 'Celecoxib_Rediscovery',
            'Deco_Hop',
            'Fexofenadine_MPO', 'Isomers_C7H8N2O2', 'Isomers_C9H10N2O2PF2Cl', 'Median1', 'Median2',
            'Mestranol_Similarity',
            'Osimertinib_MPO', 'Perindopril_MPO', 'Ranolazine_MPO', 'Scaffold_Hop', 'Sitagliptin_MPO',
            'Thiothixene_Rediscovery',
            'Troglitazone_Rediscovery', 'Valsartan_SMARTS', 'Zaleplon_MPO']

    for i in list:
        results[i] = globals()[f'cal_{i}'](smiles)

    return results


if __name__ == "__main__":
    # Example usage:
    results = cal_all_metrics(['CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
                               'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
                               'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O'])

    if results:
        for key, value in results.items():
            print(f"{key}: {value}")
    else:
        print("Could not calculate metrics for the given SMILES string.")
