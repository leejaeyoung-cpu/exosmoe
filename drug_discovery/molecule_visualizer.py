"""
RDKit을 사용한 분자 구조 시각화 유틸리티
"""

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
import io
from PIL import Image


def smiles_to_mol(smiles):
    """SMILES → Mol 객체 변환"""
    return Chem.MolFromSmiles(smiles)


def generate_2d_structure(mol, size=(400, 400)):
    """2D 구조 이미지 생성"""
    if mol is None:
        return None
    
    # 2D 좌표 생성
    AllChem.Compute2DCoords(mol)
    
    # 이미지 생성
    img = Draw.MolToImage(mol, size=size)
    return img


def generate_3d_structure(mol):
    """3D 구조 이미지 생성 (여러 각도)"""
    if mol is None:
        return []
    
    # 3D 좌표 생성
    mol_3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_3d)
    
    # 여러 각도에서 이미지 생성
    images = []
    
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import IPythonConsole
    
    # View 1: Front
    img1 = Draw.MolToImage(mol_3d, size=(300, 300))
    images.append(("Front View", img1))
    
    # View 2: Side (회전)
    # 실제 3D 회전은 복잡하므로 2D로 대체
    img2 = Draw.MolToImage(mol, size=(300, 300))
    images.append(("2D Structure", img2))
    
    return images


def calculate_properties(mol):
    """분자 특성 계산"""
    if mol is None:
        return {}
    
    props = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'Rings': Descriptors.RingCount(mol),
        'AromaticRings': Descriptors.NumAromaticRings(mol),
        'Atoms': mol.GetNumAtoms(),
        'Bonds': mol.GetNumBonds(),
        'SaturatedRings': Descriptors.NumSaturatedRings(mol),
        'AliphaticRings': Descriptors.NumAliphaticRings(mol),
    }
    
    return props


def check_lipinski(mol):
    """Lipinski Rule of Five 검사"""
    if mol is None:
        return False, {}
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    rules = {
        'MW ≤ 500': (mw <= 500, f'{mw:.1f} Da'),
        'LogP ≤ 5': (logp <= 5, f'{logp:.2f}'),
        'HBD ≤ 5': (hbd <= 5, f'{hbd}'),
        'HBA ≤ 10': (hba <= 10, f'{hba}'),
    }
    
    all_pass = all(passed for passed, _ in rules.values())
    
    return all_pass, rules


def generate_substructure_highlights(mol, substructure_smarts=None):
    """특정 substructure 하이라이트"""
    if mol is None:
        return None
    
    # 기본: 방향족 고리 하이라이트
    if substructure_smarts is None:
        # Aromatic rings
        aromatic_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsAromatic()]
        
        img = Draw.MolToImage(
            mol, 
            size=(400, 400),
            highlightAtoms=aromatic_atoms
        )
        return img
    
    # Custom substructure
    pattern = Chem.MolFromSmarts(substructure_smarts)
    if pattern:
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            atoms = list(matches[0])
            img = Draw.MolToImage(
                mol,
                size=(400, 400),
                highlightAtoms=atoms
            )
            return img
    
    return Draw.MolToImage(mol, size=(400, 400))


def compare_molecules(smiles_list, labels=None, mol_per_row=3):
    """여러 분자 비교 이미지"""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    
    if labels is None:
        labels = [f'Molecule {i+1}' for i in range(len(smiles_list))]
    
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mol_per_row,
        subImgSize=(250, 250),
        legends=labels
    )
    
    return img


def smiles_to_iupac(smiles):
    """SMILES → IUPAC 이름 (간단한 경우만)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Unknown"
    
    # RDKit은 기본적으로 IUPAC 생성 안함
    # 간단한 분자만 수동 매칭
    
    known_molecules = {
        'C1=CC=CC=C1C1=CC=NC=C1': '4-Phenylpyridine',
        'CCO': 'Ethanol',
        'CC(=O)O': 'Acetic acid',
    }
    
    return known_molecules.get(smiles, f"C{mol.GetNumAtoms()}H?N?O?")


if __name__ == "__main__":
    # 테스트
    smiles = "C1=CC=CC=C1C1=CC=NC=C1"  # NOVA-091
    
    mol = smiles_to_mol(smiles)
    
    print("분자 특성:")
    props = calculate_properties(mol)
    for key, value in props.items():
        print(f"  {key}: {value}")
    
    print("\nLipinski Rule:")
    passed, rules = check_lipinski(mol)
    print(f"  Overall: {'✅ Pass' if passed else '❌ Fail'}")
    for rule, (p, v) in rules.items():
        print(f"  {rule}: {v} {'✅' if p else '❌'}")
