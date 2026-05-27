#得到sdf文件中每个分子的所有描述符的数值
import pandas as pd
from rdkit.Chem import Descriptors,Descriptors3D
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import rdBase
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from rdkit.Chem import MACCSkeys

def SaveToMolFile(l_mol:list,list_path:str,not_save_path,l_not=[],n_not=[]):

    # l_mol is the list of mols to save

    for n in l_mol:
        l = pd.read_csv(list_path)
        l1=list(l)
        try:
            if n not in l1:
                mol = AllChem.AddHs(Chem.MolFromSmiles(n))
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
                with Chem.SDWriter(r"C:\Users\hp\Desktop\mol_new\mol"+str(len(l.index))+".sdf") as f:
                    f.write(mol)
                print(n,'成功录入')
                l.loc[len(l.index)]=[n]
                l.to_csv(list_path,index=False)
            else:
                print(n,'已存在')
        except:
            print(n,'录入失败')
            l_not.append(n)
            n_not.append(l_mol.index(n))

        if len(l.index)%500 == 0 or l_mol.index(n)==-1:
            l_not_list = pd.concat([pd.DataFrame(n_not),pd.DataFrame(l_not)],axis=1)
            l_not_list.to_csv(not_save_path,index=False)

    l_not_all = pd.concat([pd.DataFrame(n_not),pd.DataFrame(l_not)],axis=1)
    l_not_all.to_csv(not_save_path, index=False)


def GetRdkitDescriptors(num,d00,sdf_file:str,data_path:str):

    #num is the number of smiles in list
    # d00 is the columns to save of a dataframe

    rdBase.DisableLog('rdApp.warning')
    data=pd.read_csv(data_path)
    suppl = Chem.MolFromMolFile(sdf_file)
    # mols = [mol for mol in suppl]
    # print(suppl)
    smiles_list = Chem.MolToSmiles(suppl)
    # print(smiles_list)
    descs = ['MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt',
             'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge',
             'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2',
             'FpDensityMorgan3',  'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n',
             'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha',
             'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
             'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
             'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
             'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
             'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8',
             'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3',
             'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1',
             'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
             'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount',
             'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles',
             'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
             'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
             'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO',
             'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN',
             'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
             'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
             'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur',
             'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
             'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide',
             'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
             'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole',
             'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
             'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide',
             'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene',
             'fr_unbrch_alkane', 'fr_urea']
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    #Mols=[]
   # for s in  smiles_list:
    Mol = AllChem.AddHs(Chem.MolFromSmiles(smiles_list))
    conformer = Chem.Conformer()
    AllChem.EmbedMolecule(Mol, randomSeed=1)
        #Mols.append(Mol)
    for atom_idx in range(Mol.GetNumAtoms()):
         position = Mol.GetConformer().GetAtomPosition(atom_idx)
         conformer.SetAtomPosition(atom_idx, position)
    descriptors3D = pd.DataFrame([Descriptors3D.CalcMolDescriptors3D(Mol)])
    descriptors3D.index= [num]
    descriptors = pd.DataFrame([desc_calc.CalcDescriptors(Mol)])
    descriptors.columns = descs
    descriptors.index = [num]
   # index_list = list(map(str,list(range(len(Mols)))))
    y = pd.DataFrame([smiles_list])
    y.index = [num]
    y.columns = ["smiles"]
    y0 = pd.DataFrame([num])
    y0.index = [num]
    y0.columns = ["num"]
    dataset = pd.concat([y0,y,descriptors,descriptors3D],axis=1)
    dataset['Concentration'] = 2
    dataset['Temperature'] = 80
    dataset['Time(h)'] = 720
    dataset_del = dataset.loc[:, list(d00.columns)]
    data_1=pd.concat([data,dataset_del],axis=0)
    data_1.to_csv(data_path,index=False)


def Get_ECFP(data,save=True):

# data is the dataframe of aem-----------------original!!!

    x0 = []
    rdBase.DisableLog('rdApp.warning')
    for i in data["smiles"]:
        fp = np.zeros((0,))
        m = Chem.MolFromSmiles(i)
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024), fp)
        x0.append(fp)

    x1 = pd.DataFrame(x0)
    x1.columns = ['ECFP-'+str(im) for im in range(1024)]
    x = pd.concat(
        [x1, data['Concentration'], data['Temperature'],  data['Time(h)']],
        axis=1)

    if save == True:
        x.to_excel('fingerprint_ECFP.xlsx',index=False)
    else:
        return x


def Get_MACCS(data,save=True):

# data is the dataframe of aem------------------original!!!

    x0 = []
    rdBase.DisableLog('rdApp.warning')

    for i in data["smiles"]:
        fp = np.zeros((0,))
        m = Chem.MolFromSmiles(i)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), fp)
        x0.append(fp)

    x1 = pd.DataFrame(x0)
    x1.columns = ['MACCS-'+str(im) for im in range(167)]
    x = pd.concat(
        [x1, data['Concentration'], data['Temperature'], data['Time(h)']],
        axis=1)

    if save == True:
        x.to_excel('fingerprint_ECFP.xlsx', index=False)
    else:
        return x



def GetDescriptors(s: str, concentration, temperature, time):
    # s is the molecular smile

    rdBase.DisableLog('rdApp.warning')

    suppl = Chem.MolFromSmiles(s)
    smile = Chem.MolToSmiles(suppl)

    descs = ['MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt',
             'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge',
             'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2',
             'FpDensityMorgan3', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n',
             'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha',
             'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
             'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
             'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
             'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
             'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8',
             'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3',
             'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1',
             'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
             'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount',
             'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles',
             'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
             'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
             'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO',
             'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN',
             'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
             'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
             'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur',
             'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide',
             'fr_ester',
             'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide',
             'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
             'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
             'fr_oxazole',
             'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid',
             'fr_phos_ester',
             'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide',
             'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan',
             'fr_thiophene',
             'fr_unbrch_alkane', 'fr_urea']
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)

    Mol = AllChem.AddHs(Chem.MolFromSmiles(smile))
    conformer = Chem.Conformer()
    AllChem.EmbedMolecule(Mol, randomSeed=1)

    for atom_idx in range(Mol.GetNumAtoms()):
        position = Mol.GetConformer().GetAtomPosition(atom_idx)
        conformer.SetAtomPosition(atom_idx, position)

    descriptors = pd.DataFrame([desc_calc.CalcDescriptors(Mol)])
    descriptors3D = pd.DataFrame([Descriptors3D.CalcMolDescriptors3D(Mol)])
    descriptors.columns = descs
    dataset = pd.concat([descriptors, descriptors3D], axis=1)

    dataset['Concentration'] = concentration
    dataset['Temperature'] = temperature
    dataset['Time(h)'] = time

    data = pd.read_excel(r"C:\Users\hp\Desktop\AEM_project\software.xlsx")
    d = data.iloc[:, 2:]
    dataset_del = dataset.loc[:, list(d.columns)]

    return dataset_del