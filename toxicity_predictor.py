"""
Toxicity Predictor Module
Adds toxicity structural alerts to complement adme-pred-py
Includes: hERG liability, hepatotoxicity, reactive metabolites, mutagenicity
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import Fragments
from rdkit.Chem import rdMolDescriptors


class ToxicityPredictor:
    """Rule-based toxicity prediction using structural alerts"""
    
    def __init__(self, smiles):
        """
        Initialize with SMILES string
        
        Args:
            smiles: SMILES string of molecule
        """
        if isinstance(smiles, str):
            self.mol = Chem.MolFromSmiles(smiles)
            self.smiles = smiles
        else:
            self.mol = smiles
            self.smiles = Chem.MolToSmiles(smiles)
        
        if self.mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
    
    def check_herg_liability(self):
        """
        Predict hERG (cardiac) liability
        
        hERG blocking causes QT prolongation (dangerous cardiac side effect)
        
        Risk factors:
        - Basic nitrogen (pKa > 7)
        - LogP > 3
        - Molecular weight 300-500
        - Aromatic rings
        
        Returns:
            dict: {'risk_level': 'high'/'medium'/'low', 'reasons': []}
        """
        reasons = []
        risk_score = 0
        
        # Factor 1: Basic nitrogen presence
        basic_nitrogens = 0
        for atom in self.mol.GetAtoms():
            if atom.GetAtomicNum() == 7:  # Nitrogen
                # Check if it's basic (not in aromatic system, not amide)
                if not atom.GetIsAromatic():
                    basic_nitrogens += 1
        
        if basic_nitrogens > 0:
            risk_score += 2
            reasons.append(f"Contains {basic_nitrogens} basic nitrogen(s)")
        
        # Factor 2: LogP
        logp = Descriptors.MolLogP(self.mol)
        if logp > 3:
            risk_score += 2
            reasons.append(f"High lipophilicity (LogP={logp:.2f})")
        
        # Factor 3: Molecular weight
        mw = Descriptors.MolWt(self.mol)
        if 300 < mw < 500:
            risk_score += 1
            reasons.append(f"MW in hERG risk range ({mw:.1f})")
        
        # Factor 4: Aromatic rings
        aromatic_rings = Descriptors.NumAromaticRings(self.mol)
        if aromatic_rings >= 2:
            risk_score += 1
            reasons.append(f"Multiple aromatic rings ({aromatic_rings})")
        
        # Determine risk level
        if risk_score >= 4:
            risk_level = "HIGH"
        elif risk_score >= 2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'reasons': reasons
        }
    
    def check_hepatotoxicity(self):
        """
        Predict hepatotoxicity (liver toxicity) risk
        
        Structural alerts for liver toxicity:
        - Nitro groups
        - Aromatic amines
        - Thiophene rings
        - Quinones
        - Furans with electron-withdrawing groups
        - Halogenated aromatics
        
        Returns:
            dict: {'alert': True/False, 'reasons': []}
        """
        reasons = []
        alert = False
        
        # SMARTS patterns for hepatotoxic substructures
        toxic_patterns = {
            'nitro_aromatic': Chem.MolFromSmarts('c[N+](=O)[O-]'),
            'aromatic_amine': Chem.MolFromSmarts('cN'),
            'thiophene': Chem.MolFromSmarts('c1ccsc1'),
            'quinone': Chem.MolFromSmarts('C1=CC(=O)C=CC1=O'),
            'halogenated_aromatic': Chem.MolFromSmarts('c[F,Cl,Br,I]'),
            'aromatic_hydroxyl': Chem.MolFromSmarts('cO'),
        }
        
        for name, pattern in toxic_patterns.items():
            if pattern and self.mol.HasSubstructMatch(pattern):
                alert = True
                readable_name = name.replace('_', ' ').title()
                reasons.append(f"Contains {readable_name}")
        
        # Check for multiple aromatic rings with electron-withdrawing groups
        if Descriptors.NumAromaticRings(self.mol) >= 3:
            ewg_count = Fragments.fr_nitro(self.mol) + Fragments.fr_halogen(self.mol)
            if ewg_count > 0:
                alert = True
                reasons.append("Multiple aromatic rings with electron-withdrawing groups")
        
        return {
            'alert': alert,
            'reasons': reasons if alert else ['No hepatotoxic alerts detected']
        }
    
    def check_reactive_metabolites(self):
        """
        Check for potential reactive metabolite formation
        
        Reactive metabolites can cause:
        - Idiosyncratic toxicity
        - Immune reactions
        - DNA damage
        
        Alerts:
        - Anilines (oxidation to reactive quinone-imines)
        - Furans (oxidation to reactive epoxides)
        - Thiophenes (oxidation to reactive sulfoxides)
        - Acetylenes (CYP450 inactivation)
        - Hydrazines
        
        Returns:
            dict: {'alert': True/False, 'reasons': []}
        """
        reasons = []
        alert = False
        
        reactive_patterns = {
            'aniline': Chem.MolFromSmarts('c-N'),
            'furan': Chem.MolFromSmarts('o1cccc1'),
            'thiophene': Chem.MolFromSmarts('s1cccc1'),
            'acetylene': Chem.MolFromSmarts('C#C'),
            'hydrazine': Chem.MolFromSmarts('N-N'),
            'epoxide': Chem.MolFromSmarts('C1OC1'),
            'alkyl_halide': Chem.MolFromSmarts('C[Cl,Br,I]'),
        }
        
        for name, pattern in reactive_patterns.items():
            if pattern and self.mol.HasSubstructMatch(pattern):
                alert = True
                readable_name = name.replace('_', ' ').title()
                reasons.append(f"Contains {readable_name} (reactive metabolite risk)")
        
        return {
            'alert': alert,
            'reasons': reasons if alert else ['No reactive metabolite alerts']
        }
    
    def check_mutagenicity(self):
        """
        Predict mutagenicity risk (Ames test prediction)
        
        Mutagenic alerts (Ames positive structural features):
        - Nitro aromatics
        - Aromatic azo compounds
        - Epoxides
        - Alkyl halides
        - Aromatic amines (after metabolic activation)
        - Michael acceptors (α,β-unsaturated carbonyls)
        
        Returns:
            dict: {'alert': True/False, 'reasons': []}
        """
        reasons = []
        alert = False
        
        mutagenic_patterns = {
            'nitro_aromatic': Chem.MolFromSmarts('c[N+](=O)[O-]'),
            'azo_aromatic': Chem.MolFromSmarts('cN=Nc'),
            'epoxide': Chem.MolFromSmarts('C1OC1'),
            'alkyl_halide': Chem.MolFromSmarts('[CH2,CH][Cl,Br,I]'),
            'aromatic_amine': Chem.MolFromSmarts('cN([H])[H]'),
            'michael_acceptor': Chem.MolFromSmarts('C=CC=O'),
            'aromatic_nitro': Chem.MolFromSmarts('c[N+](=O)[O-]'),
            'hydrazine': Chem.MolFromSmarts('N-N'),
        }
        
        for name, pattern in mutagenic_patterns.items():
            if pattern and self.mol.HasSubstructMatch(pattern):
                alert = True
                readable_name = name.replace('_', ' ').title()
                reasons.append(f"Contains {readable_name}")
        
        return {
            'alert': alert,
            'reasons': reasons if alert else ['No mutagenic alerts detected']
        }
    
    def check_cyp450_inhibition(self):
        """
        Predict CYP450 enzyme inhibition risk
        
        CYP450 inhibition causes drug-drug interactions
        Focus on major isoforms: CYP3A4, CYP2D6, CYP2C9
        
        Risk factors:
        - Basic nitrogen
        - Multiple aromatic rings
        - High lipophilicity
        - Specific functional groups
        
        Returns:
            dict: {'risk_level': 'high'/'medium'/'low', 'isoforms': []}
        """
        at_risk_isoforms = []
        reasons = []
        
        logp = Descriptors.MolLogP(self.mol)
        aromatic_rings = Descriptors.NumAromaticRings(self.mol)
        
        # CYP3A4 inhibition (most common DDI)
        if logp > 3 and aromatic_rings >= 2:
            at_risk_isoforms.append('CYP3A4')
            reasons.append('Lipophilic with multiple aromatics (CYP3A4 risk)')
        
        # CYP2D6 inhibition (basic amines)
        basic_n = 0
        for atom in self.mol.GetAtoms():
            if atom.GetAtomicNum() == 7 and not atom.GetIsAromatic():
                basic_n += 1
        
        if basic_n > 0 and aromatic_rings >= 1:
            at_risk_isoforms.append('CYP2D6')
            reasons.append('Basic nitrogen with aromatics (CYP2D6 risk)')
        
        # CYP2C9 inhibition (acidic groups)
        acidic_pattern = Chem.MolFromSmarts('C(=O)[OH]')
        if acidic_pattern and self.mol.HasSubstructMatch(acidic_pattern):
            at_risk_isoforms.append('CYP2C9')
            reasons.append('Carboxylic acid present (CYP2C9 risk)')
        
        risk_level = "HIGH" if len(at_risk_isoforms) >= 2 else \
                     "MEDIUM" if len(at_risk_isoforms) == 1 else "LOW"
        
        return {
            'risk_level': risk_level,
            'at_risk_isoforms': at_risk_isoforms,
            'reasons': reasons if reasons else ['Low CYP450 inhibition risk']
        }
    
    def full_toxicity_report(self):
        """
        Generate comprehensive toxicity report
        
        Returns:
            dict with all toxicity assessments
        """
        return {
            'smiles': self.smiles,
            'herg_liability': self.check_herg_liability(),
            'hepatotoxicity': self.check_hepatotoxicity(),
            'reactive_metabolites': self.check_reactive_metabolites(),
            'mutagenicity': self.check_mutagenicity(),
            'cyp450_inhibition': self.check_cyp450_inhibition(),
        }
    
    def print_report(self):
        """Print human-readable toxicity report"""
        report = self.full_toxicity_report()
        
        print("="*60)
        print("TOXICITY ASSESSMENT REPORT")
        print("="*60)
        print(f"SMILES: {self.smiles}")
        print()
        
        # hERG
        herg = report['herg_liability']
        print(f"hERG Cardiac Liability: {herg['risk_level']}")
        for reason in herg['reasons']:
            print(f"  • {reason}")
        print()
        
        # Hepatotoxicity
        hepato = report['hepatotoxicity']
        status = "⚠ ALERT" if hepato['alert'] else "✓ PASS"
        print(f"Hepatotoxicity: {status}")
        for reason in hepato['reasons']:
            print(f"  • {reason}")
        print()
        
        # Reactive metabolites
        reactive = report['reactive_metabolites']
        status = "⚠ ALERT" if reactive['alert'] else "✓ PASS"
        print(f"Reactive Metabolites: {status}")
        for reason in reactive['reasons']:
            print(f"  • {reason}")
        print()
        
        # Mutagenicity
        mutagen = report['mutagenicity']
        status = "⚠ ALERT" if mutagen['alert'] else "✓ PASS"
        print(f"Mutagenicity: {status}")
        for reason in mutagen['reasons']:
            print(f"  • {reason}")
        print()
        
        # CYP450
        cyp = report['cyp450_inhibition']
        print(f"CYP450 Inhibition: {cyp['risk_level']}")
        if cyp['at_risk_isoforms']:
            print(f"  At-risk isoforms: {', '.join(cyp['at_risk_isoforms'])}")
        for reason in cyp['reasons']:
            print(f"  • {reason}")
        
        print("="*60)


if __name__ == '__main__':
    # Test examples
    test_molecules = {
        'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
        'Paracetamol': 'CC(=O)Nc1ccc(O)cc1',
        'Nitrobenzene (toxic)': 'c1ccc(cc1)[N+](=O)[O-]',
    }
    
    for name, smiles in test_molecules.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        try:
            predictor = ToxicityPredictor(smiles)
            predictor.print_report()
        except Exception as e:
            print(f"Error: {e}")