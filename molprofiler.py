
import argparse
import sys
import os
import json
import pandas as pd
import dockstring
import rdkit
from dockstring import load_target, list_all_target_names
from toxicity_predictor import ToxicityPredictor
from compound_namer import get_compound_name

import adme_pred
# Ensure openpyxl is available for XLSX support
try:
    import openpyxl
except ImportError:
    print("Warning: openpyxl not installed. Install with: pip install openpyxl")
    print("XLSX output will not work, use CSV instead.")

##################### BEGINNING OF ADME #####################################
def calculate_adme_properties(smiles):
    """
    Calculate comprehensive ADME properties for a single SMILES
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary with all ADME properties, or None if SMILES is invalid
    """
    from adme_pred import ADME
    from rdkit import Chem
    
    # Check if SMILES is valid
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        adme = ADME(smiles)
        
        properties = {
            # === Druglikeness Filters (Pass/Fail) ===
            "lipinski_pass": adme.druglikeness_lipinski(),
            "lipinski_violations": adme.druglikeness_lipinski(verbose=True),
            "egan_pass": adme.druglikeness_egan(),
            "egan_violations": adme.druglikeness_egan(verbose=True),
            "ghose_pass": adme.druglikeness_ghose(),
            "ghose_violations": adme.druglikeness_ghose(verbose=True),
            "ghose_pref_pass": adme.druglikeness_ghose_pref(),
            "muegge_pass": adme.druglikeness_muegge(),
            "muegge_violations": adme.druglikeness_muegge(verbose=True),
            "veber_pass": adme.druglikeness_veber(),
            "veber_violations": adme.druglikeness_veber(verbose=True),
            
            # === Pharmacokinetics ===
            "gi_absorption": "High" if adme.boiled_egg_hia() else "Low",
            "bbb_permeant": "Yes" if adme.boiled_egg_bbb() else "No",
            
            # === Medicinal Chemistry Filters ===
            "pains_alert": adme.pains(),
            "brenk_alert": adme.brenk(),
            
            # === Molecular Properties ===
            "molecular_weight": adme._molecular_weight(),
            "logp": adme._logp(),
            "tpsa": adme._tpsa(),
            "h_bond_donors": adme._h_bond_donors(),
            "h_bond_acceptors": adme._h_bond_acceptors(),
            "n_rotatable_bonds": adme._n_rot_bonds(),
            "n_atoms": adme._n_atoms(),
            "n_carbons": adme._n_carbons(),
            "n_heteroatoms": adme._n_heteroatoms(),
            "n_rings": adme._n_rings(),
            "molar_refractivity": adme._molar_refractivity(),
        }
        
        # Add toxicity predictions
        tox = ToxicityPredictor(smiles)
        tox_report = tox.full_toxicity_report()

        properties.update({
        # hERG
        "herg_risk": tox_report['herg_liability']['risk_level'],
        "herg_reasons": "; ".join(tox_report['herg_liability']['reasons']) if tox_report['herg_liability']['reasons'] else "None",

        # Hepatotoxicity
        "hepatotoxicity_alert": "Yes" if tox_report['hepatotoxicity']['alert'] else "No",
        "hepatotoxicity_reasons": "; ".join(tox_report['hepatotoxicity']['reasons']),

        # Reactive metabolites
        "reactive_metabolite_alert": "Yes" if tox_report['reactive_metabolites']['alert'] else "No",
        "reactive_metabolite_reasons": "; ".join(tox_report['reactive_metabolites']['reasons']),

        # Mutagenicity
        "mutagenicity_alert": "Yes" if tox_report['mutagenicity']['alert'] else "No",
        "mutagenicity_reasons": "; ".join(tox_report['mutagenicity']['reasons']),

        # CYP450
        "cyp450_risk": tox_report['cyp450_inhibition']['risk_level'],
        "cyp450_isoforms": ", ".join(tox_report['cyp450_inhibition']['at_risk_isoforms']) if tox_report['cyp450_inhibition']['at_risk_isoforms'] else "None",
        })

        return properties
        
    except Exception as e:
        print(f"Error calculating ADME for {smiles}: {e}")
        return None


def analyze_predictions_with_adme(smiles_input, output_file='results.csv'):
    """
    Analyze SMILES with ADME properties
    
    Args:
        smiles_input: Either a CSV filepath with 'smiles' column OR list of SMILES strings
        output_file: Output file (CSV or XLSX)
    """
    print("\n### ADME Analysis Starting ###\n")
    
    # Load SMILES
    if isinstance(smiles_input, str):  # CSV filepath
        df = pd.read_csv(smiles_input)
        if 'smiles' in df.columns:
            smiles_list = df['smiles'].tolist()
        elif 'predicted_smiles_list' in df.columns:  # Legacy support
            smiles_list = df['predicted_smiles_list'].tolist()
        else:
            raise ValueError("CSV must have 'smiles' column")
    elif isinstance(smiles_input, list):  # Direct list
        smiles_list = smiles_input
    else:
        raise ValueError("Input must be CSV filepath or list of SMILES")
    
    results = []
    total = len(smiles_list)
    
    for idx, smiles in enumerate(smiles_list, 1):
        print(f"Processing {idx}/{total}: {smiles[:50]}...")
        
        # Get compound name
        compound_name = get_compound_name(smiles)
        print(f"  Identified as: {compound_name}")

        # Calculate ADME
        adme_props = calculate_adme_properties(smiles)
        
        if adme_props is not None:
            result_row = {'compound_name': compound_name, 'smiles': smiles}
            result_row.update(adme_props)
            results.append(result_row)
        else:
            print(f"  ⚠ Invalid SMILES or ADME calculation failed")
            results.append({'smiles': smiles, 'error': 'Invalid SMILES'})
    
    # Save results
    results_df = pd.DataFrame(results)
    
    if output_file.endswith('.xlsx'):
        results_df.to_excel(output_file, index=False, engine='openpyxl')
    else:
        results_df.to_csv(output_file, index=False)
    
    print(f"\n✓ ADME analysis complete! Results saved to {output_file}")
    print(f"  Valid molecules: {len([r for r in results if 'error' not in r])}/{total}")
    
    return results_df

############################# end of ADME ######################################################


################################ Start of Docking ###################################################

    
def perform_molecular_docking(smiles_input, output_file='docking_results.csv', 
                              target_list=None, view_poses=False):
    """
    Perform molecular docking on compounds
    
    Args:
        smiles_input: Either CSV filepath (with 'smiles' column) OR list of SMILES strings
        output_file: Output file for docking results (CSV or XLSX)
        target_list: List of target names. If None, uses all available targets
        view_poses: If True, prompt to view each pose in PyMol
    """
    print("\n### Molecular Docking Starting ###\n")
    
    # Load SMILES from input
    if isinstance(smiles_input, str):  # CSV filepath
        df = pd.read_csv(smiles_input)
        if 'smiles' not in df.columns:
            raise ValueError("CSV must have 'smiles' column")
        smiles_list = df['smiles'].tolist()
    elif isinstance(smiles_input, list):  # Direct list of SMILES
        smiles_list = smiles_input
    else:
        raise ValueError("smiles_input must be CSV filepath or list of SMILES")
    
    # Remove duplicates and invalid SMILES
    from rdkit import Chem
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and smi not in valid_smiles:
            valid_smiles.append(smi)
    
    print(f"Loaded {len(valid_smiles)} valid unique SMILES for docking")

    # List available targets
    all_targets = list_all_target_names()
    print("\nAvailable protein targets:")
    for i, target in enumerate(all_targets, 1):
        print(f"{i}. {target}")
    
    # Get user selection
    target_input = input("\nEnter target numbers (comma-separated, e.g., 1,5,12): ").strip()
    try:
        target_indices = [int(x.strip()) - 1 for x in target_input.split(',')]
        selected_targets = [all_targets[i] for i in target_indices]
    except (ValueError, IndexError):
        print("Invalid input. Skipping docking.")
        return None
    
    print(f"\nSelected targets: {', '.join(selected_targets)}")
    print(f"Starting docking for {len(valid_smiles)} SMILES against {len(selected_targets)} targets...")
    
    # Perform docking
    docking_results = []
    total_ops = len(valid_smiles) * len(selected_targets)
    completed = 0
    
    for target_name in selected_targets:
        target = load_target(target_name)
        
        for smiles in valid_smiles:
            completed += 1
            progress = (completed / total_ops) * 100
            print(f"Progress: {progress:.1f}% - Docking {smiles[:30]}... against {target_name}")
            
            try:
                score, info = target.dock(smiles)
                
                if score is not None:
                    # Save protein-ligand complex as single PDB
                    output_dir = 'docking_poses'
                    import os
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    # Sanitize filename
                    safe_smiles = smiles[:30].replace('/', '_').replace('\\', '_')
                    safe_target = target_name.replace('/', '_')
                    complex_file = f"{output_dir}/{safe_target}_{safe_smiles}_complex.pdb"
                    
                    try:
                
                        
                        # Get protein PDBQT path and convert to PDB path
                        pdbqt_path = str(target.pdbqt_path)
                        # Try to find PDB version (usually stored alongside)
                        pdb_path = pdbqt_path.replace('.pdbqt', '.pdb')
                        
                        # Check if PDB exists, otherwise use PDBQT (compatible enough)
                        import os
                        if not os.path.exists(pdb_path):
                            pdb_path = pdbqt_path
                            print(f"  Note: Using PDBQT format (charges included)")
                        
                        # Get protein content
                        with open(pdb_path, 'r') as f:
                            protein_lines = f.readlines()
                        
                        # Get ligand PDB content
                        ligand_pdb = Chem.MolToPDBBlock(info['ligand'])
                        
                        # Combine: protein + ligand
                        with open(complex_file, 'w') as f:
                            # Write protein (remove END if present)
                            for line in protein_lines:
                                if not line.startswith('END'):
                                    f.write(line)
                            
                            # Write ligand
                            f.write(ligand_pdb)
                            
                            # Write END
                            f.write('END\n')
                        
                        print(f"  ✓ Complex saved: {complex_file}")
                        # === OFFER TO VIEW IN PYMOL ===
                        if view_poses:
                            view_choice = input(f"    View this pose in PyMol now? (y/n): ").strip().lower()
                            if view_choice == 'y':
                                try:
                                    import pymol
                                    from pymol import cmd
                                    
                                    pymol.finish_launching(['pymol', '-q'])
                                    cmd.load(complex_file, f'{target_name}_{safe_smiles}')
                                    cmd.show('cartoon', 'polymer')
                                    cmd.show('sticks', 'organic')
                                    cmd.color('cyan', 'polymer')
                                    cmd.color('yellow', 'organic')
                                    cmd.zoom('organic', 8)
                                    
                                    print(f"    ✓ PyMol launched. Close window to continue...")
                                    
                                except ImportError:
                                    print(f"    ⚠ PyMol not installed. Run: pip install pymol-open-source")
                                except Exception as e:
                                    print(f"    ⚠ PyMol error: {e}")
                    except Exception as e:
                        print(f"  Failed to save complex: {e}")
                    
                    result = {
                        'smiles': smiles,
                        'target_name': target_name,
                        'docking_score': round(score, 2),
                        'affinity_list': str(info.get('affinities', [])),
                        'num_conformers': info.get('ligand').GetNumConformers() if 'ligand' in info else 0,
                        'pdb_file': complex_file,  # Add this column
                        'docking_status': 'success'
                    }
                else:

                    result = {
                        'smiles': smiles,
                        'target_name': target_name,
                        'docking_score': None,
                        'affinity_list': None,
                        'num_conformers': 0,
                        'docking_status': 'failed: no pose found'
                    }
            except Exception as e:
                result = {
                    'smiles': smiles,
                    'target_name': target_name,
                    'docking_score': None,
                    'affinity_list': None,
                    'num_conformers': 0,
                    'docking_status': f'failed: {str(e)[:500]}'
                }
            
            docking_results.append(result)
    
    # Save results
    # Save results
    results_df = pd.DataFrame(docking_results)
    
    if output_file.endswith('.xlsx'):
        results_df.to_excel(output_file, index=False, engine='openpyxl')
    else:
        results_df.to_csv(output_file, index=False)
    
    # Summary
    success_count = results_df[results_df['docking_status'] == 'success'].shape[0]
    print(f"\nSummary:")
    print(f"  Total docking attempts: {len(results_df)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(results_df) - success_count}")
    
    return results_df

################################# End of Docking ####################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MolProfiler: ADME/Toxicity/Docking Analysis')
    
    # === Input Options ===
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--smiles', type=str,
                            help='Single SMILES string to analyze')
    input_group.add_argument('--input', type=str,
                            help='CSV file with SMILES (requires "smiles" column)')
    
    # === Output Options ===
    parser.add_argument('--adme_output', type=str, default='ADME_results.xlsx',
                        help='ADME output file (CSV or XLSX)')
    parser.add_argument('--docking_output', type=str, default='Docking_results.xlsx',
                        help='Docking output file (CSV or XLSX)')
    
    # === Analysis Options ===
    parser.add_argument('--skip_adme', action='store_true',
                        help='Skip ADME analysis')
    parser.add_argument('--skip_docking', action='store_true',
                        help='Skip molecular docking')
    parser.add_argument('--targets', type=str, default='PPARG,DPP4',
                        help='Comma-separated list of docking targets (default: PPARG,DPP4)')
    parser.add_argument('--view_poses', action='store_true',
                        help='Interactively view docking poses in PyMol')
    
    args = parser.parse_args()
    
    # === Load SMILES ===
    if args.smiles:
        smiles_list = [args.smiles]
        print(f"\nAnalyzing 1 compound: {args.smiles[:60]}...")
    elif args.input:
        df = pd.read_csv(args.input)
        if 'smiles' not in df.columns:
            raise ValueError("Input CSV must have 'smiles' column")
        smiles_list = df['smiles'].tolist()
        print(f"\nLoaded {len(smiles_list)} compounds from {args.input}")
    
    # === Run ADME Analysis ===
    if not args.skip_adme:
        print("\n" + "="*60)
        print("ADME & Toxicity Analysis")
        print("="*60)
        analyze_predictions_with_adme(smiles_list, args.adme_output)
        adme_file = args.adme_output
    else:
        # Create minimal dataframe for docking
        adme_file = None
    
    # === Run Docking ===
    if not args.skip_docking:
        print("\n" + "="*60)
        print("Molecular Docking")
        print("="*60)
        
        target_list = [t.strip() for t in args.targets.split(',')]
        
        # Use ADME output if available, otherwise use smiles_list
        smiles_input = adme_file if adme_file else smiles_list
        
        perform_molecular_docking(
            smiles_input, 
            args.docking_output,
            target_list=target_list,
            view_poses=args.view_poses
        )
    
    print("\n" + "="*60)
    print(" Analysis Complete!")
    print("="*60)