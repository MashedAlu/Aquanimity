
import argparse
import torch
import sys
import os
import json
import pandas as pd
import dockstring
from dockstring import load_target, list_all_target_names

# Import existing modules
from predict_embs import main as predict_embeddings
sys.path.insert(0, 'decoder')
from decode_embeddings import main as decode_embeddings, get_parser as get_decode_parser

def spectra_to_smiles(pos_low_file, pos_high_file, neg_low_file, neg_high_file, 
                      encoder_model='spectra_encoder.pt',
                      decoder_model='decoder/models/model.pt',
                      decoder_config='decoder/models/config.nb',
                      decoder_vocab='decoder/models/vocab.nb',
                      output_file='predicted_smiles.csv',
                      num_variants=10):
    """
    Convert MS/MS spectra CSV files to SMILES
    
    Args:
        pos_low_file: CSV file with [M+H]+ low energy spectrum
        pos_high_file: CSV file with [M+H]+ high energy spectrum  
        neg_low_file: CSV file with [M-H]- low energy spectrum
        neg_high_file: CSV file with [M-H]- high energy spectrum
        encoder_model: Path to trained encoder model
        decoder_model: Path to trained decoder model
        decoder_config: Path to decoder config
        decoder_vocab: Path to decoder vocabulary
        output_file: Where to save predicted SMILES
        num_variants: Number of SMILES variants to generate
    """
    
    print("Converting spectra to embeddings")
    
    # Create args object for predict_embs
    class Args:
        pass
    args = Args()
    args.pos_low_file = pos_low_file
    args.pos_high_file = pos_high_file
    args.neg_low_file = neg_low_file
    args.neg_high_file = neg_high_file
    
    # Get predicted embedding
    pred_emb = predict_embeddings(args)
    
    # Save embedding temporarily
    temp_emb_file = 'temp_embeddings.pt'
    embeddings_dict = {'spectrum_1': pred_emb.squeeze(0)}  # Remove batch dimension
    torch.save(embeddings_dict, temp_emb_file)
    print(f"Embeddings saved to {temp_emb_file}")
    
    print("\nStep 2: Decoding embeddings to SMILES...")
    
    # Create config for decoder
    class DecodeConfig:
        pass
    config = DecodeConfig()
    config.output_file = output_file
    config.predicted_embeddings = temp_emb_file
    config.model_load = decoder_model
    config.config_load = decoder_config
    config.vocab_load = decoder_vocab
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.n_batch = 1
    config.num_variants = num_variants
    config.model = 'translation'
    
    # Decode embeddings to SMILES
    decode_embeddings(config.model, config)
    
    print(f"\n Done! Predicted SMILES saved to {output_file}")
    print(f"  Generated {num_variants} variants per spectrum")
    
    # Clean up temp file
    if os.path.exists(temp_emb_file):
        os.remove(temp_emb_file)
    
    return output_file


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
        
        return properties
        
    except Exception as e:
        print(f"Error calculating ADME for {smiles}: {e}")
        return None


def analyze_predictions_with_adme(predictions_csv, output_file):
    """
    Read Spec2Mol predictions CSV and add ADME properties to each SMILES
    
    Args:
        predictions_csv: Path to CSV from Spec2Mol (output of decode_embeddings)
        output_file: Path to save results (CSV or XLSX based on extension)
        
    Returns:
        DataFrame with all results
    """
    
    # Read predictions
    df = pd.read_csv(predictions_csv)
    
    all_results = []
    
    # Process each row
    for idx, row in df.iterrows():
        spectrum_id = row['smiles_with_precursor']
        predicted_smiles_list = row['predicted_smiles_list'].split('|')
        
        # Calculate ADME for each predicted variant
        for i, smiles in enumerate(predicted_smiles_list):
            smiles = smiles.strip()
            
            adme_props = calculate_adme_properties(smiles)
            
            if adme_props is not None:
                result_row = {
                    'spectrum_id': spectrum_id,
                    'smiles': smiles,
                    'variant': i + 1,
                    'valid': True,
                    
                    # Druglikeness
                    'lipinski_pass': adme_props['lipinski_pass'],
                    'lipinski_violations': str(adme_props['lipinski_violations']),
                    'egan_pass': adme_props['egan_pass'],
                    'ghose_pass': adme_props['ghose_pass'],
                    'muegge_pass': adme_props['muegge_pass'],
                    'veber_pass': adme_props['veber_pass'],
                    
                    # Pharmacokinetics
                    'gi_absorption': adme_props['gi_absorption'],
                    'bbb_permeant': adme_props['bbb_permeant'],
                    
                    # Medicinal Chemistry
                    'pains_alert': adme_props['pains_alert'],
                    'brenk_alert': adme_props['brenk_alert'],
                    
                    # Molecular Properties
                    'molecular_weight': round(adme_props['molecular_weight'], 2),
                    'logp': round(adme_props['logp'], 2),
                    'tpsa': round(adme_props['tpsa'], 2),
                    'h_bond_donors': adme_props['h_bond_donors'],
                    'h_bond_acceptors': adme_props['h_bond_acceptors'],
                    'n_rotatable_bonds': adme_props['n_rotatable_bonds'],
                    'n_atoms': adme_props['n_atoms'],
                    'n_rings': adme_props['n_rings'],
                }
            else:
                result_row = {
                    'spectrum_id': spectrum_id,
                    'smiles': smiles,
                    'variant': i + 1,
                    'valid': False,
                    'lipinski_pass': None,
                    'lipinski_violations': 'Invalid SMILES',
                }
            
            all_results.append(result_row)
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save to file
    if output_file.endswith('.xlsx'):
        results_df.to_excel(output_file, index=False, engine='openpyxl')
    else:
        results_df.to_csv(output_file, index=False)
    
    print(f"  Results saved to {output_file}")
    print(f"  Total predictions: {len(results_df)}")
    print(f"  Valid SMILES: {results_df['valid'].sum()}")
    if results_df['valid'].sum() > 0:
        lipinski_compliant = results_df[results_df['valid']]['lipinski_pass'].sum()
        print(f"  Lipinski compliant: {lipinski_compliant}")
    
    return results_df

############################# end of ADME ######################################################

################################ Start of Docking ###################################################

def perform_molecular_docking(adme_results_file, output_file='docking_results.csv'):
    """
    Perform molecular docking on valid SMILES from ADME analysis
    
    Args:
        adme_results_file: Path to CSV with ADME results
        output_file: Path to save docking results
    """
    # Read ADME results and filter valid SMILES
    df = pd.read_csv(adme_results_file)
    valid_smiles = df[df['valid'] == True]['smiles'].unique().tolist()
    
    if len(valid_smiles) == 0:
        print("No valid SMILES found for docking.")
        return None
    
    print(f"\nFound {len(valid_smiles)} valid SMILES for docking")
    
    # Ask user if they want to dock
    dock_choice = input("\nWould you like to perform molecular docking? (y/n): ").strip().lower()
    if dock_choice != 'y':
        print("Skipping molecular docking.")
        return None
    
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
                    result = {
                        'smiles': smiles,
                        'target_name': target_name,
                        'docking_score': round(score, 2),
                        'affinity_list': str(info.get('affinities', [])),
                        'num_conformers': info.get('ligand').GetNumConformers() if 'ligand' in info else 0,
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
    results_df = pd.DataFrame(docking_results)
    results_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Docking complete! Results saved to {output_file}")
    
    # Summary
    success_count = results_df[results_df['docking_status'] == 'success'].shape[0]
    print(f"\nSummary:")
    print(f"  Total docking attempts: {len(results_df)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(results_df) - success_count}")
    
    return results_df

################################# End of Docking ####################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MS/MS spectra to SMILES')
    
    # Spectra input files
    parser.add_argument('--pos_low', type=str, required=True,
                        help='CSV file with [M+H]+ low energy spectrum')
    parser.add_argument('--pos_high', type=str, required=True,
                        help='CSV file with [M+H]+ high energy spectrum')
    parser.add_argument('--neg_low', type=str, required=True,
                        help='CSV file with [M-H]- low energy spectrum')
    parser.add_argument('--neg_high', type=str, required=True,
                        help='CSV file with [M-H]- high energy spectrum')
    
    # Model paths (with defaults)
    parser.add_argument('--encoder_model', type=str, default='spectra_encoder.pt',
                        help='Path to trained encoder model')
    parser.add_argument('--decoder_model', type=str, default='decoder/models/model.pt',
                        help='Path to trained decoder model')
    parser.add_argument('--decoder_config', type=str, default='decoder/models/config.nb',
                        help='Path to decoder config')
    parser.add_argument('--decoder_vocab', type=str, default='decoder/models/vocab.nb',
                        help='Path to decoder vocabulary')
    
    # Output
    parser.add_argument('--output', type=str, default='predicted_smiles.csv',
                        help='Output CSV file for predicted SMILES')
    parser.add_argument('--num_variants', type=int, default=10,
                        help='Number of SMILES variants to generate')
    
    parser.add_argument('--adme_output', type=str, default='results.csv',
                        help='Output file with ADME properties (CSV or XLSX)')
    parser.add_argument('--skip_adme', action='store_true',
                        help='Skip ADME analysis')
    
    # Docking options
    parser.add_argument('--docking_output', type=str, default='docking_results.csv',
                        help='Output file for docking results')
    parser.add_argument('--skip_docking', action='store_true',
                        help='Skip molecular docking')
    
    args = parser.parse_args()
    
    # Run Spec2Mol
    predictions_csv = spectra_to_smiles(
        args.pos_low,
        args.pos_high,
        args.neg_low,
        args.neg_high,
        args.encoder_model,
        args.decoder_model,
        args.decoder_config,
        args.decoder_vocab,
        args.output,
        args.num_variants
    )
    
    # Run ADME analysis
    if not args.skip_adme:
        print("\n" + "="*60)
        print("Running ADME Analysis...")
        print("="*60 + "\n")
        analyze_predictions_with_adme(predictions_csv, args.adme_output)
    
            # Add docking step
    if not args.skip_docking:
        print("\n" + "="*60)
        print("Molecular Docking")
        print("="*60)
        perform_molecular_docking(args.adme_output, args.docking_output)