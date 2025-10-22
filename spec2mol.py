
import argparse
import torch
import sys
import os
import json  # ADD THIS LINE

# Import existing modules
from predict_embs import main as predict_embeddings
sys.path.insert(0, 'decoder')
from decode_embeddings import main as decode_embeddings, get_parser as get_decode_parser
from models_storage import ModelsStorage

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
    
    print("Step 1: Converting spectra to embeddings...")
    
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
    
    print(f"\n✓ Done! Predicted SMILES saved to {output_file}")
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


def analyze_predictions_with_adme(predictions_csv, output_json):
    """
    Read Spec2Mol predictions CSV and add ADME properties to each SMILES
    
    Args:
        predictions_csv: Path to CSV from Spec2Mol (output of decode_embeddings)
        output_json: Path to save JSON with ADME properties
        
    Returns:
        List of dictionaries with SMILES and their ADME properties
    """
    import pandas as pd
    import json
    
    # Read predictions
    df = pd.read_csv(predictions_csv)
    
    results = []
    
    # Process each row
    for idx, row in df.iterrows():
        smiles_with_precursor = row['smiles_with_precursor']
        predicted_smiles_list = row['predicted_smiles_list'].split('|')
        
        entry = {
            'spectrum_id': smiles_with_precursor,
            'predictions': []
        }
        
        # Calculate ADME for each predicted variant
        for i, smiles in enumerate(predicted_smiles_list):
            smiles = smiles.strip()
            
            adme_props = calculate_adme_properties(smiles)
            
            prediction_entry = {
                'smiles': smiles,
                'variant_number': i + 1,
                'valid': adme_props is not None,
                'adme_properties': adme_props
            }
            
            entry['predictions'].append(prediction_entry)
        
        results.append(entry)
        print(f"Processed spectrum {idx + 1}/{len(df)}: {len(predicted_smiles_list)} variants")
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ ADME analysis complete! Saved to {output_json}")
    
    # Print summary statistics
    total_predictions = sum(len(entry['predictions']) for entry in results)
    valid_predictions = sum(
        sum(1 for p in entry['predictions'] if p['valid']) 
        for entry in results
    )
    druglike_lipinski = sum(
        sum(1 for p in entry['predictions'] 
            if p['valid'] and p['adme_properties']['lipinski_pass']) 
        for entry in results
    )
    
    print(f"\nSummary:")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Valid SMILES: {valid_predictions}")
    print(f"  Lipinski compliant: {druglike_lipinski} ({druglike_lipinski/valid_predictions*100:.1f}%)")
    
    return results

############################# end of ADME ######################################################

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
    
    parser.add_argument('--adme_output', type=str, default='predictions_with_adme.json',
                        help='Output JSON file with ADME properties')
    parser.add_argument('--skip_adme', action='store_true',
                        help='Skip ADME analysis')
    
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