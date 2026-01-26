"""
Compound name lookup via PubChem API
"""
import requests

def get_compound_name(smiles, timeout=5):
    """
    Get compound name from PubChem using SMILES
    
    Args:
        smiles: SMILES string
        timeout: API request timeout in seconds
        
    Returns:
        Compound name (common name preferred over IUPAC) or "Unknown"
    """
    try:
        # PubChem REST API
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName,Title/JSON"
        
        response = requests.get(url, timeout=timeout)
        
        if response.status_code == 200:
            data = response.json()
            props = data['PropertyTable']['Properties'][0]
            # Prefer Title (common name) over IUPACName
            return props.get('Title', props.get('IUPACName', 'Unknown'))
        else:
            return "Unknown"
            
    except Exception:
        return "Unknown"