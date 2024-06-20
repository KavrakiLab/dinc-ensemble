from typing import List
from pathlib import Path

def create_molstar_ensemble_view(receptor_paths: List[Path],
                                 output_file: Path):
    
    single_structure_template = """
            customData: {{
        url: {},
        format: {{'pdb'}},
        binary: false,
    }}
    }}
"""
    multi_structure_replacement = []
    for r_path in receptor_paths:
        multi_structure_replacement.append(single_structure_template.format(r_path))
    multi_structure_replacement = ",".join(multi_structure_replacement)
    template_path = Path(__file__).parent / "imported.html"
    template_dummy = "STRUCTURE_INPUTS"
    
    with open(template_path, 'r') as f:
        template_lines = f.readlines()
    
    result_lines = []
    for line in template_lines:
        if template_dummy in line:
            line = multi_structure_replacement
        result_lines.append(line)
        
    with open(output_file, 'w') as f:
        f.writelines(result_lines)
        