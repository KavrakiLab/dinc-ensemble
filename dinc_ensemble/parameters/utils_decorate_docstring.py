from typing import Dict
from dataclasses import dataclass

# extract the doctrings for each argument
def extract_docstr_arguments(docstr: str) -> Dict[str, str]:
    if docstr is None or not isinstance(docstr, str):
        return {}
    docstr_lines = docstr.split("\n")
    # name of the field is always infront of the ":"
    result = {}
    argument_indices = [i for i in range(len(docstr_lines)) if ":" in docstr_lines[i]]
    
    for i, index in enumerate(argument_indices[:]):
        name = docstr_lines[index].split(":")[0].strip()
        if i < len(argument_indices)-1:
            doc_val = " ".join(docstr_lines[argument_indices[i]+1:argument_indices[i+1]])
        else:
            doc_val = " ".join(docstr_lines[argument_indices[i]+1:])
    
        result[name] = " ".join(doc_val.split())
    return result

def docstring_parameter(*sub):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        obj.__doc_arg__ = extract_docstr_arguments(obj.__doc__)
        return obj
    return dec

@dataclass
class ParameterDataclasses():
    __doc_arg__: Dict[str, str]|None = None
    
