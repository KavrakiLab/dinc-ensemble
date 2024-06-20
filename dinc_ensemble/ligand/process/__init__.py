from .preprocess_molecule_strategies import AddHydrogens, AddGasteigerCharges, AddAutodock4Types, \
                                            MergeLonePairs, MergeNonpolarHydrogens

from .preprocess_molecule_registry import PreprocessMoleculeRegistry

#register all available preprocessing functions
PreprocessMoleculeRegistry.register_strategy(MergeLonePairs())
PreprocessMoleculeRegistry.register_strategy(AddHydrogens())
PreprocessMoleculeRegistry.register_strategy(AddGasteigerCharges())
PreprocessMoleculeRegistry.register_strategy(AddAutodock4Types())
PreprocessMoleculeRegistry.register_strategy(MergeNonpolarHydrogens())


__all__ = []
#expose the strategies as individual functions
for strategy_name, strategy in PreprocessMoleculeRegistry.strategies.items():
    f = strategy.to_function()
    globals()[strategy_name] = f
    __all__.append(strategy_name)

#define the global options visible for the parameters list
PREPROCESS_MOLECULE_OPTIONS = PreprocessMoleculeRegistry.strategies.keys()
__all__.append("PREPROCESS_MOLECULE_OPTIONS")

from .preprocess_molecule import preprocess_molecule
__all__.append("preprocess_molecule")
