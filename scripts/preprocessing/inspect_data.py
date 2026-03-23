from configs.config import load_paths

paths = load_paths()

circuitnet_root = paths["circuitnet_root"]
output_root = paths["output_root"]

print("CircuitNet:", circuitnet_root)
print("Output:", output_root)