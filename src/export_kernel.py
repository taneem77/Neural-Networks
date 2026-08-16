# Export the trained NumPy perceptron into a C header
# This connects Python training -> Linux kernel inference

import numpy as np
import os

SCALE = 1000


def export_to_kernel(model_path="saved_model.npz",
                     output_path="kernel/exported_weights.h"):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found")

    data = np.load(model_path)

    weights = data["weights"]
    bias = float(data["bias"][0])

    weights_fixed = (weights * SCALE).astype(int)
    bias_fixed = int(bias * SCALE)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("// AUTO-GENERATED FILE\n")
        f.write("// Generated from Python training\n\n")

        f.write("#ifndef EXPORTED_WEIGHTS_H\n")
        f.write("#define EXPORTED_WEIGHTS_H\n\n")

        f.write(f"#define INPUT_SIZE {len(weights)}\n")
        f.write(f"#define SCALE {SCALE}\n\n")

        f.write("static const int WEIGHTS[] = {")
        f.write(", ".join(map(str, weights_fixed)))
        f.write("};\n\n")

        f.write(f"static const int BIAS = {bias_fixed};\n\n")

        f.write("#endif\n")

    print("Kernel weights exported successfully.")


if __name__ == "__main__":
    export_to_kernel()