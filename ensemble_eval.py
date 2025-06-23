import os
import torch
import numpy as np
import pandas as pd
from ase.io import read
from mace.calculators import MACECalculator


print(" IMPORTS OK", flush=True)



print(f"start", flush=True)
# --- Global Test Set ---
test_path = "/home/cat/s233070/MD_pbc/MD_1000_test.xyz"

def find_model_paths(base_folder):
    print(f"Searching for model files in {base_folder}", flush=True)
    model_paths = []
    for root, _, files in os.walk(base_folder):
        for f in files:
            if f.endswith("stagetwo.model") and "run" in os.path.join(root, f):
                model_paths.append(os.path.join(root, f))
    print(f"model_paths: {model_paths}", flush=True)
    return sorted(model_paths)



def run_ensemble_prediction(folder, test_path):
    print(f"\nProcessing folder: {folder}")
    print(f"  Looking for model files...")
    if not os.path.exists(test_path):
        print(f"Test file not found: {test_path}, skipping.")
        return

    model_paths = find_model_paths(folder)
    if not model_paths:
        print(f"No model files found in {folder}, skipping.")
        return

    print(f"[CHECK] Loaded {len(model_paths)} model(s)")

    calculators = [MACECalculator(model_paths=[path], device='cuda', default_dtype='float32') for path in model_paths]


    print(f"[INFO] Reading test file: {test_path}", flush=True)
    #structures = read(test_path, index=":50")
    structures = read(test_path, index=":")
    print(f"[INFO] Loaded {len(structures)} structures", flush=True)

    actual_energies = np.array([atoms.get_potential_energy() for atoms in structures])
    actual_forces = [atoms.get_forces() for atoms in structures]
    natoms_list = np.array([len(atoms) for atoms in structures])

    ensemble_energies = []
    ensemble_forces = []

    for calc in calculators:
        print(f"[CHECK] Starting energy/force predictions using {calc}", flush=True)
        energies = []
        forces = []
        for atoms in structures:
            atoms.calc = calc
            energies.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces())
        ensemble_energies.append(energies)
        ensemble_forces.append(forces)

    ensemble_energies = np.array(ensemble_energies)
    #ensemble_forces = np.array(ensemble_forces)

    mean_energies = np.mean(ensemble_energies, axis=0)
    std_energies = np.std(ensemble_energies, axis=0, ddof=1)

    # Assume ensemble_forces is a list of shape [n_models][n_structures][n_atoms_i x 3]

    mean_forces = []
    std_forces = []

    n_structures = len(ensemble_forces[0])
    n_models = len(ensemble_forces)

    for i in range(n_structures):
        # Collect model predictions for the i-th structure
        force_set = [ensemble_forces[m][i] for m in range(n_models)]  # shape: (n_models, n_atoms, 3)

        # Stack to (n_models, n_atoms, 3)
        stacked = np.stack(force_set, axis=0)

        mean_forces.append(np.mean(stacked, axis=0))  # mean over models
        std_forces.append(np.std(stacked, axis=0, ddof=1))  # std over models




    energy_rmse = np.sqrt(np.mean(((actual_energies / natoms_list) - (mean_energies / natoms_list)) ** 2)) * 1000

    actual_forces_flat = np.concatenate([f.reshape(-1) for f in actual_forces])
    mean_forces_flat = np.concatenate([f.reshape(-1) for f in mean_forces])
    force_rmse = np.sqrt(np.mean((actual_forces_flat - mean_forces_flat) ** 2)) * 1000

    energy_rmse_per_model = []
    force_rmse_per_model = []

    for i in range(len(calculators)):
        e = ensemble_energies[i]
        f_flat = np.concatenate([f.reshape(-1) for f in ensemble_forces[i]])
        e_rmse = np.sqrt(np.mean(((actual_energies / natoms_list) - (np.array(e) / natoms_list)) ** 2)) * 1000
        f_rmse = np.sqrt(np.mean((actual_forces_flat - f_flat) ** 2)) * 1000
        energy_rmse_per_model.append(e_rmse)
        force_rmse_per_model.append(f_rmse)

    rmse_data = {
        "RMSE E (meV/atom)": [energy_rmse],
        "RMSE F (meV/Å)": [force_rmse]
    }
    for i, (e, f) in enumerate(zip(energy_rmse_per_model, force_rmse_per_model)):
        rmse_data[f"Model {i+1} RMSE E (meV/atom)"] = [e]
        rmse_data[f"Model {i+1} RMSE F (meV/Å)"] = [f]

    rmse_df = pd.DataFrame(rmse_data)
    rmse_df.to_csv(os.path.join(folder, "ensemble_rmse.csv"), index=False)

    indices = np.arange(len(structures))
    df = pd.DataFrame({
        "Index": indices,
        "Actual Energy": actual_energies,
        "Mean Energy": mean_energies,
        "Uncertainty Energy": std_energies,
    })
    for i in range(len(calculators)):
        df[f"Model {i+1} Energy"] = ensemble_energies[i]
    df.to_csv(os.path.join(folder, "ensemble_energy_new_structures.csv"), index=False)


    rows = []
    n_models = len(ensemble_forces)

    for idx, (act_f, mean_f, std_f) in enumerate(zip(actual_forces, mean_forces, std_forces)):
        n_atoms = len(act_f)
        for atom_idx in range(n_atoms):
            row = {"Index": idx, "Atom": atom_idx}
            for axis, j in zip(["x", "y", "z"], range(3)):
                row[f"Actual Force {axis}"] = act_f[atom_idx, j]
                row[f"Mean Force {axis}"] = mean_f[atom_idx, j]
                row[f"Uncertainty Force {axis}"] = std_f[atom_idx, j]
                for m in range(n_models):
                    model_force = ensemble_forces[m][idx]
                    if model_force.ndim != 2 or model_force.shape[1] != 3:
                        print(f"[ERROR] Unexpected shape for model {m}, structure {idx}: {model_force.shape}")
                        continue
                    row[f"Model {m+1} Force {axis}"] = model_force[atom_idx, j]
            rows.append(row)


    df_forces = pd.DataFrame(rows)
    df_forces.to_csv(os.path.join(folder, "ensemble_forces_new_structures.csv"), index=False)


    print(f" Saved predictions and RMSEs to {folder}")



base_models_folder = os.path.join(os.getcwd(), "models")

# Run predictions and collect RMSE CSVs
rmse_records = []
for group in sorted(os.listdir(base_models_folder)):
    group_folder = os.path.join(base_models_folder, group)
    if os.path.isdir(group_folder):
        
        print(f"\nProcessing group folder: {group_folder}", flush=True)

        #  RUN PREDICTION
        run_ensemble_prediction(group_folder, test_path)        
        print(f"Finished processing {group_folder}", flush=True)

        #  Then try to load the result
        csv_path = os.path.join(group_folder, "ensemble_rmse.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)

            # Optional: extract info from folder name if structured like "Dimers_100"
            try:
                system, size = group.split("_")
            except ValueError:
                system = group
                size = "N/A"

            df.insert(0, "System", system)
            df.insert(1, "Size", size)
            rmse_records.append(df)

# Concatenate into one overview
if rmse_records:
    overview_df = pd.concat(rmse_records, ignore_index=True)
else:
    overview_df = pd.DataFrame()  # Empty if nothing found

overview_path = os.path.join(base_models_folder, "ensemble_rmse_overview.csv")
overview_df.to_csv(overview_path, index=False)
print(f" Overview saved to {overview_path}", flush=True)
