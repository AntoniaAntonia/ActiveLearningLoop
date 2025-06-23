import os
import re
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from itertools import combinations_with_replacement, product
from sklearn.model_selection import KFold, train_test_split

from ase import Atoms, io
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.data import covalent_radii, atomic_numbers
from ase.io import read, write, Trajectory
from ase.neighborlist import NeighborList, natural_cutoffs

from mace.calculators import MACECalculator
from mace.cli.run_train import run
from mace.tools import build_default_arg_parser


####### CAREFUL! IN TRAIN  args.max_num_epochs = 300


class ActiveLearningLoop:
    def __init__(
        self,
        elements,
        r_max,
        cell_size,
        total_structures,
        dataset_name,
        df_r_min_per_pair,
        nanoparticle_path,
        max_complexity=4,
        n_points=20,
        val_ratio=0.25,
        test_ratio=0.2,           
        seeds=[1, 2, 3, 4, 5],     
        ensemble_size=5,
        uncertainty_threshold=0.3,
        n_folds=5 ,
        rattle_std=0.1,
        start_from=None,
        top_k=None,
        mode=None,
        start_model_name = None  
    ):
        self.elements = elements
        self.r_max = r_max
        self.cell_size = cell_size
        self.total_structures = total_structures
        self.base_dataset_name = dataset_name
        self.df_r_min_per_pair = df_r_min_per_pair
        self.nanoparticle_path = nanoparticle_path
        self.max_complexity = max_complexity
        self.n_points = n_points
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seeds = seeds
        self.ensemble_size = ensemble_size
        self.uncertainty_threshold = uncertainty_threshold
        self.rattle_std = rattle_std
        self.n_folds = n_folds
        self.current_complexity = start_from if start_from is not None else 2
        self.dataset_name = f"{self.base_dataset_name}_complexity_{self.current_complexity}"
        self.top_k = top_k
        self.mode = mode
        self.start_model_name = start_model_name  




    def run(self):
        print(f"\n===== Starting Complexity {self.current_complexity} =====", flush=True)
        self.structure_log_path = os.path.join("data", "info", f"{self.dataset_name}_structure_log.csv")
        os.makedirs(os.path.dirname(self.structure_log_path), exist_ok=True)
        if not os.path.exists(self.structure_log_path):
            with open(self.structure_log_path, "w") as f:
                f.write("complexity,num_structures_added\n")

        # === Step 1: Start mode: begin from dimers and train from scratch ===
        if self.mode == "start":
            print(f"[INFO] Starting from scratch with: {self.dataset_name}", flush=True)

            # 1. Generate dimers
            structure_file = self.generate_structures(
                complexity=self.current_complexity,
                elements=self.elements,
                r_max=self.r_max,
                cell_size=self.cell_size,
                total_structures=self.total_structures,
                n_points=self.n_points,
                dataset_name=self.dataset_name,
                df_r_min_per_pair=self.df_r_min_per_pair,
                nanoparticle_path=self.nanoparticle_path
            )

            if self.current_complexity > 2:
                self.calculate_ener_force(structure_file)

            self.split_test_holdout(filename=f"{structure_file}.xyz", test_ratio=self.test_ratio)


            # Step 4: Cross-validation splits
            self.create_cv_splits(f"data/structures_labeled/{structure_file}_train_val_pool.xyz",
                                  n_folds=self.n_folds)

            # # Step 5: Train initial models
            # print(f"[INFO] Training initial models for {self.dataset_name} with seeds {self.seeds}", flush=True)
            # self.train(self.dataset_name, self.seeds)
            # self.start_model_name = self.dataset_name

            # # Step 6: Summarize training logs
            # self.summarize_latest_finetuning_internal(self.dataset_name)


            self.current_complexity = self.current_complexity + 1

        # === Finetune mode: load pretrained model and finetune ===
        elif self.mode == "finetune":
            print(f"\n===== Starting Finetuning Mode at Complexity {self.current_complexity} =====", flush=True)
            next_complexity = self.current_complexity
            next_name = f"{self.base_dataset_name}_complexity_{next_complexity}"

            self.generate_structures(
                complexity=next_complexity,
                elements=self.elements,
                r_max=self.r_max,
                n_points=self.n_points,
                cell_size=self.cell_size,
                total_structures=self.total_structures,
                dataset_name=next_name,
                df_r_min_per_pair=self.df_r_min_per_pair,
                nanoparticle_path=self.nanoparticle_path
            )

                        # 2. Predict with ensemble from most recent finetuned models
            print(f"[DEBUG] Finetune mode enabled. Searching for finetuned models...", flush=True)
            finetune_dirs = [
                d for d in os.listdir("models")
                if "finetuning" in d and self.base_dataset_name in d
            ]
            best_finetune_num = -1
            best_complexity = -1
            best_dir = None

            for d in finetune_dirs:
                print(f"1", flush=True)
                match = re.search(r"finetuning_(\d+)_" + re.escape(self.base_dataset_name) + r"_complexity_(\d+)", d)
                if match:
                    finetune_num = int(match.group(1))
                    complexity_num = int(match.group(2))
                    if (finetune_num > best_finetune_num) or (
                        finetune_num == best_finetune_num and complexity_num > best_complexity
                    ):
                        best_finetune_num = finetune_num
                        best_complexity = complexity_num
                        best_dir = d

            if best_dir is None:
                raise RuntimeError(f"No previous finetuned model found for base {self.base_dataset_name}")

            print(f"[DEBUG] Best finetune found: {best_dir} (num: {best_finetune_num}, complexity: {best_complexity})", flush=True)


            #resume_from_dataset = best_dir.replace(f"finetuning_{best_finetune_num}_", "")
            #resume_from_dataset = best_dir
            resume_folder = best_dir  # e.g. 'finetuning_1_example_3_complexity_4'
            resume_dataset = best_dir.replace(f"finetuning_{best_finetune_num}_", "")  # -> 'example_3_complexity_4'
                        
            print(f"[INFO] Resuming finetuning from: {best_dir}", flush=True)


            self.ensemble_generation_prediction_and_uncertainty(
                complexity_runNr=resume_folder,
                predict_structures=self.dataset_name,
                output_name=self.dataset_name
            )
            
            print(f"after ensemble_generation_prediction_and_uncertainty in if finetune run", flush=True)

            print(f"[DEBUG] looking for num_uncertain in ensemble_forces_{self.dataset_name}.csv", flush=True)

            self.extract_most_uncertain_structures(
                csv_file=f"ensemble_forces_{self.dataset_name}.csv",
                xyz_file=f"{self.dataset_name}.xyz",
                unc_threshold=None if self.top_k is not None else self.uncertainty_threshold,
                top_k=self.top_k
            )

            self.calculate_ener_force(f"{self.dataset_name}_most_uncertain")
            
            self.split_test_holdout(f"{self.dataset_name}_most_uncertain.xyz", test_ratio=self.test_ratio)
            
            self.append_uncertain_to_sets(
                train_uncertain_filename=f"{self.dataset_name}_most_uncertain_train_val_pool.xyz",
                val_uncertain_filename=f"{self.dataset_name}_most_uncertain_train_val_pool.xyz"
            )
            
            self.finetune_from_existing_models(
                complexity_runNr=resume_dataset,
                test_file=f"{self.dataset_name}_most_uncertain_test_set.xyz",
                output_model_id=self.dataset_name
            )

            self.summarize_latest_finetuning_internal(self.dataset_name)

            self.current_complexity = self.current_complexity + 1

        # === Step 3: Main active learning loop ===
        while self.current_complexity < self.max_complexity:
            print(f"\nenter while loop ", flush=True)
            print(f"\n===== Starting Complexity {self.current_complexity} =====")
            next_name = f"{self.base_dataset_name}_complexity_{self.current_complexity}"

            print(f"next_name:{next_name}", flush=True)

            self.generate_structures(
                complexity=self.current_complexity,
                elements=self.elements,
                r_max=self.r_max,
                n_points=self.n_points,
                cell_size=self.cell_size,
                total_structures=self.total_structures,
                dataset_name=next_name,
                df_r_min_per_pair=self.df_r_min_per_pair,
                nanoparticle_path=self.nanoparticle_path
            )

            previous_complexity = self.current_complexity - 1


            # Predict using latest finetuned or baseline model depending on mode
            if self.mode == "start" and self.start_model_name is not None:
                latest_model_dir = os.path.join(self.dataset_name)
                previous_name = self.start_model_name
                print("1", flush=True)	
            else:
                print("2", flush=True)
                latest_model_dir = self.get_latest_model_dir(self.base_dataset_name)
            
            if latest_model_dir is None:
                raise RuntimeError(f"[ERROR] No model found for {self.dataset_name}")

            print(f"[DEBUG] Using model {latest_model_dir} to predict on structures for {next_name}", flush=True)


            # if latest_model_dir.startswith("finetuning_"):
            #     model_dataset_name = latest_model_dir.split("_", maxsplit=2)[-1]
            #     model_dataset_name = "_".join(latest_model_dir.split("_")[2:])  # safer
            # else:
            #     model_dataset_name = previous_name



            self.ensemble_generation_prediction_and_uncertainty(latest_model_dir, next_name)
            self.extract_most_uncertain_structures(
                csv_file=f"ensemble_forces_{next_name}.csv",
                xyz_file=f"{next_name}.xyz",
                unc_threshold=None if self.top_k is not None else self.uncertainty_threshold,
                top_k=self.top_k
            )


            self.calculate_ener_force(f"{next_name}_most_uncertain")
            self.split_test_holdout(f"{next_name}_most_uncertain.xyz", test_ratio=self.test_ratio)
            self.append_uncertain_to_sets(
                train_uncertain_filename=f"{next_name}_most_uncertain_train_val_pool.xyz",
                val_uncertain_filename=f"{next_name}_most_uncertain_train_val_pool.xyz"
            )

            # latest_model is something like 'finetuning_3_example_3_complexity_4'
            self.finetune_from_existing_models(
                #complexity_runNr=os.path.basename(latest_model),  # important!
                complexity_runNr=latest_model_dir,
                test_file=f"{next_name}_most_uncertain_test_set.xyz",
                output_model_id=self.dataset_name
            )


            self.summarize_latest_finetuning_internal(self.dataset_name)

            print("[INFO] One round of active learning complete. Continuing loop...")
            self.current_complexity = self.current_complexity +1
            self.dataset_name = next_name
            self.mode = None


        # After completing all complexities, do one last round with bulk data
        print("\n[INFO] Active learning loop complete. Starting bulk finetuning step.")

        bulk_name = f"{self.base_dataset_name}_complexity_bulk"
        latest_model_dir = self.get_latest_model_dir(self.base_dataset_name)

        if latest_model_dir is None:
            raise RuntimeError(f"[ERROR] No model found for final finetuning on bulk data")

        bulk_xyz = "bulk.xyz"
        self.split_test_holdout(bulk_xyz, test_ratio=self.test_ratio)
        self.append_uncertain_to_sets(
            train_uncertain_filename="bulk_train_val_pool.xyz",
            val_uncertain_filename="bulk_train_val_pool.xyz"
        )
        
        self.finetune_from_existing_models(
            complexity_runNr=latest_model_dir,
            test_file="bulk_test_set.xyz",
            output_model_id=bulk_name
        )

        self.summarize_latest_finetuning_internal(bulk_name)
        print("[INFO] Bulk finetuning step complete.")







    def generate_structures(
        self,
        complexity,
        elements,
        r_max,
        cell_size,
        n_points,
        total_structures=None,
        dataset_name=None,
        df_r_min_per_pair=None,
        nanoparticle_path=None,
        output_prefix="cutout",
        mult_cutoff=1.1,
        rattle_std=0.1
    ):
        """
        Generate atomic structures of given complexity (2 = dimers, 3 = trimers, 4 = quadrumers, >=5 = general nmers).
        
        Args:
            complexity (int): Number of atoms in the structure.
            elements (list of str): Elements to consider.
            r_max (float): Maximum bond length.
            cell_size (float): Box size for generated structures.
            total_structures (int): Total structures to generate (used for complexity >= 3).
            n_points (int): Structures per dimer pair (used for dimers).
            dataset_name (str): Name of dataset prefix.
            df_r_min_per_pair (pd.DataFrame): Pairwise distance threshold table.
            nanoparticle_path (str): Path to nanoparticle file for nmers >= 5.
            output_prefix (str): Output file prefix for nmers.
            mult_cutoff (float): Multiplier for ASE natural cutoff.
            rattle_std (float): Standard deviation for atomic rattling.
        """
        print(f"[DEBUG] complexity type: {type(complexity)} value: {complexity}")

        def create_initial_dimer_dataset(elements, r_max, n_points, cell_size):

            """
            Generates labeled dimer structures using EMT by varying the interatomic distance 
            for all combinations of the provided elements. Also determines the minimal repulsive 
            distance (`r_min`) for each dimer based on a repulsion energy threshold.

            For each element pair:
                - The function scans for the closest distance where repulsion becomes energetically significant.
                - Generates dimer structures across a range of distances (inverse-sampled between r_min and r_max).
                - Computes energy and maximum force norm using the EMT calculator.

            Additionally, it:
                - Saves isolated atom structures with energies.
                - Saves the generated dimers to .xyz and .traj formats.
                - Writes energy and force summary CSVs.
                - Saves r_min values to data/info/r_min_per_pair.csv.

            Args:
                elements (list of str): List of element symbols (e.g., ["C", "O", "Pt"]).
                r_max (float): Maximum dimer bond length to consider (Å).
                n_points (int): Number of distance points per dimer to sample.
                cell_size (float): Box size used to enclose each dimer (cube of cell_size³).

            Returns:
                str: Path to the generated .xyz file containing all labeled dimers.
            """


            pairs = list(combinations_with_replacement(elements, 2))

            isolated_atoms = []
            isolated_atoms_energies = {}
            for element in elements:
                atom = Atoms(element, positions=[[0, 0, 0]], cell=[cell_size]*3, pbc=False)
                atom.center()
                atom.calc = EMT()
                energy = atom.get_potential_energy()
                isolated_atoms_energies[element] = energy
                atom.info["config_type"] = "IsolatedAtom"
                isolated_atoms.append(atom)

            os.makedirs("data/structures_only", exist_ok=True)
            os.makedirs("data/structures_labeled", exist_ok=True)
            os.makedirs("data/info", exist_ok=True)

            write("data/structures_labeled/isolated_atoms.xyz", isolated_atoms)
            print("isolated_atoms_energies:" , isolated_atoms_energies)
            print("---- Saved isolated atoms to data/isolated_atoms.xyz ----")


            r0_lengths = {
                tuple(sorted(p)): covalent_radii[atomic_numbers[p[0]]] + covalent_radii[atomic_numbers[p[1]]]
                for p in pairs
            }

            dataset_name = f"dimers_start"

            xyz_file_dimers = f"data/structures_labeled/{dataset_name}.xyz"
            traj_file_dimers = f"data/structures_labeled/{dataset_name}.traj"


            structures = []
            data = []
            force_norm_data = []
            r_min_per_pair = []

            force_norm_max = 100
            threshold_start = 2.0  # pair potential starts (repulsive) at isolated_atoms_energy + 2.0 eV


            with Trajectory(traj_file_dimers, 'w') as traj:
                for (atom1, atom2), r0 in r0_lengths.items():
                    
                    # === STEP 1: Find r_min ===
                    scan_step_factor = 0.97  # scale for decreasing/increasing distance
                    initial_r = 0.7 * r0
                    threshold_energy = isolated_atoms_energies[atom1] + isolated_atoms_energies[atom2] + threshold_start

                    # Get initial energy
                    r = initial_r
                    atoms = Atoms(f"{atom1}{atom2}", positions=[[0, 0, 0], [r, 0, 0]])
                    atoms.set_cell([cell_size, cell_size, cell_size])
                    atoms.center()
                    atoms.set_pbc(False)
                    atoms.calc = EMT()
                    energy = atoms.get_potential_energy()

                    if abs(energy - threshold_energy) < 1e-6:
                        r_min = r

                    elif energy < threshold_energy:
                        # Decrease r until energy >= threshold
                        while r > 0.3:
                            r *= scan_step_factor  # make r smaller
                            atoms = Atoms(f"{atom1}{atom2}", positions=[[0, 0, 0], [r, 0, 0]])
                            atoms.set_cell([cell_size, cell_size, cell_size])
                            atoms.center()
                            atoms.set_pbc(False)
                            atoms.calc = EMT()
                            energy = atoms.get_potential_energy()
                            if energy >= threshold_energy:
                                r_min = r
                                break
                        else:
                            r_min = r  # fallback

                    elif energy > threshold_energy:
                        prev_r = r
                        while r < r_max:
                            r *= 1.0 / scan_step_factor  # make r larger
                            atoms = Atoms(f"{atom1}{atom2}", positions=[[0, 0, 0], [r, 0, 0]])
                            atoms.set_cell([cell_size, cell_size, cell_size])
                            atoms.center()
                            atoms.set_pbc(False)
                            atoms.calc = EMT()
                            energy = atoms.get_potential_energy()
                            if energy <= threshold_energy:
                                r_min = prev_r
                                break
                            prev_r = r
                        else:
                            r_min = prev_r  # fallback


                    print(f"{atom1}-{atom2}: r_min = {r_min:.3f} Å, threshold = {threshold_energy:.3f} eV")
                    r_min_per_pair.append([atom1, atom2, r_min])

                    inv_r_space = np.linspace(1 / r_max, 1 / r_min, n_points)
                    distances = 1 / inv_r_space

                    for d in distances:
                        atoms = Atoms(f"{atom1}{atom2}", positions=[[0, 0, 0], [d, 0, 0]])
                        atoms.set_cell([cell_size, cell_size, cell_size])
                        atoms.center()
                        atoms.set_pbc(False)
                        atoms.calc = EMT()
                        energy = atoms.get_potential_energy()
                        forces = atoms.get_forces()
                        force_norm = np.linalg.norm(forces, axis=1).max()
                        
                        if force_norm < force_norm_max:
                            traj.write(atoms)
                            structures.append(atoms)
                            data.append([atom1, atom2, d, energy])
                            force_norm_data.append([atom1, atom2, d, force_norm])
                            

                        
            write(xyz_file_dimers, structures)

            df = pd.DataFrame(data, columns=["Atom 1", "Atom 2", "Distance (Å)", "Energy (eV)"])
            df_force = pd.DataFrame(force_norm_data, columns=["Atom 1", "Atom 2", "Distance (Å)", "Force Norm"])

            df_r_min_per_pair = pd.DataFrame(r_min_per_pair, columns=["Atom 1", "Atom 2", "closest distance"])
            df_r_min_per_pair.to_csv("data/info/r_min_per_pair.csv")

            print(df_r_min_per_pair)
            print(df.shape)



            log_path = os.path.join("data", "info", "number_uncertainty_log.csv")

            log_entry = {
                "threshold": None,
                "xyz_file": xyz_file_dimers,
                "complexity": 2,
                "num_uncertain": len(structures)
            }

            # Append or create log
            log_exists = os.path.isfile(log_path)
            log_df = pd.DataFrame([log_entry])

            if log_exists:
                log_df.to_csv(log_path, mode='a', header=False, index=False)
            else:
                log_df.to_csv(log_path, index=False)

            print(f"Log entry added to: {log_path}")

            return "dimers_start"


        def generate_trimer_structures(dataset_name, elements, r_max, total_structures, cell_size, df_r_min_per_pair):
            output_folder = "data/structures_only"
            os.makedirs(output_folder, exist_ok=True)

            triplets = list(product(elements, repeat=3))
            n_triplets = len(set((min(a1, a3), a2, max(a1, a3)) for a1, a2, a3 in triplets))
            n_structures_per_triplet = int(np.ceil(total_structures / n_triplets))

            sorted_structures = set()
            traj_file = os.path.join(output_folder, f"{dataset_name}.traj")
            xyz_file = os.path.join(output_folder, f"{dataset_name}.xyz")

            structures = []
            data = []
            print(f"generationg {total_structures} trimer structures")


            def get_check_distance(atom1, atom3, df):
                a1, a3 = sorted([atom1, atom3])
                row = df[(df["Atom 1"] == a1) & (df["Atom 2"] == a3)]
                if not row.empty:
                    return float(row["closest distance"].values[0])
                else:
                    raise ValueError(f"No check_distance found for pair ({a1}, {a3})")

            def sample_inverse_weighted(min_r, max_r, n, softness=0.07, power=0.95):
                r_range = np.linspace(min_r + softness, max_r, 1000)
                weights = 1 / (r_range - min_r + softness) ** power
                weights /= weights.sum()
                return np.random.choice(r_range, size=n, replace=False, p=weights)



            with Trajectory(traj_file, 'w') as traj:
                for (atom1, atom2, atom3) in triplets:
                    sorted_triplet = (min(atom1, atom3), atom2, max(atom1, atom3))
                    if sorted_triplet in sorted_structures:
                        continue

                    min_12 = get_check_distance(atom1, atom2, df_r_min_per_pair)
                    min_23 = get_check_distance(atom2, atom3, df_r_min_per_pair)
                    min_13 = get_check_distance(atom1, atom3, df_r_min_per_pair)

                    generated = 0
                    attempts = 0
                    max_attempts = n_structures_per_triplet * 10

                    while generated < n_structures_per_triplet and attempts < max_attempts:
                        r1 = sample_inverse_weighted(min_12, r_max, 1)[0]
                        r2 = sample_inverse_weighted(min_23, r_max, 1)[0]
                        theta = np.random.uniform(10, 180)
                        theta_rad = np.radians(180 - theta)

                        positions = np.array([
                            [-r1, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [r2 * np.cos(theta_rad), r2 * np.sin(theta_rad), 0.0]
                        ])

                        atoms = Atoms(f"{atom1}{atom2}{atom3}", positions=positions)
                        atoms.set_cell([cell_size] * 3)
                        atoms.center()
                        atoms.set_pbc(False)
                        atoms.info["angle_deg"] = round(theta, 2)

                        r_pair = tuple(sorted((round(r1, 3), round(r2, 3)))) if atom1 == atom3 else (round(r1, 3), round(r2, 3))
                        structure_id = (sorted_triplet, r_pair, round(theta, 2))

                        if structure_id in sorted_structures:
                            attempts += 1
                            continue

                        distance_13 = atoms.get_distance(0, 2)
                        if distance_13 > min_13:
                            traj.write(atoms)
                            structures.append(atoms)
                            data.append([
                                *sorted_triplet,
                                round(r1, 3), round(r2, 3),
                                round(theta, 2)
                            ])
                            sorted_structures.add(structure_id)
                            generated += 1
                        else:
                            attempts += 1

                    if attempts >= max_attempts:
                        print(f"⚠️ Max attempts reached for {atom1}-{atom2}-{atom3}.")

            # Filter if more than requested
            if len(structures) > total_structures:
                indices = np.linspace(0, len(structures) - 1, total_structures, dtype=int)
                structures = [structures[i] for i in indices]
                data = [data[i] for i in indices]

            print(f"\nSaved {len(structures)} trimer structures as .traj")

            # Save as XYZ
            write(xyz_file, structures)
            print(f"Saved as .xyz: {xyz_file}")

            # Save CSV
            df = pd.DataFrame(data, columns=[
                "Atom 1", "Atom 2", "Atom 3",
                "R1 (Å)", "R2 (Å)", "Angle (°)"
            ])
            df.to_csv(os.path.join("data", "info",f"{dataset_name}.csv"), index=False)

            return f"{dataset_name}"
            



        def generate_quadrumer_structures(dataset_name, elements, r_max, total_structures, cell_size, df_r_min_per_pair):

            """
            Generates a dataset of random quadrumer (4-atom) structures with geometrically diverse configurations.
            
            For each element combination (with replacement), the function:
                - Samples 3 interatomic distances (r1, r2, r3) using inverse weighting.
                - Samples two angles and one dihedral angle to control geometry.
                - Builds the corresponding ASE Atoms object using geometric placement.
                - Checks that all interatomic distances satisfy element-specific minimum distance constraints.
                - Saves accepted structures in .traj and .xyz format and logs metadata to CSV.

            Args:
                dataset_name (str): Prefix used to name output .xyz, .traj, and .csv files.
                elements (List[str]): List of chemical elements to consider in quadrumer combinations.
                r_max (float): Maximum bond length used in sampling (Å).
                total_structures (int): Total number of quadrumers to generate (across all combinations).
                cell_size (float): Size of the cubic simulation box.
                df_r_min_per_pair (pd.DataFrame): DataFrame containing closest allowed distances between elements,
                                                with columns ["Atom 1", "Atom 2", "closest distance"].

            Returns:
                str: Dataset name prefix (same as input `dataset_name`).
            """


            output_folder = "data/structures_only"
            os.makedirs(output_folder, exist_ok=True)

            quads = list(combinations_with_replacement(elements, 4))
            n_structures_per_quad = int(np.ceil(total_structures / len(quads)))
            traj_file = os.path.join(output_folder, f"{dataset_name}.traj")
            xyz_file = os.path.join(output_folder, f"{dataset_name}.xyz")

            print(f"generationg {total_structures} quadrumer structures", flush=True)

            data = []
            structures = []



            def get_check_distance(atom1, atom2, df):
                a1, a2 = sorted([atom1, atom2])
                row = df[(df["Atom 1"] == a1) & (df["Atom 2"] == a2)]
                if not row.empty:
                    return float(row["closest distance"].values[0])
                else:
                    raise ValueError(f"No check_distance found for pair ({a1}, {a2})")


            def sample_inverse_weighted(min_r, max_r, n, softness=0.07, power=0.95):
                r_range = np.linspace(min_r + softness, max_r, 1000)
                weights = 1 / (r_range - min_r + softness) ** power
                weights /= weights.sum()
                return np.random.choice(r_range, size=n, replace=False, p=weights)

            def build_quadrumer(atom0, atom1, atom2, atom3, r1, r2, r3, angle1_deg, angle2_deg, dihedral_deg, cell_size):

                # Atom1 at origin
                p1 = np.array([0.0, 0.0, 0.0])
                # Atom0 along -x axis (bond r1)
                p0 = np.array([-r1, 0.0, 0.0])
                # Atom2 in x-y plane (bond r2, angle angle1)
                theta1 = np.radians(angle1_deg)
                p2 = p1 + np.array([r2 * np.cos(theta1), r2 * np.sin(theta1), 0.0])
                # Atom3: placed along z, rotated later
                p3 = p2 + np.array([0.0, 0.0, r3])

                atoms = Atoms([atom0, atom1, atom2, atom3], positions=[p0, p1, p2, p3])
                atoms.set_cell([cell_size] * 3)
                atoms.center()
                atoms.set_pbc(False)

                # Apply angle and dihedral
                atoms.set_angle(1, 2, 3, angle2_deg)
                atoms.set_dihedral(0, 1, 2, 3, dihedral_deg)

                atoms.info.update({
                    "angle1_deg": angle1_deg,
                    "angle2_deg": angle2_deg,
                    "dihedral_deg": dihedral_deg
                })
                return atoms




            with Trajectory(traj_file, 'w') as traj:
                for base_quad in quads:
                    generated = 0
                    attempts = 0
                    max_attempts = n_structures_per_quad * 15

                    while generated < n_structures_per_quad and attempts < max_attempts:
                        quad = list(base_quad)
                        random.shuffle(quad)
                        atom0, atom1, atom2, atom3 = quad

                        try:
                            min_01 = get_check_distance(atom0, atom1, df_r_min_per_pair)
                            min_12 = get_check_distance(atom1, atom2, df_r_min_per_pair)
                            min_23 = get_check_distance(atom2, atom3, df_r_min_per_pair)
                        except ValueError as e:
                            print(f"Skipping due to missing pair distance: {e}")
                            break

                        r1 = sample_inverse_weighted(min_01, r_max, 1)[0]
                        r2 = sample_inverse_weighted(min_12, r_max, 1)[0]
                        r3 = sample_inverse_weighted(min_23, r_max, 1)[0]

                        angle1 = np.random.uniform(10, 170)
                        angle2 = np.random.uniform(10, 170)
                        dihedral = np.random.uniform(-180, 180)

                        atoms = build_quadrumer(atom0, atom1, atom2, atom3, r1, r2, r3, angle1, angle2, dihedral, cell_size)

                        all_dists = [atoms.get_distance(i, j) for i in range(4) for j in range(i + 1, 4)]
                        check_dists = [
                            get_check_distance(a1, a2, df_r_min_per_pair)
                            for i, a1 in enumerate(quad) for j, a2 in enumerate(quad)
                            if j > i
                        ]

                        if all(d > cd for d, cd in zip(all_dists, check_dists)):
                            traj.write(atoms)
                            mean_d = np.mean(all_dists)
                            data.append([
                                *quad, r1, r2, r3, angle1, angle2, dihedral, mean_d
                            ])
                            structures.append(atoms)
                            generated += 1
                        else:
                            attempts += 1

                    if attempts >= max_attempts:
                        print(f"Max attempts reached for {base_quad}")

            # Downsample if needed
            if len(structures) > total_structures:
                indices = np.linspace(0, len(structures) - 1, total_structures, dtype=int)
                structures = [structures[i] for i in indices]
                data = [data[i] for i in indices]

            print(f"\n Saved {len(structures)} quadrumer structures to .traj", flush=True)

            # Save structures and data
            write(xyz_file, structures)
            print(f" Saved .xyz: {xyz_file}", flush=True)

            df = pd.DataFrame(data, columns=[
                "Atom 0", "Atom 1", "Atom 2", "Atom 3",
                "R1", "R2", "R3",
                "Angle1", "Angle2", "Dihedral", "Mean Pairwise Distance"
            ])
            df.to_csv(os.path.join(output_folder, f"{dataset_name}.csv"), index=False)
            print(f" Saved CSV: {dataset_name}.csv")

            structure_log_path = os.path.join("data", "info", f"{dataset_name}_structure_log.csv")
            os.makedirs(os.path.dirname(structure_log_path), exist_ok=True)

            if not os.path.exists(structure_log_path):
                with open(structure_log_path, "w") as f:
                    f.write("complexity,num_structures_added\n")

            with open(structure_log_path, "a") as f:
                f.write(f"dimers,{len(structures)}\n")

            print(f"Logged {len(structures)} dimers to {structure_log_path}")



            return f"{dataset_name}"









        def extract_random_nmers(
            nanoparticle: Atoms,
            total_structures: int,
            cluster_size: int = 5,
            output_prefix: str = "cutout",
            mult_cutoff: float = 1.1,
            output_dir: str = "data/structures_only",
            rattle_std: float = 0.1
        ):
            """
            Extracts random n-mer (e.g., pentamer if cluster_size=5) substructures from a nanoparticle,
            and adds a small random displacement ("rattle") to each structure.

            Args:
                nanoparticle (Atoms): ASE Atoms object (e.g., a nanoparticle).
                total_structures (int): Number of clusters to extract.
                cluster_size (int): Desired number of atoms per cluster.
                output_prefix (str): File prefix (e.g., "cutout").
                mult_cutoff (float): Multiplier for ASE natural cutoff radii.
                output_dir (str): Output directory for .xyz and .traj files.
                rattle_std (float): Standard deviation (Å) for rattling atom positions.
            """
            assert cluster_size >= 2, "Cluster size must be at least 2 atoms."

            cutoffs = natural_cutoffs(nanoparticle, mult=mult_cutoff)
            nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
            nl.update(nanoparticle)

            total_atoms = len(nanoparticle)
            unique_keys = set()
            all_clusters = []
            attempts = 0

            print(f"Generating {total_structures} {cluster_size}-mer structures...")

            while len(all_clusters) < total_structures and attempts < 20 * total_structures:
                seed_idx = random.randint(0, total_atoms - 1)

                indices, offsets = nl.get_neighbors(seed_idx)
                if len(indices) < cluster_size - 1:
                    attempts += 1
                    continue

                distances = np.linalg.norm(nanoparticle.positions[indices] - nanoparticle.positions[seed_idx], axis=1)
                sorted_neighbors = [i for _, i in sorted(zip(distances, indices))][:cluster_size - 1]

                cluster_indices = [seed_idx] + sorted_neighbors
                cluster_indices.sort()
                cluster_key = tuple(cluster_indices)

                if cluster_key in unique_keys:
                    attempts += 1
                    continue

                unique_keys.add(cluster_key)
                cluster_atoms = nanoparticle[cluster_indices].copy()
                cluster_atoms.info['origin_indices'] = cluster_indices
                cluster_atoms.center()
                all_clusters.append(cluster_atoms)

                print(f"Collected {cluster_size}-mer {len(all_clusters)}: atoms {cluster_indices}")

            if len(all_clusters) < total_structures:
                print(f" Warning: Only {len(all_clusters)} unique clusters could be extracted after {attempts} attempts.")

            # Rattle structures
            print(f"🔧 Rattling all {len(all_clusters)} structures with std={rattle_std} Å...")
            for atoms in all_clusters:
                atoms.rattle(stdev=rattle_std)

            # Save output
            os.makedirs(output_dir, exist_ok=True)
            # base_name = f"{output_prefix}_{cluster_size}_atoms"
            # xyz_path = os.path.join(output_dir, f"{base_name}.xyz")
            # traj_path = os.path.join(output_dir, f"{base_name}.traj")
            dataset_name = f"{self.base_dataset_name}_complexity_{complexity}"
            xyz_path = os.path.join("data", "structures_only", f"{dataset_name}.xyz")
            traj_path = os.path.join("data", "structures_only", f"{dataset_name}.traj")




            write(xyz_path, all_clusters)
            write(traj_path, all_clusters)

            print(f" Saved {len(all_clusters)} rattled {cluster_size}-mers to:")
            print(f"   {xyz_path}")
            print(f"   {traj_path}")

            return dataset_name


        if complexity == 2:
            print(" Generating dimers...")
            if n_points is None:
                raise ValueError("`n_points` is required for dimers.")
            #create_initial_dimer_dataset(elements, r_max, n_points, cell_size)
            return create_initial_dimer_dataset(elements, r_max, n_points, cell_size)

        elif complexity == 3:
            print(" Generating trimers...")
            if dataset_name is None or total_structures is None or df_r_min_per_pair is None:
                raise ValueError("Missing one of: dataset_name, total_structures, df_r_min_per_pair for trimers.")
            #generate_trimer_structures(dataset_name, elements, r_max, total_structures, cell_size, df_r_min_per_pair)
            return generate_trimer_structures(dataset_name, elements, r_max, total_structures, cell_size, df_r_min_per_pair)


        elif complexity == 4:
            print(" Generating quadrumers...")
            if dataset_name is None or total_structures is None or df_r_min_per_pair is None:
                raise ValueError("Missing one of: dataset_name, total_structures, df_r_min_per_pair for quadrumers.")
            #generate_quadrumer_structures(dataset_name, elements, r_max, total_structures, cell_size, df_r_min_per_pair)
            #return generate_quadrumer_structures(dataset_name, elements, r_max, total_structures, cell_size, df_r_min_per_pair)
            return generate_quadrumer_structures(dataset_name, elements, r_max, total_structures, cell_size, df_r_min_per_pair)



        elif complexity > 4:
            print(f" Extracting {complexity}-mers from nanoparticle...")
            if nanoparticle_path is None:
                raise ValueError("`nanoparticle_path` must be provided for complexity >= 5")
            if total_structures is None:
                raise ValueError("`total_structures` must be specified for complexity >= 5")

            nanoparticle = read(nanoparticle_path)
            # extract_random_nmers(
            #     nanoparticle=nanoparticle,
            #     total_structures=total_structures,
            #     cluster_size=complexity,
            #     output_prefix=output_prefix,
            #     mult_cutoff=mult_cutoff,
            #     output_dir="data/structures_only",
            #     rattle_std=rattle_std
            # )
            return extract_random_nmers(
                nanoparticle=nanoparticle,
                total_structures=total_structures,
                cluster_size=complexity,
                output_prefix=output_prefix,
                mult_cutoff=mult_cutoff,
                output_dir="data/structures_only",
                rattle_std=rattle_std
            )


        else:
            raise ValueError("Complexity must be an integer >= 2.")



    def calculate_ener_force(self, structure_filename):

        """
        Calculates potential energy and forces for a set of atomic structures using the EMT calculator.
        The labeled structures are saved both as .traj and .xyz files in data/structures_labeled/.

        Args:
            structure_filename (str): Name of the input .xyz file (without extension) located in data/structures_only.
                                    For example, 'quadrumers_1' refers to 'data/structures_only/quadrumers_1.xyz'.
        """

        input_folder = "data/structures_only"
        output_folder = "data/structures_labeled"
        os.makedirs(output_folder, exist_ok=True)

        # Correct paths
        input_path = os.path.join(input_folder, structure_filename + ".xyz")
        traj_file = os.path.join(output_folder, structure_filename + ".traj")
        xyz_file = os.path.join(output_folder, structure_filename + ".xyz")

        if os.path.exists(traj_file):
            os.remove(traj_file)
        if os.path.exists(xyz_file):
            os.remove(xyz_file)

        # Read all structures in the trajectory file
        structures = io.read(input_path, index=":")


        with Trajectory(traj_file, 'w') as traj:
            for i, atoms in enumerate(structures):
                atoms.calc = EMT()
                atoms.get_potential_energy()
                traj.write(atoms, append=True)
                atoms.write(xyz_file, append=True)
                if i % 500 == 0:
                    print(f"Processed {i}/{len(structures)} structures")


    
        print(f"Labeled {len(structures)} structures — saved to '{traj_file}' and '{xyz_file}'")




    def append_uncertain_to_sets(self, train_uncertain_filename, val_uncertain_filename):

        """
        Appends uncertainty-selected structures to existing train/validation splits (train_cv_*.xyz and val_cv_*.xyz).
        
        If a current dataset directory (data/structures_labeled/current_dataset) exists, it uses that as the base.
        Otherwise, it starts from the initial splits in data/structures_labeled/initial and creates the current dataset.

        The updated splits are saved in: data/structures_labeled/current_dataset/

        Args:
            train_uncertain_filename (str): Filename of the .xyz file containing uncertain training structures.
            val_uncertain_filename (str): Filename of the .xyz file containing uncertain validation structures.
        """



        base_dir = os.path.join("data", "structures_labeled")
        current_dataset_dir = os.path.join(base_dir, "current_dataset")
        initial_dataset_dir = os.path.join(base_dir, "initial")

        # Determine source directory for train/val cv files
        source_dir = current_dataset_dir if os.path.exists(current_dataset_dir) else initial_dataset_dir
        if not os.path.exists(current_dataset_dir):
            os.makedirs(current_dataset_dir)

        # Full paths to uncertainty files
        train_unc_path = os.path.join(base_dir, train_uncertain_filename)
        val_unc_path = os.path.join(base_dir, val_uncertain_filename)

        # Load uncertain structures
        train_unc = read(train_unc_path, index=":")
        val_unc = read(val_unc_path, index=":")

        # Scan for all train_cv and val_cv files
        all_files = os.listdir(source_dir)
        train_cv_files = sorted(f for f in all_files if f.startswith("train_cv_") and f.endswith(".xyz"))
        val_cv_files = sorted(f for f in all_files if f.startswith("val_cv_") and f.endswith(".xyz"))

        # Append uncertain train data to each train_cv file
        for fname in train_cv_files:
            full_path = os.path.join(source_dir, fname)
            structures = read(full_path, index=":")
            combined = structures + train_unc
            output_path = os.path.join(current_dataset_dir, fname)
            write(output_path, combined)
            print(f"Appended {train_uncertain_filename} to {output_path}")

        # Append uncertain val data to each val_cv file
        for fname in val_cv_files:
            full_path = os.path.join(source_dir, fname)
            structures = read(full_path, index=":")
            combined = structures + val_unc
            output_path = os.path.join(current_dataset_dir, fname)
            write(output_path, combined)
            print(f"Appended {val_uncertain_filename} to {output_path}")



    def create_cv_splits(self, input_path, n_folds=5, seed=42):
        """
        Splits the input structure dataset into K non-overlapping train/validation sets using K-fold cross-validation.
        Ensures that each structure appears in a validation set exactly once across all folds.

        The resulting files are saved to: data/structures_labeled/initial/
        as train_cv_1.xyz, val_cv_1.xyz, ..., train_cv_K.xyz, val_cv_K.xyz.

        Args:
            input_path (str): Path to the input .xyz file containing all labeled structures 
                            (e.g., 'data/structures_labeled/trimers_start_train_val_pool.xyz').
            n_folds (int): Number of cross-validation folds to create.
            seed (int): Random seed for reproducibility of shuffle.
        """
        output_dir = os.path.join("data", "structures_labeled", "initial")
        os.makedirs(output_dir, exist_ok=True)

        all_structures = read(input_path, ":")
        print(f"✅ Read {len(all_structures)} structures from {input_path}")

        indices = np.arange(len(all_structures))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for i, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
            train_structures = [all_structures[j] for j in train_idx]
            val_structures = [all_structures[j] for j in val_idx]

            write(os.path.join(output_dir, f"train_cv_{i}.xyz"), train_structures)
            write(os.path.join(output_dir, f"val_cv_{i}.xyz"), val_structures)

            print(f"CV Fold {i}: {len(train_structures)} train, {len(val_structures)} val")


    def ensemble_generation_prediction_and_uncertainty(self, complexity_runNr, predict_structures, output_name=None, device="cuda"):

        """
        Loads an ensemble of trained MACE models (Stage Two) and performs prediction of energies and forces 
        for a given set of atomic structures. Computes mean and standard deviation across the ensemble to quantify uncertainty.

        Saves the resulting predictions and uncertainties as:
            - data/info/ensemble_energy_<complexity_runNr>.csv
            - data/info/ensemble_forces_<complexity_runNr>.csv

        Args:
            complexity_runNr (str): Subfolder name under 'models/' where ensemble models are stored.
                                    Each model subfolder must contain a 'stagetwo.model' file.
            predict_structures (str): Name of the .xyz file (without extension) located in data/structures_only/ 
                                    that contains the structures to predict on.
            device (str): Device to run the predictions on (e.g., "cuda" or "cpu").

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing ensemble energy and force predictions 
                                            with their uncertainties.
        """



        # 1. Find stagetwo.model files
        model_base_dir = os.path.join("models", str(complexity_runNr))
        model_paths = []

        for model_folder in os.listdir(model_base_dir):
            full_model_folder = os.path.join(model_base_dir, model_folder)
            if os.path.isdir(full_model_folder):
                for file in os.listdir(full_model_folder):
                    if file.endswith("stagetwo.model") and "run" not in file:
                        model_paths.append(os.path.join(full_model_folder, file))


        model_paths.sort()
        print(f"Found {len(model_paths)} models for ensemble.", flush=True)
        print("Model paths:", model_paths, flush=True)

        # 2. Load ensemble calculators
        calculators = [
            MACECalculator(model_paths=[path], device=device, default_dtype="float32")
            for path in model_paths
        ]

        # 3. Load structures
        structure_path = os.path.join("data", "structures_only", f"{predict_structures}.xyz")
        print(f"Predicting on structures from: {structure_path}", flush=True)

        new_structures = read(structure_path, index=":")
        new_structure_indices = list(range(len(new_structures)))
        #new_structure_types = [atoms.get_chemical_formula() for atoms in new_structures]


        # 5. Predict energies and forces
        ensemble_energies = []
        ensemble_forces = []

        for calc in calculators:
            energies = []
            forces = []
            for atoms in new_structures:
                atoms.calc = calc
                energies.append(atoms.get_potential_energy())
                forces.append(atoms.get_forces())
            ensemble_energies.append(energies)
            ensemble_forces.append(forces)

        ensemble_energies = np.array(ensemble_energies)
        ensemble_forces = np.array(ensemble_forces)
        print(f"finished predicting energies and forces for {len(new_structures)} structures", flush=True)



        # 6. Calculate mean and std
        mean_energies = np.mean(ensemble_energies, axis=0)
        std_energies = np.std(ensemble_energies, axis=0, ddof=1)

        mean_forces = np.mean(ensemble_forces, axis=0)
        std_forces = np.std(ensemble_forces, axis=0, ddof=1)

        df_energy = pd.DataFrame({
            "Index": new_structure_indices,
            "Predicted Energy": mean_energies,
            "Uncertainty Energy (Std)": std_energies,
        })
        for i in range(len(calculators)):
            df_energy[f"Model {i+1} Energy"] = ensemble_energies[i]

        df_forces = pd.DataFrame({"Index": new_structure_indices})
        for i in range(mean_forces.shape[1]):
            for j, axis in enumerate(["x", "y", "z"]):
                df_forces[f"Mean Force {i+1} {axis}"] = mean_forces[:, i, j]
                df_forces[f"Uncertainty Force {i+1} {axis}"] = std_forces[:, i, j]
                for m in range(len(calculators)):
                    df_forces[f"Model {m+1} Force {i+1} {axis}"] = ensemble_forces[m, :, i, j]


        # 8. Save to CSV
        info_dir = os.path.join("data", "info")
        os.makedirs(info_dir, exist_ok=True)


        if output_name is None:
            output_name = predict_structures

        energy_csv = os.path.join(info_dir, f"ensemble_energy_{output_name}.csv")
        forces_csv = os.path.join(info_dir, f"ensemble_forces_{output_name}.csv")

        df_energy.to_csv(energy_csv, index=False)
        df_forces.to_csv(forces_csv, index=False)
        print(f"[DEBUG] Attempting to save CSV to: {forces_csv}", flush=True)

        print(f"Saved energy DataFrame to: {energy_csv}", flush=True)
        print(f"Saved forces DataFrame to: {forces_csv}", flush=True)
        print("Finished predictions.", flush=True)

        return df_energy, df_forces




    #def finetune_from_existing_models(self, complexity_runNr, test_file):

    def finetune_from_existing_models(self, complexity_runNr, test_file, output_model_id=None):

        """
        Finetunes an ensemble of pretrained MACE models using new training data.
        
        The script automatically detects whether this is the first or a subsequent finetuning round,
        and selects the appropriate set of models from the previous stage. It uses the structure
        of the directory names and filenames to determine seeds and model paths.

        Finetuned models are saved in:
            models/finetuning_<N>_<complexity_runNr>/<complexity_runNr>_seed_<i>/

        Args:
            complexity_runNr (str): Unique ID or folder prefix for the run, e.g., "dimers_initial_test1".
            test_file (str): Filename of the .xyz file in data/structures_labeled/test used for testing.
        """
        
        print(f"[DEBUG] Finetuning from models in: models/{complexity_runNr}", flush=True)

        def count_lines(path):
            try:
                return sum(1 for _ in open(path))
            except:
                return -1



        def get_latest_finetune_index(models_base_dir, complexity_runNr):

            """
            Identifies the latest finetuning round by looking for folders matching 
            the pattern finetuning_<N>_<complexity_runNr>.
            
            Args:
                models_base_dir (str): Root directory of all model folders.
                complexity_runNr (str): Run identifier string.
                
            Returns:
                int: Highest finetuning round index found. Returns 0 if none found.
            """

            pattern = re.compile(f"finetuning_(\\d+)_({re.escape(complexity_runNr)})$")
            indices = []
            for folder in os.listdir(models_base_dir):
                match = pattern.match(folder)
                if match:
                    indices.append(int(match.group(1)))
            return max(indices) if indices else 0

        def find_stagetwo_model(path):
            for fname in os.listdir(path):
                if fname.endswith("stagetwo.model"):
                    return os.path.join(path, fname)
            raise FileNotFoundError(f"No stagetwo.model found in {path}")

        def make_default_args():

            """
            Builds and returns default MACE training arguments by extending sys.argv
            """

            if "--name" not in sys.argv:
                sys.argv.extend(["--name", "mace"])
            return build_default_arg_parser().parse_args()

        def extract_seeds_from_folder(path, prefix):

            """
            Extracts all seed numbers from folder names that follow the pattern <prefix>_seed_<number>.
            
            Args:
                path (str): Directory to search in.
                prefix (str): Common prefix of the model folders.
                
            Returns:
                List[int]: Sorted list of extracted seed integers.
            """

            seeds = []
            for folder in os.listdir(path):
                match = re.match(f"{re.escape(prefix)}_seed_(\\d+)", folder)
                if match:
                    seeds.append(int(match.group(1)))
            return sorted(seeds)

        if output_model_id is None:
            output_model_id = complexity_runNr

        data_root=os.path.join("data", "structures_labeled", "current_dataset")
        test_file = os.path.join("data", "structures_labeled","test", test_file)
        output_base = "models"

        latest_round = get_latest_finetune_index(output_base, complexity_runNr)
        next_round = latest_round + 1

        if latest_round == 0:
            foundation_base = os.path.join("models", complexity_runNr)
        else:
            foundation_base = os.path.join(output_base, f"finetuning_{latest_round}_{complexity_runNr}")


        if complexity_runNr.startswith("finetuning_"):
            model_prefix = "_".join(complexity_runNr.split("_")[2:])
        else:
            model_prefix = complexity_runNr

        # this worked for finetune mode
        #seeds = extract_seeds_from_folder(foundation_base, model_prefix)
        all_subdirs = os.listdir(foundation_base)
        candidate_prefixes = set()

        for subdir in all_subdirs:
            match = re.match(r"(.+)_seed_\d+", subdir)
            if match:
                candidate_prefixes.add(match.group(1))

        if not candidate_prefixes:
            raise RuntimeError(f"[ERROR] No model folders found with seed pattern in {foundation_base}")

        if len(candidate_prefixes) > 1:
            print(f"[WARNING] Multiple prefixes found: {candidate_prefixes} — using first one.")
        model_prefix = sorted(candidate_prefixes)[0]
        print(f"[DEBUG] Using detected model_prefix: {model_prefix}", flush=True)

        seeds = extract_seeds_from_folder(foundation_base, model_prefix)
        print(f"[DEBUG] Extracted seeds: {seeds}", flush=True)


        print(f"[DEBUG] model_prefix: {model_prefix}", flush=True)
        print(f"[DEBUG] foundation_base: {foundation_base}", flush=True)
        print(f"[DEBUG] Extracted seeds: {seeds}", flush=True)


        finetune_root = os.path.join(output_base, f"finetuning_{next_round}_{output_model_id}")
        os.makedirs(finetune_root, exist_ok=True)

        for seed in seeds:
            print(f"Finetuning seed {seed} from: {foundation_base}")
            args = make_default_args()

            foundation_model_dir = os.path.join(foundation_base, f"{model_prefix}_seed_{seed}")

            foundation_model = find_stagetwo_model(foundation_model_dir)

            assert os.path.exists(foundation_model_dir), f"[ERROR] Foundation model directory missing: {foundation_model_dir}"

            pt_train_file = os.path.join(foundation_model_dir, f"train_cv_{seed}.xyz")
            train_file = os.path.join(data_root, f"train_cv_{seed}.xyz")
            val_file   = os.path.join(data_root, f"val_cv_{seed}.xyz")

            args.name = f"fine_mixed_naive_bagging_seed_{seed}_bag_{seed}"
            args.foundation_model = foundation_model
            args.multiheads_finetuning = False
            args.pt_train_file = pt_train_file
            args.train_file = train_file
            args.valid_file = val_file
            args.test_file = test_file
            args.E0s = "{6: 3.5, 8: 4.6, 78: 5.85}"
            args.model = "MACE"
            args.num_channels = 128
            args.max_L = 2
            args.correlation = 3
            args.r_max = 8.0
            args.batch_size = 4
            args.max_num_epochs = 1000
            args.valid_batch_size=1
            args.eval_interval = 1
            args.patience = 100
            args.swa = True
            args.start_swa = 200
            args.ema = True
            args.ema_decay = 0.99
            args.amsgrad = True
            args.device = "cuda"
            args.seed = seed
            args.energy_key = "energy"
            args.forces_key = "forces"
            args.default_dtype="float32" 

            out_dir = os.path.join(finetune_root, f"{complexity_runNr}_seed_{seed}")
            os.makedirs(out_dir, exist_ok=True)
            args.model_dir = out_dir
            args.log_dir = out_dir
            args.checkpoints_dir = out_dir
            args.results_dir = out_dir

            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            # Optionally reset the root logger to default
            logging.basicConfig(level=logging.INFO)
            
            print(f"[CHECK] train_file exists: {os.path.exists(train_file)} ({count_lines(train_file)} lines)")
            print(f"[CHECK] val_file exists:   {os.path.exists(val_file)} ({count_lines(val_file)} lines)")

            print(f"[INFO] Launching training for seed {seed} with:", flush=True)
            print(f"       foundation_model: {foundation_model}", flush=True)
            print(f"       pt_train_file: {pt_train_file}", flush=True)
            print(f"       train_file: {train_file}", flush=True)
            print(f"       val_file: {val_file}", flush=True)
            print(f"       test_file: {test_file}", flush=True)
            print(f"       model_dir: {out_dir}", flush=True)


            # Ensure we are in a valid working directory before calling run()
            try:
                os.chdir(os.path.dirname(os.path.abspath(__file__)))
            except Exception as e:
                print(f"[WARNING] Failed to reset working directory before seed {seed}: {e}", flush=True)

            try:
                run(args)
                print(f"[INFO] Finished training for seed {seed}", flush=True)
            except Exception as e:
                print(f"[ERROR] Training failed for seed {seed}: {e}", flush=True)
                continue




            print(f"[INFO] Finished training for seed {seed}", flush=True)
            stagetwo_model_path = os.path.join(out_dir, f"fine_mixed_naive_bagging_seed_{seed}_bag_{seed}_stagetwo.model")
            print(f"[CHECK] stagetwo.model exists after training: {os.path.exists(stagetwo_model_path)}", flush=True)



    def extract_most_uncertain_structures(self, csv_file, xyz_file, unc_threshold=None, top_k=None):

        """
        Identifies and extracts the most uncertain structures from a dataset based on ensemble force uncertainties.

        Structures with a mean per-structure force uncertainty above the given threshold are saved separately.
        The remaining (less uncertain) structures are saved as a new test set.

        Outputs:
            - Most uncertain and remaining structures saved as .xyz and .csv
            - A log entry written to: data/info/number_uncertainty_log.csv

        Args:
            csv_file (str): Filename of the input CSV containing force predictions and uncertainties (in data/info/).
            xyz_file (str): Filename of the corresponding .xyz file with atomic structures (in data/structures_only/).
            unc_threshold (float): Threshold above which a structure is considered uncertain.

        Returns:
            int: Number of structures classified as most uncertain.
        """
        assert os.path.exists(os.path.join("data", "info", csv_file)), f"[ERROR] CSV file not found: {csv_file}"
        print(f"[DEBUG] looking for num_uncertain in {csv_file}", flush=True)
        
        # Load CSV and XYZ structures
        csv_path = os.path.join("data", "info", csv_file)

        df = pd.read_csv(csv_path)
        xyz_path = os.path.join("data", "structures_only", xyz_file)
        atoms_list = read(xyz_path, index=':')

        # Extract uncertainty columns
        unc_cols = [col for col in df.columns if "Uncertainty Force" in col]
        
        # Compute mean force uncertainty per structure
        mean_uncs = df[unc_cols].mean(axis=1)

        # get all indices with mean unc > threshold

        if top_k is not None:
            uncertain_indices = mean_uncs.nlargest(top_k).index
            print(f"Selected top {top_k} most uncertain structures.", flush=True)
        elif unc_threshold is not None:
            uncertain_indices = mean_uncs[mean_uncs > unc_threshold].index
            print(f"Found {len(uncertain_indices)} uncertain structures with mean uncertainty > {unc_threshold}")
        else:
            raise ValueError("Either 'unc_threshold' or 'top_k' must be provided.")

        print(f"[DEBUG] Selected top {top_k} structures with uncertainty.", flush=True)


        print(f"Found {len(uncertain_indices)} uncertain structures with mean uncertainty > {unc_threshold}", flush=True)

        # Get top-N uncertain indices
        rest_indices = df.index.difference(uncertain_indices)

        # Split CSVs
        df_selected = df.loc[uncertain_indices]
        df_remaining = df.loc[rest_indices]

        # Split structures accordingly
        selected_atoms = [atoms_list[i] for i in uncertain_indices]
        remaining_atoms = [atoms_list[i] for i in rest_indices]

        csv_selected_path = csv_path.replace(".csv", "_most_uncertain.csv")
        csv_remaining_path = csv_path.replace(".csv", "_certain.csv")
        df_selected.to_csv(csv_selected_path, index=False)
        df_remaining.to_csv(csv_remaining_path, index=False)

        # Write both sets
        unc_xyz_path = os.path.join("data", "structures_only", xyz_file.replace(".xyz", "_most_uncertain.xyz"))
        write(unc_xyz_path, selected_atoms)

        rest_xyz_path = os.path.join("data", "structures_only", xyz_file.replace(".xyz", "_certain.xyz"))
        write(rest_xyz_path.replace(".xyz", "_test.xyz"), remaining_atoms)

        log_path = os.path.join("data", "info", "number_uncertainty_log.csv")

        with open(xyz_path, "r") as f:
            first_line_number = f.readline().strip()
        # Extract logging info
        log_entry = {
            "threshold": unc_threshold,
            "xyz_file": xyz_file,
            "complexity": first_line_number,
            "num_uncertain": len(uncertain_indices)
        }

        # Append or create log
        log_exists = os.path.isfile(log_path)
        log_df = pd.DataFrame([log_entry])

        if log_exists:
            log_df.to_csv(log_path, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_path, index=False)

        print(f"Log entry added to: {log_path}")



        print(f"{len(uncertain_indices)} uncertain structures out {len(mean_uncs)} saved here", flush=True)
        print(f"  XYZ:  {unc_xyz_path}", flush=True)
        print(f"  CSV:  {csv_selected_path}")
        print(f"Saved remaining structures to:", flush=True)
        print(f"  XYZ:  {rest_xyz_path.replace('.xyz', '_test.xyz')}", flush=True)
        print(f"  CSV:  {csv_remaining_path}")


        # Append to log
        with open(self.structure_log_path, "a") as f:
            f.write(f"{self.current_complexity},{len(uncertain_indices)}\n")

        if top_k is not None:
            return 0  # Always trigger complexity advance after one round

        else:
            # Return the number of uncertain structures
            return len(uncertain_indices)




    def split_test_holdout(self, filename, test_ratio, seed=42):
        """
        Splits a labeled .xyz file into a holdout test set and a remaining train/validation pool.

        The resulting files are saved as:
            - data/structures_labeled/test/<original>_test_set.xyz
            - data/structures_labeled/<original>_train_val_pool.xyz

        Args:
            filename (str): Filename of the input .xyz file located in data/structures_labeled/.
            test_ratio (float): Fraction of the data to allocate to the test set.
            seed (int): Random seed for reproducibility of the split.


            PSA - careful with test ratio
        """

        input_path = os.path.join("data", "structures_labeled", filename)

        # Output directories
        base_dir = os.path.join("data", "structures_labeled")
        test_output_dir = os.path.join(base_dir, "test")
        os.makedirs(test_output_dir, exist_ok=True)

        # Read structures
        all_structures = read(input_path, ":")

        # Split
        train_val, test = train_test_split(all_structures, test_size=test_ratio, random_state=seed)

        # File naming
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        test_filename = f"{input_name}_test_set.xyz"
        pool_filename = f"{input_name}_train_val_pool.xyz"

        # Write files
        write(os.path.join(test_output_dir, test_filename), test)
        write(os.path.join(base_dir, pool_filename), train_val)

        print(f"Split complete:")
        print(f"- Total:     {len(all_structures)}")
        print(f"- Test set:       {len(test)} -> {os.path.join(test_output_dir, test_filename)}")
        print(f"- Train/Val pool: {len(train_val)} -> {os.path.join(base_dir, pool_filename)}")




    def summarize_latest_finetuning_internal(self, complexity_runNr):

        """
        Extracts and summarizes test performance metrics from the latest finetuning round for a given run.

        The function:
            - Identifies the most recent folder matching 'finetuning_<N>_<complexity_runNr>' in the models directory.
            - Locates all `.log` files (excluding debug logs) from each seed-specific subfolder.
            - Parses test set RMSE energy, force, and relative force error from "Default_default" rows.
            - Separates out Stage Two results based on log markers.
            - Saves two summary CSVs: one for all stages and one filtered to only Stage Two.

        Summary files are saved in: data/info/summaries/
            - <complexity_runNr>_finetuning_<N>_<complexity_runNr>_allstages.csv
            - <complexity_runNr>_finetuning_<N>_<complexity_runNr>_stage_two.csv

        Args:
            complexity_runNr (str): Run identifier, e.g., "dimers_initial_test1".
        """



        models_dir = "models"
        summary_dir = os.path.join("data", "info", "summaries")
        os.makedirs(summary_dir, exist_ok=True)

        # Step 1: Find the latest folder
        pattern = re.compile(f"finetuning_(\\d+)_({re.escape(complexity_runNr)})$")
        candidates = [(int(m.group(1)), fname) for fname in os.listdir(models_dir)
                    if (m := pattern.match(fname))]

        if candidates:
            _, latest_folder = max(candidates)
            latest_path = os.path.join(models_dir, latest_folder)
            print(f"✅ Found latest finetuning folder: {latest_path}")
        else:
            latest_folder = complexity_runNr
            latest_path = os.path.join(models_dir, latest_folder)
            if os.path.exists(latest_path):
                print(f"[INFO] No finetuning folders found — using initial training results from: {latest_path}")
            else:
                raise FileNotFoundError(
                    f"No finetuning or initial training folder found for '{complexity_runNr}' in '{models_dir}'")

        test_row_pattern = re.compile(r"^\|\s*Default_default\s*\|")
        all_results = []
        stage_two_results = []

        for seed_dir in os.listdir(latest_path):
            seed_dir_path = os.path.join(latest_path, seed_dir)
            if not os.path.isdir(seed_dir_path):
                continue

            seed_match = re.search(r"seed_(\d+)", seed_dir)
            seed = int(seed_match.group(1)) if seed_match else "Unknown"

            log_file = None
            for fname in os.listdir(seed_dir_path):
                if fname.endswith(".log") and "debug" not in fname:
                    log_file = os.path.join(seed_dir_path, fname)
                    break

            if not log_file:
                print(f"⚠️  No valid .log file found in {seed_dir_path}")
                continue

            with open(log_file, "r") as f:
                lines = f.readlines()

            print(f"🔍 Processing log file: {log_file}")

            seed_from_log = seed
            test_tables = []
            temp_test_table = []
            capture = False
            stage_two_active = False
            test_table_count = 0

            for line in lines:
                if "INFO: Loaded Stage two model" in line:
                    stage_two_active = True
                    print("✅ Stage Two detected")

                if "Error-table on TEST:" in line:
                    if temp_test_table:
                        print(f"📥 Storing test table {test_table_count} for seed {seed_from_log} (Stage Two: {stage_two_active})")
                        test_tables.append((temp_test_table, seed_from_log, stage_two_active, test_table_count))
                    temp_test_table = []
                    capture = True
                    test_table_count += 1
                    continue

                if capture:
                    if line.startswith("|") and test_row_pattern.match(line):
                        temp_test_table.append(line.strip())
                    elif "INFO" in line or "Saving model" in line or "Evaluating" in line:
                        capture = False

            if temp_test_table:
                test_tables.append((temp_test_table, seed_from_log, stage_two_active, test_table_count))

            for test_table, seed_value, is_stage_two, table_idx in test_tables:
                if seed_value is None:
                    seed_value = "Unknown"
                for row in test_table:
                    cols = row.split("|")[1:-1]
                    cols = [c.strip() for c in cols]
                    if len(cols) == 4:
                        entry_data = [os.path.basename(log_file), seed_value] + cols
                        all_results.append(entry_data)
                        # Only store every second table (stage two assumed)
                        if table_idx % 2 != 1:
                            stage_two_results.append(entry_data)

        # Save DataFrames
        df_all = pd.DataFrame(all_results, columns=["Filename", "Seed", "Config Type",
                                                    "RMSE E (meV/atom)", "RMSE F (meV/Å)",
                                                    "Relative F RMSE (%)"])
        df_stage_two = pd.DataFrame(stage_two_results, columns=["Filename", "Seed", "Config Type",
                                                                "RMSE E (meV/atom)", "RMSE F (meV/Å)",
                                                                "Relative F RMSE (%)"])

        for df in [df_all, df_stage_two]:
            df["Seed"] = pd.to_numeric(df["Seed"], errors="coerce")
            df["RMSE E (meV/atom)"] = pd.to_numeric(df["RMSE E (meV/atom)"], errors="coerce")
            df["RMSE F (meV/Å)"] = pd.to_numeric(df["RMSE F (meV/Å)"], errors="coerce")
            df["Relative F RMSE (%)"] = pd.to_numeric(df["Relative F RMSE (%)"], errors="coerce")

        base_name = f"{complexity_runNr}_{latest_folder}"
        summary_all_path = os.path.join(summary_dir, f"{base_name}_allstages.csv")
        summary_stage_two_path = os.path.join(summary_dir, f"{base_name}_stage_two.csv")

        df_all.to_csv(summary_all_path, index=False)
        df_stage_two.to_csv(summary_stage_two_path, index=False)

        print(f"\n✅ Summary saved to: {summary_all_path}")
        print(f"✅ Summary saved to: {summary_stage_two_path}")
        print(f"✅ Total stage_two models: {len(df_stage_two)} rows\n")


    def train(self, complexity_runNr, seeds):

        """
        Trains a MACE model from scratch for each provided seed using predefined training, validation, and test sets.

        For each seed:
            - Loads the corresponding train/val split from data/structures_labeled/initial/
            - Uses a fixed test set from data/structures_labeled/test/
            - Sets all training hyperparameters explicitly (e.g., model architecture, optimizer settings)
            - Saves model checkpoints, logs, and results into models/<complexity_runNr>/<complexity_runNr>_seed_<seed>/

        Args:
            complexity_runNr (str): Identifier for the training round (e.g., "dimers_initial_test1").
            seeds (List[int]): List of random seeds to train independent models (used for ensembling or cross-validation).
        """

        structures_path = os.path.join("data", "structures_labeled", "initial")
        test_path = os.path.join("data", "structures_labeled", "test")

        def make_default_args():   # make name an argument, parser needs name
            """
            Builds the default argument parser for MACE training.

            Automatically appends a default `--name` argument to sys.argv if not already present
            (required by the MACE CLI parser). Prints the parsed arguments for debugging.

            Returns:
                argparse.Namespace: Parsed command-line arguments ready for training.
            """

            # if "--name" not in sys.argv:
            #     sys.argv.extend(["--name", "mace"])
            # args = build_default_arg_parser().parse_args()
            # print(args)   # for checking if it messes up args when the train function gets called several times

            # return args
            default_args = ["--name", "mace"]
            args = build_default_arg_parser().parse_args(default_args)
            return args


        for seed in seeds:
            train_file = os.path.join(structures_path, f"train_cv_{seed}.xyz")
            val_file   = os.path.join(structures_path, f"val_cv_{seed}.xyz")
            test_file  = os.path.join(test_path, "dimers_start_test_set.xyz")


        
            args = make_default_args()

            # manually override the parameters you care about
            args.name = f"mace_initial_{complexity_runNr}_seed_{seed}_fold_{seed}"
            args.foundation_model="/home/cat/s233070/MD_baseline_models/baseline_model_MP_foundation/mace_foundationMP_baseline_seed_${seed}_stagetwo.model"
            args.multiheads_finetuning=False
            args.train_file = train_file
            args.valid_file = val_file
            args.test_file = test_file
            args.E0s="{6: 3.5, 8: 4.6, 78: 5.85}"
            args.model = "MACE"
            args.num_channels = 128
            args.max_L = 2
            args.correlation = 3
            args.r_max = 8.0
            args.batch_size = 4
            args.max_num_epochs = 1000
            args.valid_batch_size=1
            args.eval_interval = 1
            args.patience = 100
            args.swa = True
            args.start_swa = 200
            args.ema = True
            args.ema_decay = 0.99
            args.amsgrad = True
            args.device = "cuda"
            args.seed = seed
            args.energy_key = "energy"
            args.forces_key = "forces"
            args.default_dtype="float32" 

            base_dir = os.path.join("models", str(complexity_runNr), f"{complexity_runNr}_seed_{seed}")
            os.makedirs(base_dir, exist_ok=True)
            args.model_dir = base_dir
            args.log_dir = base_dir
            args.checkpoints_dir = base_dir
            args.results_dir = base_dir
            print(args)

            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            # Optionally reset the root logger to default
            logging.basicConfig(level=logging.INFO)

            run(args)




    def split_train_val(self, filename, val_ratio, seed=42):
        """
        Splits a labeled .xyz file into training and validation sets using a fixed ratio.

        The resulting files are saved in the same directory with "_train" and "_val" suffixes:
            - data/structures_labeled/<filename>_train.xyz
            - data/structures_labeled/<filename>_val.xyz

        Args:
            filename (str): Name of the .xyz file in data/structures_labeled to split.
            val_ratio (float): Fraction of the dataset to reserve for validation.
            seed (int): Random seed to ensure reproducibility of the split.
        """

        input_path = os.path.join("data", "structures_labeled", filename)

        # Output directories
        base_dir = os.path.join("data", "structures_labeled")

        # Read structures
        all_structures = read(input_path, ":")

        # Split
        train, val = train_test_split(all_structures, test_size=val_ratio, random_state=seed)

        # File naming
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        train_filename = f"{input_name}_train.xyz"
        val_filename = f"{input_name}_val.xyz"

        # Write files
        write(os.path.join(base_dir, val_filename), val)
        write(os.path.join(base_dir, train_filename), train)

        print(f"✅ Split complete:")
        print(f"- Validation set: {len(val)} -> {os.path.join(base_dir, val_filename)}")
        print(f"- Train ser: {len(train)} -> {os.path.join(base_dir, train_filename)}")


    def get_latest_model_dir(self, dataset_name_prefix):
        """
        Returns the latest finetuned or baseline model folder for any available
        complexity ≤ current complexity.

        Args:
            dataset_name_prefix (str): e.g. 'example_3'

        Returns:
            str or None: Best model folder name
        """
        model_base = "models"
        best_complexity = -1
        best_finetune_num = -1
        best_dir = None

        for d in os.listdir(model_base):
            if not os.path.isdir(os.path.join(model_base, d)):
                continue

            # Match: finetuning_<num>_example_3_complexity_<comp>
            match = re.match(rf"^finetuning_(\d+)_({re.escape(dataset_name_prefix)}_complexity_(\d+))$", d)
            if match:
                finetune_num = int(match.group(1))
                full_dataset_name = match.group(2)
                complexity = int(match.group(3))

                if complexity <= self.current_complexity:
                    if (complexity > best_complexity) or (
                        complexity == best_complexity and finetune_num > best_finetune_num
                    ):
                        best_complexity = complexity
                        best_finetune_num = finetune_num
                        best_dir = d

        if best_dir:
            print(f"[DEBUG] Found finetuned model: {best_dir}", flush=True)
            return best_dir

        # Fallback: try baseline folders
        for d in os.listdir(model_base):
            if not os.path.isdir(os.path.join(model_base, d)):
                continue
            match = re.match(rf"^({re.escape(dataset_name_prefix)}_complexity_(\d+))$", d)
            if match:
                comp = int(match.group(2))
                if comp <= self.current_complexity and comp > best_complexity:
                    best_complexity = comp
                    best_dir = d

        if best_dir:
            print(f"[DEBUG] Fallback to baseline model: {best_dir}", flush=True)
            return best_dir

        print(f"[DEBUG] No model found for prefix: {dataset_name_prefix}", flush=True)
        return None
