import os
import numpy as np
from qdmt.analysis import tools
from qdmt.analysis import conserved_quantities as cq 
from src.qdmt.uniform_mps import UniformMps
from qdmt.analysis import magnetization as mag

from src.qdmt.evolve import *




def filepath_gen(dt: float, steps: int, tolerance: float, iterations: int, D: int, cut_off: float):
    # path to the folder containing THIS script
    here = os.path.dirname(__file__)
    save_folder = os.path.join(here, "..", "results")  # one level up, into results/
    os.makedirs(save_folder, exist_ok=True)  # create it if it doesn't exist

    file_name=f"benchmark_dt={dt}_steps={steps}_tol={tolerance}_it={iterations}_D={D}_cut={cut_off}"

    file_path = os.path.join(save_folder, file_name)
    return  file_path




def load_data(dt, steps, tolerance, iterations, D, cut_off):
     # --- Configuration ---
    # filename = "benchmark_dt=0.01_steps=100_tol=1e-08_it=500_D=4_cut=20.npz"

    
        
    # Compute absolute path to qdmt-main/results/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # go up one level
    results_dir = os.path.join(project_root, "results")

    filename = f"benchmark_dt={dt}_steps={steps}_tol={tolerance}_it={iterations}_D={D}_cut={cut_off}.npz"
    filepath = os.path.join(results_dir, filename)

    # --- Load data ---
    data = np.load(filepath)
    return data




def count_initial_nonzero(arr, cutoff=1e-16):
    """
    Count how many leading entries in `arr` have |value| > cutoff.
    Stops at the first element below the cutoff.
    """
    arr = np.asarray(arr)
    for i, val in enumerate(arr):
        if not np.isfinite(val) or abs(val) <= cutoff or abs(val) > 1 or val < 0:
            print(val)
            print(i)
            return i
            
    return len(arr)




def trim_data(data, l):
    """
    Trim all arrays in an np.load(...) result (npz file) to length l.
    Returns a new dict of trimmed arrays.
    """
    trimmed = {}
    for key in data.files:
        arr = data[key]
        trimmed[key] = arr[:l]
    return trimmed


def save_trimmed_data(data, dt, steps, tolerance, iterations, D, cut_off):
    # Base (original) path
    original_path = filepath_gen(dt, steps, tolerance, iterations, D, cut_off)

    # Folder that will hold trimmed datasets
    trimmed_folder = os.path.join(
        os.path.dirname(original_path),  # the results/ folder
        "datasets_trimmed"
    )
    os.makedirs(trimmed_folder, exist_ok=True)

    # Use same filename but with _trimmed suffix
    base_name = os.path.basename(original_path)
    trimmed_name = base_name + "_trimmed.npz"

    save_path = os.path.join(trimmed_folder, trimmed_name)

    # --- AUTO-SKIP if already present ---
    if os.path.exists(save_path) and False:
        print(f"Trimmed dataset already exists: {save_path}")
        return

    # --- Save compressed in same format as original ---
    np.savez_compressed(
        save_path,
        time=data["time"],
        state=data["state"],
        gradient_norm=data["gradient_norm"],
        cost=data["cost"],
        duration=data["duration"]
    )

    print(f"Trimmed dataset saved to: {save_path}")





def prep_data(dt, steps, tolerance, iterations, D, cut_off, timerange = None):
    data = load_data(dt, steps, tolerance, iterations, D, cut_off)   
    if timerange is None:
        valid_len = count_initial_nonzero(data["cost"])
        timerange = valid_len*dt
    else:
        valid_len = int(timerange // dt)      
    print(valid_len)    
    data = trim_data(data, valid_len)
    # --- save trimmed version ---
    save_trimmed_data(
        data,
        dt, steps, tolerance, iterations, D, cut_off
    )

    return data    



import os
import numpy as np

def load_trimmed_data(dt, steps, tolerance, iterations, D, cut_off):
    # Recreate the original untrimmed base path
    original_path = filepath_gen(dt, steps, tolerance, iterations, D, cut_off)

    # Path to the trimmed dataset folder
    trimmed_folder = os.path.join(
        os.path.dirname(original_path),   # results/
        "datasets_trimmed"
    )

    # Expected filename with suffix
    base_name = os.path.basename(original_path)
    trimmed_name = base_name + "_trimmed.npz"

    load_path = os.path.join(trimmed_folder, trimmed_name)

    # Check existence
    if not os.path.isfile(load_path):
        print(f"Trimmed dataset not found: {load_path}")
        return None

    # Load and return dict-like object
    data = np.load(load_path)

    print(f"Loaded trimmed dataset: {load_path}")

    return {
        "time": data["time"],
        "state": data["state"],
        "gradient_norm": data["gradient_norm"],
        "cost": data["cost"],
        "duration": data["duration"]
    }


def unfold_data(
    data_trimmed,
    dt,
    steps,
    tolerance,
    iterations,
    D,
    cut_off,overwrite = False, L=4,g=1.05, h=-0.5, J=-1
):
    """
    Take trimmed data (dict) and compute derived quantities:
    norm, energy, renyi_entropy. Save as a new NPZ in
    results/datasets_unfolded/, and return the extended dict.
    """

    state = data_trimmed["state"]
    cost = data_trimmed["cost"]

    # --- derived quantities ---
    # hard code delta_t here just for the hamiltonian generation, it doesnt do anything. we use just H not exp(iHt)
    model_H = TransverseFieldIsing(g=g, delta_t=0.1, h=h, J=J)

    print("compute norm")
    norm = np.array(list(map(lambda x: cq.compute_norm(UniformMps(x)), state)))
    print("compute energy")
    energy = np.array(list(map(lambda x: cq.compute_energy(UniformMps(x), model_H), state)))
    print("compute renyi")
    renyi_entropy = np.array(list(map(lambda x: tools.compute_second_Reyni(UniformMps(x), L), state)))
    print("compute log cost")
    log_cost = np.array(list(map(lambda x: np.log(x), cost)))
    # print(log_cost)
    print("compute magnetization")
    t_magnetization = np.array(list(map(lambda x: mag.transverse_magnetization(UniformMps(x)), state)))
   

    # --- build extended data dict ---
    data_unfolded = {
        "time": data_trimmed["time"],
        "state": data_trimmed["state"],
        "gradient_norm": data_trimmed["gradient_norm"],
        "cost": data_trimmed["cost"],
        "duration": data_trimmed["duration"],
        "norm": norm,
        "energy": energy,
        "renyi_entropy": renyi_entropy,
        "log_cost": log_cost,
        "t_magnetization": t_magnetization
    }

    # --- save to results/datasets_unfolded/ ---

    # Use the same base filepath pattern as before
    original_path = filepath_gen(dt, steps, tolerance, iterations, D, cut_off)

    # Folder parallel to datasets_trimmed
    unfolded_folder = os.path.join(
        os.path.dirname(original_path),  # results/
        "datasets_unfolded"
    )
    os.makedirs(unfolded_folder, exist_ok=True)

    base_name = os.path.basename(original_path)
    # derived from trimmed data → keep "_trimmed" in lineage
    unfolded_name = base_name + "_trimmed_unfolded.npz"
    save_path = os.path.join(unfolded_folder, unfolded_name)

    # Auto-skip if already exists
    if os.path.exists(save_path) and not overwrite:
        print(f"Unfolded dataset already exists: {save_path}")
        return data_unfolded

    np.savez_compressed(
        save_path,
        time=data_unfolded["time"],
        state=data_unfolded["state"],
        gradient_norm=data_unfolded["gradient_norm"],
        cost=data_unfolded["cost"],
        duration=data_unfolded["duration"],
        norm=data_unfolded["norm"],
        energy=data_unfolded["energy"],
        renyi_entropy=data_unfolded["renyi_entropy"],
        log_cost=data_unfolded["log_cost"],
        t_magnetization=data_unfolded["t_magnetization"]
        
    )

    print(f"Unfolded dataset saved to: {save_path}")

    return data_unfolded




def load_unfolded_data(dt, steps, tolerance, iterations, D, cut_off,L=4,g=1.05, h=-0.5, J=-1):
    """
    Load unfolded data saved in results/datasets_unfolded/.
    Returns a dict or None if file not found.
    """

    # Base untrimmed filepath
    original_path = filepath_gen(dt, steps, tolerance, iterations, D, cut_off)

    # Folder parallel to datasets_trimmed
    unfolded_folder = os.path.join(
        os.path.dirname(original_path),   # results/
        "datasets_unfolded"
    )

    base_name = os.path.basename(original_path)
    unfolded_name = base_name + "_trimmed_unfolded.npz"
    load_path = os.path.join(unfolded_folder, unfolded_name)

    # Check existence
    if not os.path.isfile(load_path):
        print(f"Unfolded dataset not found: {load_path}")
        return None

    data = np.load(load_path)

    print(f"Loaded unfolded dataset: {load_path}")

    return {
        "time": data["time"],
        "state": data["state"],
        "gradient_norm": data["gradient_norm"],
        "cost": data["cost"],
        "duration": data["duration"],
        "norm": data["norm"],
        "energy": data["energy"],
        "renyi_entropy": data["renyi_entropy"],
        "log_cost": data["log_cost"],
    }



def generate_trimmed_and_unfolded(
    dt,
    steps,
    tolerance,
    iterations,
    D,
    cut_off,overwrite = False,
    timerange=None,L=4,g=1.05, h=-0.5, J=-1
):
    """
    Full processing pipeline:
    1. Load + trim data  (prep_data)
    2. Save trimmed dataset
    3. Compute derived quantities (norm, energy, renyi)
    4. Save unfolded dataset
    Returns (trimmed_data, unfolded_data).
    """

    print("\n--- Generating trimmed dataset ---")
    trimmed = prep_data(
        dt,
        steps,
        tolerance,
        iterations,
        D,
        cut_off,
        timerange=timerange
    )

    print("\n--- Generating unfolded dataset ---")
    unfolded = unfold_data(
        trimmed,
        dt,
        steps,
        tolerance,
        iterations,
        D,
        cut_off,overwrite
    )

    print("\n--- Done ---")
    return trimmed, unfolded



def spit_out_state(dt, steps, tolerance, iterations, D, cut_off, timerange = None):
    # RETURN THE STATE ONE STEP BEFORE TIMERANGE
    data = prep_data(dt, steps, tolerance, iterations, D, cut_off, timerange)
    state=data["state"]
    cost=data["cost"]
    print(f"cost here={cost[-1]}\nprevious cost was={cost[-2]}")
    return UniformMps(state[-2])