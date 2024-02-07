import json
from pathlib import Path
import pandas as pd


def compose_angiometric_csv() -> None:
    """
    Compose the simulation parameters of each simulation with the corresponding angiometrics
    :return: Nothing
    """
    # set identifier for the CAM used in the experiment
    egg_code = "w1_d0_CTRL_H1"

    # load csv containing the parameters value for each simulation
    angiometrics_sim_folder = Path("saved_sim/saved_sim_angiometrics/")
    convergence_csv = angiometrics_sim_folder / Path("convergence_2days.csv")
    convergence_df = pd.read_csv(convergence_csv)

    # get simulation ids
    simulation_ids = convergence_df["sim_i"]

    # init output list
    output_list = []

    # iterate on simulations
    for sim_id in simulation_ids:
        # get angiometrics json
        angiometrics_json = angiometrics_sim_folder / Path(f"{egg_code}_2days_{str(sim_id).zfill(3)}/sim_info/"
                                                           f"angiometrics.json")
        # load angiometrics dict
        with open(angiometrics_json, "r") as infile:
            angiometrics_dict = json.load(infile)


        # we generate a dict containing:
        # - the simulation id (sim_id)
        # - each angiometric (e.g. vf, bpa) at each time
        angiometrics_names = ["vf", "bpa", "bpl", "median_radius"]

        sim_dict = {
            "sim_id": str(sim_id).zfill(3)
        }

        for angiometrics_at_time in angiometrics_dict:
            time = angiometrics_at_time["time"]
            for angiometric in angiometrics_names:
                sim_dict[f"{angiometric} (t={time})"] = angiometrics_at_time[angiometric]

        # append sim_dict to the output
        output_list.append(sim_dict)

    # concat angiometrics to the parameters dataframe
    output_df = pd.concat([convergence_df, pd.DataFrame(output_list)], axis=1)

    # save
    output_df.to_csv(f"out_postprocessing/angiometrics_{egg_code}.csv", index=False)

