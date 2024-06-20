import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# MACROS
OUT_POSTPROCESSING_FOLDER = Path("out_postprocessing")


def compose_angiometric_csv() -> None:
    """
    Compose the simulation parameters of each simulation with the corresponding angiometrics
    :return: Nothing
    """
    # set identifier for the CAM used in the experiment
    egg_code = "w1_d0_CTRL_H1"

    # load csv containing the parameters value for each simulation
    # angiometrics_sim_folder = Path("github/cam_mocafe/")
    # convergence_csv = angiometrics_sim_folder / Path("convergence_2days.csv")
    convergence_csv = Path("convergence_2days.csv")
    print(convergence_csv.resolve())
    convergence_df = pd.read_csv(convergence_csv)

    # get simulation ids
    simulation_ids = convergence_df["sim_i"]

    # init output list
    output_list = []

    # iterate on simulations
    for sim_id in simulation_ids:
        # get angiometrics json
        # angiometrics_json = angiometrics_sim_folder / Path(f"{egg_code}_2days_{str(sim_id).zfill(3)}/sim_info/"
        #                                                    f"angiometrics.json")
        angiometrics_json = Path(f"saved_sim/no_incremental_sim/{egg_code}_2days_{str(sim_id).zfill(3)}/sim_info/"
                                                            f"angiometrics.json")
        # load angiometrics dict
        with open(angiometrics_json, "r") as infile:
            angiometrics_dict = json.load(infile)


        # we generate a dict containing:
        # - the simulation id (sim_id)
        # - each angiometric (e.g. vf, bpa) at each time
        # - the percentage variation of each angiometric at each time
        angiometrics_names = ["vf", "bpa", "bpl", "median_radius"]

        sim_dict = {
            "sim_id": str(sim_id).zfill(3)
        }

        for angiometrics_at_time in angiometrics_dict:
            # get time
            time = int(angiometrics_at_time["time"])
            # add value for angiometric
            for angiometric in angiometrics_names:
                sim_dict[f"{angiometric} (t={time})"] = angiometrics_at_time[angiometric]

                if time > 0:
                    # add percentage variation
                    percentage_variation = angiometrics_at_time[angiometric] / sim_dict[f"{angiometric} (t=0)"]
                    sim_dict[f"%{angiometric} (t={time})"] = percentage_variation

        # append sim_dict to the output
        output_list.append(sim_dict)

    # concat angiometrics to the parameters dataframe
    output_df = pd.concat([convergence_df, pd.DataFrame(output_list)], axis=1)

    # save
    output_df.to_csv(f"out_postprocessing/no_incremental_ang/angiometrics_{egg_code}.csv", index=False)


def visualize_parameter_influence_on_angioparameters():
    """
    Generate plots showing, for each simulation parameters (V_pH_af, V_uc_af, epsilon, alpha_pc, M), the correlation
    with each angioparameter (volume fraction (vf), branches per area (bpa), branches per length (bpl), median radius
    (median_radius).
    :return:
    """
    input_data = OUT_POSTPROCESSING_FOLDER / Path("angiometrics_w1_d0_CTRL_H1.csv")
    input_df = pd.read_csv(input_data)
    sim_parameters_names = ["V_pH_af", "V_uc_af", "epsilon", "alpha_pc", "M"]
    angiometrics_names = ["vf", "bpa", "bpl", "median_radius"]

    # for each simulation parameters, generate a 2x2 plot showing the correlation with each angiometric
    for sim_parameter in sim_parameters_names:
        sim_parameters_values = input_df[sim_parameter].array  # get sim parameters values
        fig, axes = plt.subplots(2, 2, sharex=True, figsize=(16, 9))  # init figure
        fig.suptitle(f"Correlation of {sim_parameter} with angiometrics")

        for ax, angiometric in zip(axes.ravel(), angiometrics_names):
            # extract angiometric values at different times
            angiometric_percentage_variation_55 = input_df[f"%{angiometric} (t=55)"].array
            angiometric_percentage_variation_110 = input_df[f"%{angiometric} (t=110)"].array

            # compute correlations
            r_55 = np.corrcoef(sim_parameters_values, angiometric_percentage_variation_55)[0, 1]
            r_110 = np.corrcoef(sim_parameters_values, angiometric_percentage_variation_110)[0, 1]

            # generate scatter plots
            ax.scatter(sim_parameters_values, angiometric_percentage_variation_55,
                       label=f"day1 (r={r_55:.2g})")
            ax.scatter(sim_parameters_values, angiometric_percentage_variation_110,
                       label=f"day2 (r={r_110:.2g})")
            ax.set_xscale('log')
            ax.set_title(f"{angiometric}")
            ax.legend()

        plt.savefig(OUT_POSTPROCESSING_FOLDER / Path(f"{sim_parameter}_correlations.png"), dpi=300)


