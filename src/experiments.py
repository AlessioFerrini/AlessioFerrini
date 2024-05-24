import json
import time
import logging
import argparse
from pathlib import Path
from itertools import product
from tqdm import tqdm
import numpy as np
import pandas as pd
import dolfinx
from dolfinx.nls.petsc import NewtonSolver
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from mocafe.fenut.parameters import Parameters
from mocafe.fenut.fenut import get_colliding_cells_for_points
from src.forms import ox_form_eq
from src.simulation import CAMTimeSimulation

logger = logging.getLogger(__name__)

N_STEPS_2_DAYS = 110


def timer(py_func):
    t0 = time.perf_counter()
    py_func()
    tfin = time.perf_counter() - t0
    if MPI.COMM_WORLD.rank == 0:
        logger.info(f"Execution time ({MPI.COMM_WORLD.size}): {tfin}")


def cli():
    """
    CLI: Command Line Interface

    """
    parser = argparse.ArgumentParser(description="Simple CLI for RH simulations")
    # add slurm_job_id
    parser.add_argument("-slurm_job_id",
                        type=int,
                        help="Slurm job ID for the simulation")
    return parser.parse_args()


def preamble():
    """
    Load general data for simulations
    """
    # load simulation parameters
    parameters_csv = "debugging_sim_413/sim_parameters_413.csv"#"/home/alefer/github/cam_mocafe/parameters/parameters.csv"
    standard_parameters_df = pd.read_csv(parameters_csv, index_col="name")
    sim_parameters = Parameters(standard_parameters_df)

    # get cli args, if any
    args = cli()

    # load eggs parameters
    with open("/home/alefer/github/cam_mocafe/input_data/all_eggs_parameters.json", "r") as infile:
        patients_parameters = json.load(infile)

    distributed_data_folder = "debugging_sim_413"
    # if args.slurm_job_id is None:
    #     distributed_data_folder = "temp"
    # else:
    #     distributed_data_folder = "/local/frapra/cam"

    return sim_parameters, patients_parameters, args.slurm_job_id, distributed_data_folder


def oxygen_consumption_test():
    """
    Experiment to check if the oxygen profile with a fake is the one expected.

    :return: None
    """
    # import parameters
    sim_parameters = Parameters(pd.read_csv("/home/alefer/github/cam_mocafe/parameters/parameters.csv"))

    # define rectangular mesh
    Lx = Ly = 0.2
    nx = ny = 200
    mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, points=[[0., 0.], [Lx, Ly]], n=[nx, ny])
    bbt = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)

    # define function space
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))

    # define fake capillary
    c = dolfinx.fem.Function(V)
    c.interpolate(lambda x: np.where(x[0] < 0.02, 1., -1))
    c.x.scatter_forward()

    # define capillaries locator
    def capillaries_locator(x):
        points_on_proc, cells = get_colliding_cells_for_points(x.T, mesh, bbt)
        c_values = np.ravel(c.eval(points_on_proc, cells))
        return c_values > 0

    # define dofs where Dirichled BC should be applied
    cells_bc = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, capillaries_locator)
    dofs_bc = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim, cells_bc)
    bc = dolfinx.fem.dirichletbc(PETSc.ScalarType(1.), dofs_bc, V)

    # define weak form for oxygen
    ox = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    ox_form = ox_form_eq(ox, v, sim_parameters)

    # define problem
    problem = dolfinx.fem.petsc.NonlinearProblem(ox_form, ox, bcs=[bc])

    # define solver
    lsp = {"ksp_type": "preonly", "pc_type": "lu"}
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    # set options for krylov solver
    opts = PETSc.Options()
    option_prefix = solver.krylov_solver.getOptionsPrefix()
    for o, v in lsp.items():
        opts[f"{option_prefix}{o}"] = v
    solver.krylov_solver.setFromOptions()

    # solve
    solver.solve(ox)

    # define points on line at 20 ums
    line_20ums = np.array([[0.04, y, 0.] for y in np.linspace(start=0., stop=Ly, num=11)])

    # get mean value over line
    points_on_proc, cells = get_colliding_cells_for_points(line_20ums, mesh, bbt)
    mean_20ums_max = np.mean(np.ravel(ox.eval(points_on_proc, cells)))
    logger.info(f"Mean value: {mean_20ums_max} (should be close to 0.2)")

    # save solution
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "saved_sim/ox_calibration/ox.xdmf", "w") as outfile:
        outfile.write_mesh(mesh)
        outfile.write_function(ox)


def test_convergence_1_step():
    sim_parameters, eggs_parameters, slurm_job_id, distributed_data_folder = preamble()

    egg_code = "w1_d0_CTRL_H1"

    egg_parameters = eggs_parameters[egg_code]

    sim = CAMTimeSimulation(sim_parameters=sim_parameters,
                            egg_parameters=egg_parameters,
                            steps=1,
                            save_rate=10,
                            out_folder_name=f"dolfinx_{egg_code}_preconditioners",
                            out_folder_mode=None,
                            sim_rationale="Testing",
                            slurm_job_id=slurm_job_id,
                            save_distributed_files_to=distributed_data_folder)

    # set up the convergence test
    sim.setup_convergence_test()

    # setup list for storing performance
    performance_dicts = []

    # setup list of linear solver parameters to test
    lsp_list = []

    # --------------------------------------------------------------------------------------------------------------- #
    # Add Iterative solvers to lsp list                                                                               #
    # --------------------------------------------------------------------------------------------------------------- #
    # create list of solver and preconditioners
    iterative_solver_list = ["cg", "gmres"]
    pc_type_list = ["jacobi", "bjacobi", "sor", "asm", "gasm", "gamg"]

    # add all combinations to lsp list
    lsp_list.extend([{"ksp_type": solver, "pc_type": pc, "ksp_monitor": None}
                     for solver, pc in product(iterative_solver_list, pc_type_list)])

    # add all combination using mumps as backend
    lsp_list.extend([{"ksp_type": solver, "pc_type": pc, "ksp_monitor": None, "pc_factor_mat_solver_type": "mumps"}
                     for solver, pc in product(iterative_solver_list, pc_type_list)])

    # add hypre preconditioners
    hypre_type_list = ["euclid", "pilut", "parasails", "boomeramg"]
    lsp_list.extend([{"ksp_type": solver, "pc_type": "hypre", "pc_hypre_type": hypre_type, "ksp_monitor": None}
                     for solver, hypre_type in product(iterative_solver_list, hypre_type_list)])

    # --------------------------------------------------------------------------------------------------------------- #
    # Add Direct solvers to lsp list                                                                                  #
    # --------------------------------------------------------------------------------------------------------------- #
    direct_solver_list = ["lu", "cholesky"]
    lsp_list.extend([{"ksp_type": "preonly", "pc_type": ds, "ksp_monitor": None}
                    for ds in direct_solver_list])
    # add also with mumps
    lsp_list.extend([{"ksp_type": "preonly", "pc_type": ds, "ksp_monitor": None, "pc_factor_mat_solver_type": "mumps"}
                    for ds in direct_solver_list])

    # --------------------------------------------------------------------------------------------------------------- #
    # Iterate
    # --------------------------------------------------------------------------------------------------------------- #
    if MPI.COMM_WORLD.rank == 0:
        pbar_file = open("convergence_pbar.o", "w")
    else:
        pbar_file = None
    pbar = tqdm(total=len(lsp_list), ncols=100, desc="convergence_test", file=pbar_file,
                disable=True if MPI.COMM_WORLD.rank != 0 else False)

    for lsp in lsp_list:
        # get characteristics of lsp
        current_solver = lsp['ksp_type']
        if lsp['pc_type'] == "hypre":
            current_pc = f"{lsp['pc_type']} ({lsp['pc_hypre_type']})"
        else:
            current_pc = lsp['pc_type']
        using_mumps = ("mumps" in lsp.values())

        # logging
        msg = f"Testing solver {current_solver} with pc {current_pc}"
        if using_mumps:
            msg += f" (MUMPS)"
        logger.info(msg)

        # set linear solver parameters
        sim.lsp = lsp

        # time solution
        time0 = time.perf_counter()
        sim.test_convergence()
        tot_time = time.perf_counter() - time0

        # check if error occurred
        error = sim.runtime_error_occurred
        error_msg = sim.error_msg

        # build performance dict
        perf_dict = {
            "solver": current_solver,
            "pc": current_pc,
            "mumps": using_mumps,
            "time": tot_time,
            "error": error,
            "error_msg": error_msg
        }

        # append dict to list
        performance_dicts.append(perf_dict)
        df = pd.DataFrame(performance_dicts)
        if MPI.COMM_WORLD.rank == 0:
            df.to_csv(sim.data_folder / Path("performance.csv"))

        # reset runtime error and error msg
        sim.runtime_error_occurred = False
        sim.error_msg = None

        # update pbar
        pbar.update(1)

    if MPI.COMM_WORLD.rank == 0:
        pbar_file.close()


def compute_initial_conditions():
    sim_parameters, eggs_parameters, slurm_job_id, distributed_data_folder = preamble()

    egg_code = "w1_d0_CTRL_H1"

    sim = CAMTimeSimulation(sim_parameters=sim_parameters,
                            egg_parameters=eggs_parameters["w1_d0_CTRL_H1"],
                            steps=0,
                            out_folder_name=f"{egg_code}_initial_condition")
    sim.run()


def run_sim_413():
    # Inspired from sprouting from param sampling; load 413 param; 
    sim_parameters, eggs_parameters, slurm_job_id, distributed_data_folder = preamble()
    sim_parameters_path = "debugging_sim_413/sim_parameters_413.csv" #debugging_sim_413/sim_parameters_413.csv
    egg_code = "w1_d0_CTRL_H1" 

    # Load simulation parameters from CSV file
    sim_parameters_df = pd.read_csv(sim_parameters_path, index_col='name')

    # Set precise values for parameters from the CSV file
    V_pH_af_val = float(sim_parameters_df.loc["V_pH_af", "sim_value"])
    V_uc_af_val = float(sim_parameters_df.loc["V_uc_af", "sim_value"])
    epsilon_val = float(sim_parameters_df.loc["epsilon", "sim_value"])
    alpha_pc_val = float(sim_parameters_df.loc["alpha_pc", "sim_value"])
    M_val = float(sim_parameters_df.loc["M", "sim_value"])
    dt = 1

    # Set parameters
    sim_parameters.set_value("V_pH_af", V_pH_af_val)
    sim_parameters.set_value("V_uc_af", V_uc_af_val)
    sim_parameters.set_value("epsilon", epsilon_val)
    sim_parameters.set_value("alpha_pc", alpha_pc_val)
    sim_parameters.set_value("M", M_val)
    sim_parameters.set_value("dt", dt)

    # Generate sim object
    sim = CAMTimeSimulation(sim_parameters=sim_parameters,
                            egg_parameters=eggs_parameters["w1_d0_CTRL_H1"],
                            slurm_job_id=slurm_job_id,
                            steps=int(110 / dt),
                            save_rate=int(110 / dt),  # modified to save one step
                            out_folder_name=f"debugging_sim_413/output_no_var",
                            sim_rationale=f"Testing combination: "
                                          f"V_pH_af: {V_pH_af_val}; V_uc_af: {V_uc_af_val}; epsilon: {epsilon_val}; "
                                          f"alpha_pc: {alpha_pc_val}; M: {M_val}",
                            save_distributed_files_to=distributed_data_folder)
    
    # Run simulation
    sim.run()

    # Generate sim dictionary
    sim_dict = {
        "sim_i": 0,
        "V_pH_af": V_pH_af_val,
        "V_uc_af": V_uc_af_val,
        "epsilon": epsilon_val,
        "alpha_pc": alpha_pc_val,
        "M": M_val,
        "ERROR": sim.runtime_error_occurred,
        "Error msg": sim.error_msg
    }

    # Save the output
    #if MPI.COMM_WORLD.rank == 0:
    pd.DataFrame([sim_dict]).to_csv("convergence_2days_debugged.csv", index=False)

def sprouting_for_parameters_sampling():
    sim_parameters, eggs_parameters, slurm_job_id, distributed_data_folder = preamble()

    egg_code = "w1_d0_CTRL_H1"

    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # set parameters to test
    V_pH_af_min = float(sim_parameters_df.loc["V_pH_af", "sim_range_min"])
    V_pH_af_max = float(sim_parameters_df.loc["V_pH_af", "sim_range_max"])
    V_pH_af_range = np.logspace(start=np.log10(V_pH_af_min), stop=np.log10(V_pH_af_max), num=4, endpoint=True)
    V_uc_af_min = float(sim_parameters.get_value("V_d_af"))
    V_uc_af_range = np.logspace(start=0, stop=3, num=4, endpoint=True) * V_uc_af_min
    epsilon_range = np.logspace(start=-1, stop=1, num=3, endpoint=True) * float(sim_parameters.get_value("epsilon"))
    alpha_pc_range = np.logspace(start=-4, stop=0, num=5, endpoint=True) * float(sim_parameters.get_value("alpha_pc"))
    M_range = np.logspace(start=-1, stop=1, num=3, endpoint=True) * float(sim_parameters.get_value("M"))

    # do product to test all possible combinations
    combinations = list(product(V_pH_af_range, V_uc_af_range, epsilon_range, alpha_pc_range, M_range))

    # set pbar
    if MPI.COMM_WORLD.rank == 0:
        pbar_file = open("convergence_pbar.o", "w")
    else:
        pbar_file = None
    pbar = tqdm(total=len(combinations), ncols=100, desc="convergence_test", file=pbar_file,
                disable=True if MPI.COMM_WORLD.rank != 0 else False)

    # set dict to hold errors
    out = []

    for sim_i, (V_pH_af_val, V_uc_af_val, epsilon_val, alpha_pc_val, M_val) in enumerate(combinations):
        # set parameters
        sim_parameters.set_value("V_pH_af", V_pH_af_val)
        sim_parameters.set_value("V_uc_af", V_uc_af_val)
        sim_parameters.set_value("epsilon", epsilon_val)
        sim_parameters.set_value("alpha_pc", alpha_pc_val)
        sim_parameters.set_value("M", M_val)

        # generate sim object
        sim = CAMTimeSimulation(sim_parameters=sim_parameters,
                                egg_parameters=eggs_parameters["w1_d0_CTRL_H1"],
                                slurm_job_id=slurm_job_id,
                                steps=N_STEPS_2_DAYS,
                                save_rate=int(np.floor(N_STEPS_2_DAYS / 2)), 
                                out_folder_name=f"{egg_code}_2days_{str(sim_i).zfill(3)}",
                                sim_rationale=f"Testing combination: "
                                              f"V_pH_af: {V_pH_af_val}; V_uc_af: {V_uc_af_val}; epsilon: {epsilon_val}"
                                              f"alpha_pc: {alpha_pc_val}; M: {M_val}",
                                save_distributed_files_to=distributed_data_folder)
        # run
        sim.run()

        # generate sim dictionary
        sim_dict = {
            "sim_i": sim_i,
            "V_pH_af": V_pH_af_val,
            "V_uc_af": V_uc_af_val,
            "epsilon": epsilon_val,
            "alpha_pc": alpha_pc_val,
            "M": M_val,
            "ERROR": sim.runtime_error_occurred,
            "Error msg": sim.error_msg
        }

        # append to out
        out.append(sim_dict)

        # update pbar
        pbar.update(1)
        break

    # at the end, close files and save the out
    if MPI.COMM_WORLD.rank == 0:
        pbar_file.close()
        pd.DataFrame(out).to_csv("convergence_2days.csv", index=False)


def vascular_sprouting():
    sim_parameters, eggs_parameters, slurm_job_id, distributed_data_folder = preamble()

    egg_code = "w1_d0_CTRL_H1"

    alpha_p_range = np.logspace(start=-4, stop=0, num=5, endpoint=True) * float(sim_parameters.get_value("alpha_p"))

    for alpha_p_value in [alpha_p_range[-1]]:
        sim = CAMTimeSimulation(sim_parameters=sim_parameters,
                                egg_parameters=eggs_parameters["w1_d0_CTRL_H1"],
                                slurm_job_id=slurm_job_id,
                                steps=N_STEPS_2_DAYS,
                                save_rate=int(np.floor(N_STEPS_2_DAYS/2)),
                                out_folder_name=f"{egg_code}_vascular_sprouting_2days_alpha_p={alpha_p_value:.5g}",
                                save_distributed_files_to=distributed_data_folder)
        sim.run()