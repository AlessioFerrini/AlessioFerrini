import json
import time
import logging
import argparse
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

N_STEPS_2_DAYS = 111


def timer(py_func):
    t0 = time.perf_counter()
    py_func()
    tfin = time.perf_counter() - t0
    if MPI.COMM_WORLD.rank == 0:
        logger.info(f"Execution time ({MPI.COMM_WORLD.size}): {tfin}")


def cli():
    parser = argparse.ArgumentParser(description="Simple CLI for RH simulations")
    # add slurm_job_id
    parser.add_argument("-slurm_job_id",
                        type=int,
                        help="Slurm job ID for the simulation")
    # add flag for test timulation
    parser.add_argument("-run_2d",
                        action="store_true",
                        help="Run the simulation in 2d to check if everything runs smoothly")
    return parser.parse_args()


def preamble():
    """
    Load general data for simulations
    """
    # load simulation parameters
    parameters_csv = "parameters/parameters.csv"
    standard_parameters_df = pd.read_csv(parameters_csv, index_col="name")
    sim_parameters = Parameters(standard_parameters_df)

    # get cli args, if any
    args = cli()

    # load patient parameters
    with open("input_data/all_eggs_parameters.json", "r") as infile:
        patients_parameters = json.load(infile)

    if args.run_2d:
        spatial_dimension = 2
        distributed_data_folder = "temp"
    else:
        spatial_dimension = 3
        distributed_data_folder = "/local/frapra/3drh"

    return sim_parameters, patients_parameters, args.slurm_job_id, spatial_dimension, distributed_data_folder


def oxygen_consumption_test():
    """
    Experiment to check if the oxygen profile with a fake is the one expected.

    :return: None
    """
    # import parameters
    sim_parameters = Parameters(pd.read_csv("parameters/parameters.csv"))

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


def compute_initial_conditions():
    sim_parameters, eggs_parameters, slurm_job_id, spatial_dimension, distributed_data_folder = preamble()

    egg_code = "w1_d0_CTRL_H1"

    sim = CAMTimeSimulation(spatial_dimension=2,
                            sim_parameters=sim_parameters,
                            egg_parameters=eggs_parameters["w1_d0_CTRL_H1"],
                            steps=0,
                            out_folder_name=f"{egg_code}_initial_condition")
    sim.run()


def vascular_sprouting():
    sim_parameters, eggs_parameters, slurm_job_id, spatial_dimension, distributed_data_folder = preamble()

    egg_code = "w1_d0_CTRL_H1"

    sim = CAMTimeSimulation(spatial_dimension=2,
                            sim_parameters=sim_parameters,
                            egg_parameters=eggs_parameters["w1_d0_CTRL_H1"],
                            steps=N_STEPS_2_DAYS,
                            out_folder_name=f"{egg_code}_vascular_sprouting_2days",
                            save_distributed_files_to=distributed_data_folder)
    sim.run()