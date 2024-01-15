"""
File containing the FEniCS Forms used throughout the simulation
"""
import logging
import ufl
import dolfinx
from dolfinx.fem import Function, Constant
from mocafe.fenut.parameters import Parameters, _unpack_parameters_list

logger = logging.getLogger(__name__)


def shf(mesh: dolfinx.mesh.Mesh, variable, slope: float = 100):
    r"""
    Smoothed Heavyside Function (SHF) using the sigmoid function, which reads:

    .. math::
        \frac{e^{slope * variable}}{(1 + e^{slope * variable})}


    :param mesh: domain (necessary for constant)
    :param variable: varible for the SHF
    :param slope: slope of the SHF. Default is 100
    :return: the value of the sigmoid for the given value of the variable
    """
    slope = Constant(mesh, dolfinx.default_scalar_type(slope))
    return ufl.exp(slope * variable) / (Constant(mesh, dolfinx.default_scalar_type(1.)) + ufl.exp(slope * variable))


def ox_form_eq(ox: Function,
               v: ufl.TestFunction,
               par: Parameters,
               **kwargs):
    D_ox, V_u_ox = _unpack_parameters_list(["D_ox", "V_u_ox"], par, kwargs)

    # set to constants
    if isinstance(D_ox, float):
        D_ox = Constant(ox._V.mesh, dolfinx.default_scalar_type(D_ox))
    if isinstance(V_u_ox, float):
        V_u_ox = Constant(ox._V.mesh, dolfinx.default_scalar_type(V_u_ox))

    ox_form = ((D_ox * ufl.dot(ufl.grad(ox), ufl.grad(v)) * ufl.dx)
               + (V_u_ox * ox * v * ufl.dx))

    return ox_form


def angiogenic_factors_form_dt(af: Function,
                               af_old: Function,
                               ox: Function,
                               c: Function,
                               v: ufl.TestFunction,
                               par: Parameters,
                               **kwargs):
    """
    Time variant angiogenic factors form.
    """
    form = time_derivative_form(af, af_old, v, par, **kwargs) + angiogenic_factors_form_eq(af, ox, c, v, par, **kwargs)
    return form


def angiogenic_factors_form_eq(af: Function,
                               ox: Function,
                               c: Function,
                               v: ufl.TestFunction,
                               par: Parameters,
                               **kwargs):
    """
    Equilibrium angiogenic factors form.
    """
    # load parameters
    D_af, V_pH_af, V_uc_af, V_d_af = _unpack_parameters_list(["D_af", "V_pH_af", "V_uc_af", "V_d_af"],
                                                             par, kwargs)
    # transform in constants
    mesh = ox.function_space.mesh
    D_af = Constant(mesh, dolfinx.default_scalar_type(D_af))
    V_pH_af = Constant(mesh, dolfinx.default_scalar_type(V_pH_af))
    V_uc_af = Constant(mesh, dolfinx.default_scalar_type(V_uc_af))
    V_d_af = Constant(mesh, dolfinx.default_scalar_type(V_d_af))

    # define each part of the form
    # diffusion
    diffusion_form = (D_af * ufl.dot(ufl.grad(af), ufl.grad(v)) * ufl.dx)
    # production
    production_form = V_pH_af * shf(mesh, Constant(mesh, dolfinx.default_scalar_type(0.2)) - ox) * v * ufl.dx
    # uptake
    uptake_term = V_uc_af * shf(mesh, c)
    uptake_term_non_negative = ufl.conditional(
        condition=ufl.gt(uptake_term, Constant(mesh, dolfinx.default_scalar_type(0.))),
        true_value=uptake_term,
        false_value=Constant(mesh, dolfinx.default_scalar_type(0.))
    )
    uptake_form = uptake_term_non_negative * af * v * ufl.dx
    # degradation
    degradation_form = V_d_af * af * v * ufl.dx

    # assemble form
    form = diffusion_form - production_form + uptake_form + degradation_form

    return form


def time_derivative_form(var: Function,
                         var_old: Function,
                         v: ufl.TestFunction,
                         par: Parameters,
                         **kwargs):
    """
    General time derivative form.
    """
    dt, = _unpack_parameters_list(["dt"], par, kwargs)
    if isinstance(dt, int) or isinstance(dt, float):
        dt = Constant(var_old.function_space.mesh, dolfinx.default_scalar_type(dt))
    return ((var - var_old) / dt) * v * ufl.dx


def chan_hillard_free_enery(mu: Function):

    energy = - (ufl.grad(mu) ** 2) * ufl.dx

    return energy
