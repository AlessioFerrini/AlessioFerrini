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


def vascular_proliferation_form(alpha_p,
                                af: dolfinx.fem.Function,
                                af_p,
                                c: dolfinx.fem.Function or ufl.variable.Variable,
                                v: ufl.TestFunction):
    r"""
    Returns the UFL Form for the proliferation term of the vascular tissue as defined by the paper of Travasso et al.
    (2011) :cite:`Travasso2011a`.

    The corresponding term of the equation is (H is the Heaviside function):

    .. math::
       \alpha_p(af_p) \cdot c \cdot H(c)

    Where :math: `af` is the angiogenic factor concentration, and :math: `\alpha_p(af)` represents the proliferation
    rate, that is defined as the follwing function of :math: `af`. The definition of the latter function is the
    following:

    .. math::
       \alpha_p(af) &= \alpha_p \cdot af_p \quad \textrm{if} \quad af>af_p \\
                    &= \alpha_p \cdot af  \quad \textrm{if} \quad 0<af \le af_p \\
                    & = 0 \quad \textrm{if} \quad af \le 0

    Where :math: `\alpha-p` and :math: `af_p` are constants.

    :param alpha_p: costant of the proliferation rate function for the capillaries
    :param af: FEniCS function representing the angiogenic factor distribution
    :param af_p: maximum concentration of angiogenic factor leading to proliferation. If af > af_p, the proliferation
        rate remains alpha_p * af_p
    :param c: FEniCS function representing the capillaries
    :param v: FEniCS test function
    :return: the UFL form for the proliferation term
    """
    # def the proliferation function
    proliferation_function = alpha_p * af
    # def the max value for the proliferation function
    proliferation_function_max = alpha_p * af_p
    # take the bigger between the two of them
    proliferation_function_hysteresis = ufl.conditional(ufl.gt(proliferation_function,
                                                               proliferation_function_max),
                                                        proliferation_function_max,
                                                        proliferation_function)
    # multiply the proliferation term with the vessel field
    proliferation_term = proliferation_function_hysteresis * c
    # take it oly if bigger than 0
    proliferation_term_heaviside = ufl.conditional(ufl.gt(proliferation_term, 0.),
                                                   proliferation_term,
                                                   0.)
    # build the form
    proliferation_term_form = proliferation_term_heaviside * v * ufl.dx
    return proliferation_term_form


def cahn_hillard_form(c,
                      c0: dolfinx.fem.Function,
                      mu: dolfinx.fem.Function,
                      mu0: dolfinx.fem.Function,
                      q: dolfinx.fem.Function,
                      v: dolfinx.fem.Function,
                      dt,
                      theta,
                      chem_potential,
                      lmbda,
                      M):
    r"""
    Returns the UFL form of a for a general Cahn-Hillard equation, discretized in time using the theta method. The
    method is the same reported by the FEniCS team in one of their demo `1. Cahn-Hillard equation`_ and is briefly
    discussed below for your conveneince.

    .. _1. Cahn-Hillard equation:
       https://fenicsproject.org/olddocs/dolfin/2016.2.0/cpp/demo/documented/cahn-hilliard/cpp/documentation.html

    The Cahn-Hillard equation reads as follows:

    .. math::
       \frac{\partial c}{\partial t} - \nabla \cdot M (\nabla(\frac{d f}{d c}
             - \lambda \nabla^{2}c)) = 0 \quad \textrm{in} \ \Omega

    Where :math: `c` is the unknown field to find, :math: `f` is some kind of energetic potential which defines the
    phase separation, and :math: `M` is a scalar parameter.

    The equation involves 4th order derivatives, so its weak form could not be handled with the standard Lagrange
    finite element basis. However, the equation can be split in two second-order equations adding a second unknown
    auxiliary field :math: `\mu`:

    .. math::
       \frac{\partial c}{\partial t} - \nabla \cdot M \nabla\mu  &= 0 \quad \textrm{in} \ \Omega, \\
       \mu -  \frac{d f}{d c} + \lambda \nabla^{2}c &= 0 \quad \textrm{ in} \ \Omega.

    In this way, it is possible to solve this equation using the standard Lagrange basis and, indeed, this
    implementation uses this form.

    :param c: main Cahn-Hillard field
    :param c0: initial condition for the main Cahn-Hillard field
    :param mu: auxiliary field for the Cahn-Hillard equation
    :param mu0: initial condition for the auxiliary field
    :param q: test function for c
    :param v: test function for mu
    :param dt: time step
    :param theta: theta value for theta method
    :param chem_potential: UFL form for the Cahn-Hillard potential
    :param lmbda: energetic weight for the gradient of c
    :param M: scalar parameter
    :return: the UFL form of the Cahn-Hillard Equation
    """
    # Define form for mu (theta method)
    mu_mid = (1.0 - theta) * mu0 + theta * mu

    # chem potential derivative
    dfdc = ufl.diff(chem_potential, c)

    # define form
    l0 = ((c - c0) / dt) * q * ufl.dx + M * ufl.dot(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
    l1 = mu * v * ufl.dx - dfdc * v * ufl.dx - lmbda * ufl.dot(ufl.grad(c), ufl.grad(v)) * ufl.dx
    form = l0 + l1

    # return form
    return form


def angiogenesis_form(c: dolfinx.fem.Function,
                      c0: dolfinx.fem.Function,
                      mu: dolfinx.fem.Function,
                      mu0: dolfinx.fem.Function,
                      v1: ufl.TestFunction,
                      v2: ufl.TestFunction,
                      af: dolfinx.fem.Function,
                      parameters: Parameters = None,
                      **kwargs):
    r"""
    Returns the UFL form for the Phase-Field model for angiogenesis reported by Travasso et al. (2011)
    :cite:`Travasso2011a`.

    The equation reads simply as the sum of a Cahn-Hillard term and a proliferation term (for further details see
    the original paper):

    .. math::
       \frac{\partial c}{\partial t} = M \cdot \nabla^2 [\frac{df}{dc}\ - \epsilon \nabla^2 c]
       + \alpha_p(T) \cdot c H(c)

    Where :math: `c` is the unknown field representing the capillaries, and :

    .. math:: f = \frac{1}{4} \cdot c^4 - \frac{1}{2} \cdot c^2

    .. math::
       \alpha_p(af) &= \alpha_p \cdot af_p \quad \textrm{if} \quad af>af_p \\
                    &= \alpha_p \cdot af  \quad \textrm{if} \quad 0<af \le af_p \\
                    & = 0 \quad \textrm{if} \quad af \le 0

    In this implementation, the equation is split in two equations of lower order, in order to make the weak form
    solvable using standard Lagrange finite elements:

    .. math::
       \frac{\partial c}{\partial t} &= M \nabla^2 \cdot \mu + \alpha_p(T) \cdot c H(c) \\
       \mu &= \frac{d f}{d c} - \epsilon \nabla^{2}c

    :param c: capillaries field
    :param c0: initial condition for the capillaries field
    :param mu: auxiliary field
    :param mu0: initial condition for the auxiliary field
    :param v1: test function for c
    :param v2: test function  for mu
    :param af: angiogenic factor field
    :param parameters: simulation parameters
    :return:
    """
    # get parameters
    dt, epsilon, M, alpha_pc, T_p = _unpack_parameters_list(["dt", "epsilon", "M", "alpha_pc", "T_p"],
                                                            parameters,
                                                            kwargs)
    # define theta
    theta = 0.5

    # define chemical potential for the phase field
    c = ufl.variable(c)
    chem_potential = ((c ** 4) / 4) - ((c ** 2) / 2)

    # define total form
    form_cahn_hillard = cahn_hillard_form(c, c0, mu, mu0, v1, v2, dt, theta, chem_potential, epsilon, M)
    form_proliferation = vascular_proliferation_form(alpha_pc, af, T_p, c, v1)
    form = form_cahn_hillard - form_proliferation

    return form
