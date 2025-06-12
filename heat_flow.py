import ufl
import firedrake
from firedrake import Constant, inner, dot, grad, dx, ds, as_vector, as_tensor
import irksome
from irksome import Dt


def form_jacobian(x, b, h):
    shape = x.ufl_shape
    if shape ==(2,):
        g = b.dx(0) + x[1] * h.dx(0)
        return as_tensor(
            [
                [1, 0],
                [-g / h, 1 / h],
            ],
        )
    elif shape == (3,):
        g = as_vector([b.dx(0) + x[2] * h.dx(0), b.dx(1) + x[2] * h.dx(1)])
        return as_tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [-g[0] / h, -g[1] / h, 1 / h],
            ]
        )
    else:
        raise ValueError("WAT")


def form_problem(**kwargs):
    field_names = (
        "temperature",
        "bed",
        "thickness",
        "heat_source",
        "surface_flux",
        "basal_flux",
        "test_function",
    )
    T, b, h, q, q_s, q_b, φ = map(kwargs.get, field_names)

    parameter_names = ("density", "heat_capacity", "conductivity")
    ρ, c, k = map(kwargs.get, parameter_names)

    mesh = ufl.domain.extract_unique_domain(T)
    x = firedrake.SpatialCoordinate(mesh)
    J = form_jacobian(x, b, h)

    K = k * dot(J, J.T)
    F = (ρ * c * Dt(T) * φ + inner(dot(K, grad(T)), grad(φ)) - q * φ) * h * dx
    return F
