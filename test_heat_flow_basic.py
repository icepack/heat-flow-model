import tqdm
import firedrake
from firedrake import Constant
import irksome
import heat_flow

lx, lz = 20e3, 3e3
nz = 19
nx = 40

mesh = firedrake.RectangleMesh(nx, nz, lx, lz, diagonal="crossed")
cg1 = firedrake.FiniteElement("CG", "triangle", 1)
Q = firedrake.FunctionSpace(mesh, cg1)

q = Constant(0.0)
q_b = Constant(0.0)
q_s = Constant(0.0)

T = firedrake.Function(Q)
x, z = firedrake.SpatialCoordinate(mesh)
Lx, Lz = Constant(lx), Constant(lz)
ξ, ζ = x / Lx, z / Lz
T_expr = 16 * ξ * (1 - ξ) * ζ * (1 - ζ)
T.interpolate(T_expr)

fields = {
    "temperature": T,
    "velocity": Constant((0.0, 0.0)),
    "temperature_in": Constant(0.0),
    "heat_source": q,
    "surface_flux": q_s,
    "basal_flux": q_b,
    "test_function": firedrake.TestFunction(Q),
}

ρ = Constant(917)
c = Constant(2.09)
k = Constant(2.2)

parameters = {
    "density": ρ,
    "heat_capacity": c,
    "conductivity": k,
}

G = heat_flow.form_problem_cartesian(**fields, **parameters)

sec_per_year = 24 * 60 * 60 * 365.25

t = Constant(0.0)
dt = Constant(sec_per_year)
final_time = 100 * sec_per_year
num_steps = int(final_time / float(dt))

method = irksome.BackwardEuler()
solver = irksome.TimeStepper(G, method, t, dt, T)

for step in tqdm.trange(num_steps):
    solver.advance()

print(T.dat.data_ro.min(), T.dat.data_ro.max())

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
colors = firedrake.tripcolor(T, axes=ax)
fig.colorbar(colors)
plt.show()
