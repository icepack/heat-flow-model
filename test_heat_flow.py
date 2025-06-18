import tqdm
import firedrake
from firedrake import Constant
import irksome
import heat_flow


# interval = firedrake.IntervalMesh(nx, lx)
# mesh = firedrake.ExtrudedMesh(interval, nz)

lx, lz = 20e3, 1.0
nz = 19
nx = 40

mesh = firedrake.RectangleMesh(nx, nz, lx, lz, diagonal="crossed")
cg1 = firedrake.FiniteElement("CG", "triangle", 1)
Q = firedrake.FunctionSpace(mesh, cg1)

b = firedrake.Function(Q)
h = firedrake.Function(Q)
h.assign(3e3)

q = Constant(0.0)
q_b = Constant(0.0)
q_s = Constant(0.0)

T = firedrake.Function(Q)
x, ζ = firedrake.SpatialCoordinate(mesh)
Lx = Constant(lx)
T_expr = 16 * x / Lx * (1 - x / Lx) * ζ * (1 - ζ)
T.interpolate(T_expr)

fields = {
    "temperature": T,
    "bed": b,
    "thickness": h,
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

G = heat_flow.form_problem(**fields, **parameters)

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
