import tqdm
import matplotlib.pyplot as plt
import firedrake
from firedrake import Constant, as_vector
import irksome
import heat_flow

lx, lz = 20e3, 3e3
nz = 19
nx = 40

mesh = firedrake.RectangleMesh(nx, nz, lx, lz, diagonal="crossed")
cg1 = firedrake.FiniteElement("CG", "triangle", 1)
Q = firedrake.FunctionSpace(mesh, cg1)

q = Constant(0.0)
T = firedrake.Function(Q)
x, z = firedrake.SpatialCoordinate(mesh)
Lx, Lz = Constant(lx), Constant(lz)
ξ, ζ = x / Lx, z / Lz

u_0 = Constant(1.0)
w_0 = Constant(1e-3)
# Here I've fixed the vertical gradient of the horizontal velocity to do
# something sensible. You could change the power to `n + 1` to be the exact
# solution for SIA. I've also added some vertical velocity, and we need to
# be careful that it's divergence-free.
u_expr = as_vector((u_0 * (1 - (1 - ζ) ** 2) + w_0 * x / Lz, -w_0 * ζ))
V = firedrake.VectorFunctionSpace(mesh, cg1)
u = firedrake.Function(V).interpolate(u_expr)

T_b = Constant(-5)
T_s = Constant(-50)
T_in = (1 - ζ) * T_b + ζ * T_s

T.interpolate(T_in)

ρ = Constant(917)
c = Constant(2.09)
k = Constant(2.2)

# Here you can either pick a fixed surface and basal heat flux, or you can
# make it relax to some external temperature fields `T_b, T_s`, or you can
# do a combination of both. If you make it relax to some external values, you
# have to pick some exchange coefficients `σ_b, σ_s`. These have units of
# length. I don't know what those should be. I've chosen the basal heat flux
# to be equal to a commonly used value of 80 mW / m^2 (note the units).
q_b = Constant(80e-3)
σ_s = Constant(1e-3)
q_s = -σ_s * k * (T - T_s)

fields = {
    "temperature": T,
    "velocity": u,
    "temperature_in": T_in,
    "heat_source": q,
    "surface_flux": q_s,
    "basal_flux": q_b,
    "test_function": firedrake.TestFunction(Q),
}

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
solver_parameters = {
    "pc_type": "lu",
    "snes_atol": 1e-6,
}
solver = irksome.TimeStepper(
    G, method, t, dt, T, solver_parameters=solver_parameters,
)

for step in tqdm.trange(num_steps):
    solver.advance()

print(T.dat.data_ro.min(), T.dat.data_ro.max())

fig, ax = plt.subplots()
colors = firedrake.tripcolor(T, axes=ax, cmap="inferno")
fig.colorbar(colors)
plt.show()
