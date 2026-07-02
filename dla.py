import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


#change

#3*x**2 + 2*x*y + 6*y**2 - 4*x + 3*y | quadratic | start[4, -4] | x[-5, 5], y[-5, 5]
#(1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2 | Beale function | start[-3, -3] | x[-4.5 4.5], y[-4.5, 4.5] | min at 3.0.5
#(x**2 + y - 11)**2 + (x + y**2 - 7)**2 | Himmelblau function | start[-4, 4], [4, 4], [-4, -4], [4, -4] | x[-6, 6], y[-6, 6]
#100*(y - x**2)**2 + (1 - x)**2 | Rosenbrock function | start[-1.5, 1.5] | x[-2, 2], y[-1, 3]
FUNC = "(1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2"
START = np.array([-3, -3], dtype=float)

DELTA0 = 0.3
DELTA_MAX = 1.5
ETA = 0.25

X_RANGE = (-4.5, 4.5)
Y_RANGE = (-4.5, 4.5)


#gradient and hessian======================================================================

x, y = sp.symbols("x y")
expr = sp.sympify(FUNC.replace("^", "**"))

grad_expr = [sp.diff(expr, x), sp.diff(expr, y)]
hess_expr = sp.Matrix([
    [sp.diff(expr, x, x), sp.diff(expr, x, y)],
    [sp.diff(expr, y, x), sp.diff(expr, y, y)]
])

#f(x,y)
f_raw = sp.lambdify((x, y), expr, "numpy")
g_raw = sp.lambdify((x, y), grad_expr, "numpy")
H_raw = sp.lambdify((x, y), hess_expr, "numpy")

#f((x,y)) = f(x_k) = f(vector)
def f(z):
    return float(f_raw(z[0], z[1]))


def grad(z):
    return np.array(g_raw(z[0], z[1]), dtype=float).reshape(2)


def hess(z):
    return np.array(H_raw(z[0], z[1]), dtype=float)

print(f_raw)

#algorithm step =================================================================================

#make hess positive definite
def make_spd(B):
    B = 0.5 * (B + B.T)
    smallest = np.min(np.linalg.eigvalsh(B))

    if smallest > 0:
        return B

    return B + (abs(smallest) + 1e-4) * np.eye(2)


#f'(xk), f''(xk), delta_k
def dogleg_step(g, B, Delta):
    B = make_spd(B)

    #newton point: unconstrained minimizer of the quadratic model
    #Bx+g=0 -> Bx = -g
    pN = np.linalg.solve(B, -g)

    if np.linalg.norm(pN) <= Delta:
        return pN, None, pN, "Newton step"

    # Cauchy point
    # = - ||f'(xk)|| / (f'k (f''k) f'k)
    alpha = (g @ g) / (g @ B @ g)
    pC = -alpha * g

    if np.linalg.norm(pC) >= Delta:
        p = Delta * pC / np.linalg.norm(pC)
        return p, pC, pN, "Cauchy boundary"

    #dogleg intersection between pC and pN
    d = pN - pC

    #we solve ||pC + tau*d||^2 = Delta^2
    #(d.d)tau^2 + 2(pC.d)tau + pC.pC-delta^2 = 0
    a = d @ d
    b = 2 * (pC @ d)
    c = pC @ pC - Delta**2

    tau = (-b + np.sqrt(max(b*b - 4*a*c, 0))) / (2*a)
    p = pC + tau * d

    return p, pC, pN, "Inbetween"

# -(1/2 pBp + gp) = -m_k(p)
def predicted_reduction(g, B, p):
    return -(g @ p + 0.5 * p @ B @ p)


#state vars ======================================================================================

xk = START.copy()
Delta = DELTA0
k = 0
path = [xk.copy()]
status = "ready"


def get_step_data():
    g = grad(xk)
    B = make_spd(hess(xk))

    p, pC, pN, step_type = dogleg_step(g, B, Delta)

    actual = f(xk) - f(xk + p)
    predicted = predicted_reduction(g, B, p)

    rho = actual / predicted if predicted > 0 else -np.inf

    return g, B, p, pC, pN, rho, step_type


#background, contour levels, plotting values for f =================================================================================

#n in x-direction, n in y-direction points
grid_n = 450

xs = np.linspace(X_RANGE[0], X_RANGE[1], grid_n)
ys = np.linspace(Y_RANGE[0], Y_RANGE[1], grid_n)
X, Y = np.meshgrid(xs, ys)

Z = np.asarray(f_raw(X, Y), dtype=float)

#make contours readable
#plot log-shifted values, but algorithm still uses real f

#array, which is finite
finite = np.isfinite(Z)

#minimum finite z
Z_min = np.nanmin(Z[finite])

#this will store transformed values
Z_plot = np.full_like(Z, np.nan, dtype=float)
Z_plot[finite] = np.log10(np.maximum(Z[finite] - Z_min + 1e-8, 1e-8))

low = np.nanpercentile(Z_plot[finite], 0) #8
high = np.nanpercentile(Z_plot[finite], 100) #92
levels = np.linspace(low, high, 50)


#plot ===================================================================================

#window, plotting area, (size)
fig, ax = plt.subplots(figsize=(9, 7))

#room bottom, right (0.17, 0.78)
plt.subplots_adjust(bottom=0.09, right=0.8)

next_ax = plt.axes([0.05, 0.05, 0.05, 0.05])
reset_ax = plt.axes([0.1, 0.05, 0.05, 0.05])

next_button = Button(next_ax, "Next")
reset_button = Button(reset_ax, "Reset")


def draw():
    if status == "converged": print(xk)
    ax.clear()

    g, B, p, pC, pN, rho, step_type = get_step_data()

    proposed = xk + p
    newton = xk + pN
    path_array = np.array(path)

    #contours
    ax.contour(
        X,
        Y,
        Z_plot,
        levels=levels,
        linewidths=0.7, #thin countour lines
        alpha=0.95 #transparent contour lines
    )

    #trust region circle
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(
        xk[0] + Delta*np.cos(theta),
        xk[1] + Delta*np.sin(theta),
        "--",
        linewidth=2,
        label="trust region"
    )

    #accepted path
    ax.plot(
        path_array[:, 0], #all x-coordiantes
        path_array[:, 1], #all y-coordinates
        "o-",
        linewidth=3,
        markersize=6,
        label="accepted path"
    )

    #proposed step
    ax.plot(
        [xk[0], proposed[0]],
        [xk[1], proposed[1]],
        linewidth=4,
        label="proposed step"
    )

    #dogleg broken path
    if pC is not None:
        cauchy = xk + pC

        ax.plot(
            [xk[0], cauchy[0], newton[0]],
            [xk[1], cauchy[1], newton[1]],
            ".--",
            linewidth=2.5,
        )

        ax.scatter(
            cauchy[0],
            cauchy[1],
            marker="s",
            s=90,
            label="Cauchy point"
        )

    #points
    ax.scatter(xk[0], xk[1], s=110, label="current point")
    ax.scatter(proposed[0], proposed[1], s=110, label="proposed point")
    ax.scatter(newton[0], newton[1], marker="x", s=140, label="Newton point")

    ax.set_title(
        f" {FUNC} \n k ={k} | {status} | "
        f"f = {f(xk):.5g}, ||grad|| = {np.linalg.norm(g):.2e}, "
        f"Delta = {Delta:.4g}, rho = {rho:.4g}"
    )

    ax.set_xlim(X_RANGE)
    ax.set_ylim(Y_RANGE)
    ax.set_aspect("equal", adjustable="box") #1:1 x:y
    ax.grid(True, alpha=0.25) #GRID LINES

    # ax.legend(
    #     loc="upper left",
    #     bbox_to_anchor=(1.02, 1.0),
    #     borderaxespad=0
    # )

    fig.canvas.draw_idle()


def next_step(event=None): #event for pressing n
    global xk, Delta, k, path, status

    g, B, p, pC, pN, rho, step_type = get_step_data()

    #print(B)
    #print(pN)
    if np.linalg.norm(g) < 1e-8:
        status = "converged"
        draw()
        return

    #update trust-region radius
    if rho < 0.25:
        Delta *= 0.25
    elif rho > 0.75 and abs(np.linalg.norm(p) - Delta) < 1e-8: #or just rho > 0.75
        Delta = min(2 * Delta, DELTA_MAX)

    #accept or reject
    if rho > ETA:
        xk = xk + p
        path.append(xk.copy())
        status = "accepted"
    else:
        status = "rejected"

    k += 1
    draw()


def reset(event=None):
    global xk, Delta, k, path, status

    xk = START.copy()
    Delta = DELTA0
    k = 0
    path = [xk.copy()]
    status = "ready"

    draw()


def key_press(event):
    if event.key == "n":
        next_step()
    elif event.key == "r":
        reset()


next_button.on_clicked(next_step)
reset_button.on_clicked(reset)
fig.canvas.mpl_connect("key_press_event", key_press)



draw()
plt.show()
