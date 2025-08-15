import numpy as np
from numba import njit
import matplotlib.pyplot as plt
G=1.0; SOFT=1e-2
@njit
def accelerations(pos, mass):
    N = pos.shape[0]; acc = np.zeros_like(pos)
    for i in range(N):
        for j in range(N):
            if i==j: continue
            dx = pos[j]-pos[i]
            r2 = (dx*dx).sum() + SOFT*SOFT
            inv = 1.0/(r2*np.sqrt(r2))
            acc[i] += G*mass[j]*dx*inv
    return acc
def simulate(n=40, steps=600, dt=0.01, seed=42):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1,1,(n,2)); vel = rng.normal(0,0.1,(n,2)); mass = rng.uniform(0.5,2.0,n)
    acc = accelerations(pos, mass); xs,ys=[],[]
    for _ in range(steps):
        vel += 0.5*dt*acc; pos += dt*vel; acc = accelerations(pos, mass); vel += 0.5*dt*acc
        xs.append(pos[:,0].copy()); ys.append(pos[:,1].copy())
    return np.array(xs), np.array(ys)
def main():
    xs,ys = simulate()
    plt.figure(figsize=(6,6))
    for i in range(xs.shape[1]): plt.plot(xs[:,i], ys[:,i], linewidth=0.6, alpha=0.85)
    plt.axis('equal'); plt.tight_layout(); plt.savefig('trajectory.png', dpi=200); print('Saved trajectory.png')
if __name__=='__main__': main()
