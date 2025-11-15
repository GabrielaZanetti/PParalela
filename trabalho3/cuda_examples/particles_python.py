#!/usr/bin/env python3
"""
particles_python.py - Simple particle simulation (Python + NumPy)
Outputs particle positions/velocities to CSV.
"""

import numpy as np
import sys

def simulate(N, steps, dt, damping=0.999, G=1.0):
    """Particle simulation with central attraction"""
    pos = np.random.uniform(-1, 1, (N, 3))
    vel = np.zeros((N, 3))
    
    for _ in range(steps):
        r2 = np.sum(pos**2, axis=1, keepdims=True) + 1e-6
        r_inv3 = r2 ** (-1.5)
        acc = -G * pos * r_inv3
        
        vel += acc * dt
        pos += vel * dt
        vel *= damping
    
    return pos, vel

def save_csv(filename, pos, vel):
    """Save particle data to CSV"""
    with open(filename, 'w') as f:
        f.write("id,x,y,z,vx,vy,vz\n")
        for i in range(len(pos)):
            f.write(f"{i},{pos[i,0]:.6f},{pos[i,1]:.6f},{pos[i,2]:.6f},"
                   f"{vel[i,0]:.6f},{vel[i,1]:.6f},{vel[i,2]:.6f}\n")
    print(f"Wrote {filename} (N={len(pos)}, steps={steps})")

if __name__ == "__main__":
    np.random.seed(1234)
    N, steps, dt = 20000, 200, 0.01
    out = "particles.csv"
    
    if len(sys.argv) >= 2:
        N = int(sys.argv[1])
    if len(sys.argv) >= 3:
        steps = int(sys.argv[2])
    if len(sys.argv) >= 4:
        dt = float(sys.argv[3])
    if len(sys.argv) >= 5:
        out = sys.argv[4]
    
    pos, vel = simulate(N, steps, dt)
    save_csv(out, pos, vel)
