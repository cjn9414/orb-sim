#!/usr/local/bin/python3

from random import random

BODIES=1000
PXMIN=0
PXMAX=50000
PYMIN=0
PYMAX=50000
VXMIN=-0.001
VXMAX=0.001
VYMIN=-0.001
VYMAX=0.001
MASS_MIN=10000
MASS_MAX=1000000

if __name__=='__main__':
    for i in range(BODIES):
        mass = MASS_MIN + random() * (MASS_MAX-MASS_MIN)
        px = PXMIN + random() * (PXMAX-PXMIN)
        py = PXMIN + random() * (PYMAX-PYMIN)
        vx = VXMIN + random() * (VXMAX-VXMIN)
        vy = VXMIN + random() * (VYMAX-VYMIN)
        print(f"{mass} {px} {py} {vx} {vy}")

