import numpy as np

### Fundamental constants

hbar = 1.0545718e-34
c = 2.99792458e8
e = 1.60217663e-19

### Properties of phase retarder (perfect diamond crystal) ###
a = 3.5668e-10

### Dielectric susceptibilities (H = (220)) ###
chi0 = -0.22580e-04 + 0.30548e-07j

### Bragg reflection properties ###
H = 2*2*np.sqrt(2)*np.pi / a