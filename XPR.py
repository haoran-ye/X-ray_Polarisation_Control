import numpy as np
from Constants import *

### Calculating transmitted and diffracted amplitudes in Loue geometry from incident plane wave ###

def Gaussian(E, E0, deltaE):
  return np.exp(-(E-E0)**2 / (2 * (deltaE/2.355)**2)) 

def calculate_sigma(E, t, theta):
  w = E / hbar
  thetaB = np.arcsin(H/2/(w/c))/np.pi*180
  chiH = 0.72138e-05 + 0.29084e-07j
  chiHbar = 0.72138e-05 + 0.29084e-07j

  alpha = 2*np.sin(np.deg2rad(theta))*H / (w/c) - H**2 / (w**2/c**2) # Deviation from Bragg

  delta1 = alpha/2 + np.sqrt(alpha**2/4 + chiH*chiHbar)
  delta2 = alpha/2 - np.sqrt(alpha**2/4 + chiH*chiHbar)

  u1 = (delta1 + chi0) / (2*np.cos(np.deg2rad(thetaB)))
  u2 = (delta2 + chi0) / (2*np.cos(np.deg2rad(thetaB)))

  v1 = delta1 / chiHbar
  v2 = delta2 / chiHbar

  Tsigma = (v1*np.exp(1j*u2*(w/c)*t) - v2*np.exp(1j*u1*(w/c)*t)) / (v1 - v2)
  Dsigma = (v1*v2*np.exp(1j*u2*(w/c)*t) - v2*v1*np.exp(1j*u1*(w/c)*t)) / (v1 - v2)

  return alpha, u1, u2, v1, v2, delta1, delta2, Dsigma, Tsigma

def calculate_pi(E, t, theta):
  w = E / hbar
  thetaB = np.arcsin(H/2/(w/c))/np.pi*180
  chiH = (0.72138e-05 + 0.29084e-07j) * np.cos(2*np.deg2rad(thetaB))
  chiHbar = (0.72138e-05 + 0.29084e-07j) * np.cos(2*np.deg2rad(thetaB))

  alpha = 2*np.sin(np.deg2rad(theta))*H / (w/c) - H**2 / (w**2/c**2) # Deviation from Bragg

  delta1 = alpha/2 + np.sqrt(alpha**2/4 + chiH*chiHbar)
  delta2 = alpha/2 - np.sqrt(alpha**2/4 + chiH*chiHbar)

  u1 = (delta1 + chi0) / (2*np.cos(np.deg2rad(thetaB)))
  u2 = (delta2 + chi0) / (2*np.cos(np.deg2rad(thetaB)))

  v1 = delta1 / chiHbar
  v2 = delta2 / chiHbar

  Tpi = (v1*np.exp(1j*u2*(w/c)*t) - v2*np.exp(1j*u1*(w/c)*t)) / (v1 - v2)
  Dpi = (v1*v2*np.exp(1j*u2*(w/c)*t) - v2*v1*np.exp(1j*u1*(w/c)*t)) / (v1 - v2)

  return alpha, u1, u2, v1, v2, delta1, delta2, Dpi, Tpi

### Calculating T and D in both polarisations with averaging over photon energy ###

def bandwidth_avg_Tsigma(E0, deltaE, step, t, theta):
  E = np.arange(E0 - 3*deltaE, E0 + 3*deltaE, step)
  weight = Gaussian(E, E0, deltaE)
  weight_sum = np.sum(weight)
  T_sigma = calculate_sigma(E, t, theta)[-1]
  additional_phase = np.exp(1j*E/hbar/c*np.cos(theta)*t)
  T_sigma_weighted = T_sigma * weight / weight_sum #* additional_phase
  return np.sum(T_sigma_weighted)

def bandwidth_avg_Tpi(E0, deltaE, step, t, theta):
  E = np.arange(E0 - 3*deltaE, E0 + 3*deltaE, step)
  weight = Gaussian(E, E0, deltaE)
  weight_sum = np.sum(weight)
  T_pi = calculate_pi(E, t, theta)[-1]
  additional_phase = np.exp(1j*E/hbar/c*np.cos(theta)*t) ### E0 works, E does not work
  T_pi_weighted = T_pi * weight / weight_sum #* additional_phase
  return np.sum(T_pi_weighted)

def bandwidth_avg_Dsigma(E0, deltaE, step, t, theta):
  E = np.arange(E0 - 3*deltaE, E0 + 3*deltaE, step)
  weight = Gaussian(E, E0, deltaE)
  weight_sum = np.sum(weight)
  D_sigma = calculate_sigma(E, t, theta)[-2]
  additional_phase = np.exp(1j*E/hbar/c*np.cos(theta)*t)
  D_sigma_weighted = D_sigma * weight / weight_sum #* additional_phase
  return np.sum(D_sigma_weighted)

def bandwidth_avg_Dpi(E0, deltaE, step, t, theta):
  E = np.arange(E0 - 3*deltaE, E0 + 3*deltaE, step)
  weight = Gaussian(E, E0, deltaE)
  weight_sum = np.sum(weight)
  D_pi = calculate_pi(E, t, theta)[-2]
  additional_phase = np.exp(1j*E/hbar/c*np.cos(theta)*t) ### E0 works, E does not work
  D_pi_weighted = D_pi * weight / weight_sum #* additional_phase
  return np.sum(D_pi_weighted)