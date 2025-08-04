import numpy as np
import scipy.constants as sp_const
from scipy.interpolate import RegularGridInterpolator
import scipy.special as sp_func
import scipy.signal as sp_sign

def Roh(X):
    chi_h = -0.79955e-05 + 1j*0.24361e-06
    chi_mh = chi_h
    chi_0 = -0.15127e-04 + 1j*0.34955e-06
    theta_B = (np.pi/180) * 14.221
    
    t = X.t - (X.tmax/2)
    
    Tg = (2 * np.sin(theta_B)**2) / ( (X.hwKalpha1N/X.hbar) * np.sqrt(chi_h*chi_mh))
    exparg = - ( (X.hwKalpha1N/X.hbar) * np.imag(chi_0) * t) / (2 * np.sin(theta_B)**2)
    
    # plt.plot(np.real((sp_func.jv(1, t/Tg) / (1j*t)) * np.exp(exparg)))
    # plt.plot(np.imag((sp_func.jv(1, t/Tg) / (1j*t)) * np.exp(exparg)))
    # plt.show()
    
    return np.heaviside(t, 1) * (sp_func.jv(1, t/Tg) / (1j*t)) * np.exp(exparg)

def Ocelot_SASE_seed_220_dbm_pstxy(X):
    SASE = RadiationField()  # initialize RadiationField object
    
    seed_sigma_rx = X.config['seed_width_FWHM_x'] / (2 * np.sqrt(2*np.log(2)))
    seed_sigma_ry = X.config['seed_width_FWHM_y'] / (2 * np.sqrt(2*np.log(2)))
    seed_sigma_t  = X.config['seed_duration_FWHM_t'] / (2 * np.sqrt(2*np.log(2)))
    
    # The transverse domain is considered to be space [m], since I hace input
    # parameters in time [fs], the correct conversion is needed
    
    kwargs={'xlamds':1e-9*X.lambdaKalpha1N,                     # [m] - central wavelength
            'shape':(X.xgrid, X.ygrid, X.tgrid),            # size of field matrix (x,y,z=ct) (number of points)
            'dgrid':(2e-9*X.xmax, 2e-9*X.ymax, 1e-15*X.tmax*sp_const.c),                # size of field grid (max value) 
            'power_rms':(1e-9*seed_sigma_rx, 1e-9*seed_sigma_ry, 1e-15*seed_sigma_t*sp_const.c),   # rms size of radiation distribution
            'power_center':(0,0,None),                      # (x,y,z) [m] - position of the radiation distribution
            'power_angle':(0,0),                            # (x,y) [rad] - angle of further radiation propagation
            'power_waistpos':(0,0),                         # (Z_x,Z_y) [m] downstrean location of the waist of the beam
            'wavelength':None,                              # central frequency of the radiation, if different from xlamds
            'zsep':None,                                    # distance between slices in z as zsep*xlamds
            'freq_chirp':0,                                 # dw/dt=[1/fs**2] - requency chirp of the beam around power_center[2]
            'en_pulse':X.E_seed_uJ*1e-6,        # total energy or max power of the pulse, use only one
            'power':None,
            'rho':X.seed_FEL_bandwidth/2
            }
    
    SASE = imitate_sase_dfl(**kwargs);
    
    field_txy = SASE.fld  
        
    # plt.figure()
    # plt.imshow(np.abs(field_txy[0,:,:])**2)
    # plt.figure()
    # plt.plot(np.linspace(-SASE.Lz()/sp_const.c, SASE.Lz()/sp_const.c, X.tgrid), np.abs(field_txy[:,1,1])**2)
    # plt.show()
    
    # Effect of mono: convolution between field_t and Roh
    field_temp_t0 = np.einsum('txy -> t', field_txy) # extract t domain
    field_temp_t0 = roll_zeropad(field_temp_t0, int(X.seed_delay/X.dt))
    
    field_temp_xy = np.einsum('txy -> xy', field_txy)
    
    field_temp_t1 = sp_sign.convolve(field_temp_t0, Roh(X), mode='same')
    field_temp_t2 = sp_sign.convolve(field_temp_t1, Roh(X), mode='same')
    
    field_txy = np.einsum('t, xy -> txy', field_temp_t2, field_temp_xy)
    
    # plt.plot(np.abs(1e12*Roh(X))**2)
    # plt.show()
    # plt.plot(np.abs(field_temp_t0)**2)
    # plt.show()
    # plt.plot(np.abs(0.1*field_temp_t1)**2)
    # plt.show()
    # plt.plot(np.abs(0.01*field_temp_t2)**2)
    # plt.show()
    # plt.plot(np.abs(field_txy[:,0,0])**2)
    # plt.show()  
       
    
    # Normalization: it has to be compatible with the units used in the rest of
    # the code. imitate_sase_dfl uses SI units [m, s, J], need to normalize with
    # respect to [nm, fs, eV]
    
    # dx = 1e9 * (SASE.Lx() / (SASE.Nx()-1))
    # dy = 1e9 * (SASE.Ly() / (SASE.Ny()-1))
    # dt = 1e15 * (SASE.Lz() / (SASE.Nz()-1)) / sp_const.c
    
    norm = np.sqrt(np.sum(np.abs(field_txy)**2 * X.dx * X.dy * X.dt))
    
    field_txy = field_txy / norm
    
    # plt.figure()
    # plt.imshow(np.abs(field_txy[0,:,:])**2)
    # plt.figure()
    # #plt.plot(np.linspace(-SASE.Lz()/sp_const.c*1.e15, SASE.Lz()*1.e15/sp_const.c, X.tgrid), np.abs(field_txy[:,1,1])**2)
    # plt.plot(np.linspace(0, SASE.Lz()*1.e15/sp_const.c, X.tgrid), np.abs(field_txy[:,1,1])**2)
    # plt.xlabel('t (fs)')
    # plt.show()
    
    N_seed_photons = X.E_seed_uJ * 1e-6 / (X.hwKalpha1N * sp_const.e)
    print('number of seed photons = ' + f"{N_seed_photons:.1e}")
    
    SASE_for_seed = np.sqrt(N_seed_photons) * field_txy
    
    Omega_seed_pstxy = np.zeros((2, 2, X.tgrid, X.xgrid, X.ygrid), dtype=complex)
    fluxfield2Rabi = np.sqrt((3.0 * X.lambdaKalpha1N**2 * X.Gamma_sp_fsm1N / 8.0 / np.pi))
    Omega_seed_pstxy[0,1,:,:,:] = fluxfield2Rabi * SASE_for_seed # in linear polarization basis, along y-axis
    Omega_seed_pstxy[1,:,:,:,:] = np.conj(Omega_seed_pstxy[0,:,:,:,:])
        
    return Omega_seed_pstxy