import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

import glob, os, re, subprocess
path = subprocess.run("module load ffmpeg && echo $PATH", shell=True,
                      capture_output=True, text=True,
                      executable="/bin/bash").stdout.strip().splitlines()[-1]
os.environ["PATH"] = path
desiredfontsize=18
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # cmr10 has no minus sign, so tick labels need this
})


def plot_domain_slice(mb_data_var, mb_geometry, n_mbs, y_target=0.0, cmap ='viridis', xmin_global=-10., xmax_global=10., zmin_global=-10., zmax_global=10.):
    fig, ax = plt.subplots()

    for n in range(0, n_mbs):
        # Extents: [xmin, xmax, ymin, ymax, zmin, zmax]
        xmin, xmax, ymin, ymax, zmin, zmax = mb_geometry[n]
        
        # 1. Check if the block intersects the slice plane
        if ymin <= y_target < ymax:
            data = mb_data_var[n]  # 3D array of shape (Nx, Ny, Nz)
            Ny = data.shape[1]
            print(n)
            # 2. Find local index along the Y-axis
            y_coords = np.linspace(ymin, ymax, Ny)
            y_idx = np.argmin(np.abs(y_coords - y_target))
            
            # 3. Extract 2D slice (Nx, Nz)
            slice_2d = data[:, y_idx, :]
            
            # 4. Draw patch using physical spatial bounds
            # Note: transpose (.T) so X maps to horizontal and Z to vertical
            ax.imshow(
                slice_2d, 
                origin='lower', 
                extent=[xmin, xmax, zmin, zmax],
                aspect='equal',
                cmap=cmap
            )

    ax.set_xlabel('X', fontsize=desiredfontsize)
    ax.set_ylabel('Z', fontsize=desiredfontsize)
    ax.set_xlim(xmin_global, xmax_global)
    ax.set_ylim(zmin_global, zmax_global)

    plt.show()


def plot_domain_slice_with_vectors(mb_data_var, mb_data_vector_coord1, mb_data_vector_coord2, mb_geometry, n_mbs, y_target=0.0, cmap ='viridis', xmin_global=-10., xmax_global=10., ymin_global=-10., ymax_global=10., stride = 1, norm=1.e-8, save=False, savename="density.pdf", dpi = 100, with_r=True, vmin=0.5, vmax=10.):
    """x-z slice at y = y_target.  mb_data arrays are ordered (Nz, Ny, Nx)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = None

    for n in range(0, n_mbs):
        # Extents: [xmin, xmax, ymin, ymax, zmin, zmax]
        xmin, xmax, ymin, ymax, zmin, zmax = mb_geometry[n]
        
        # 1. Check if the block intersects the slice plane
        if not (ymin <= y_target < ymax):
            continue

        data = mb_data_var[n]  # 3D array of shape (Nz, Ny, Nx)
        Nz, Ny, Nx = data.shape

        # 2. Find local index along the Y-axis.  mb_geometry holds the block's
        # outer faces, so cell centers sit half a cell inside them --
        # linspace(ymin, ymax, Ny) would sample the faces instead.
        y = ymin + (np.arange(Ny) + 0.5)*(ymax - ymin)/Ny
        y_idx = np.argmin(np.abs(y - y_target))

        # 3. Extract 2D slice (Nz, Nx)
        slice_2d = data[:, y_idx, :]

        vector1_2d = mb_data_vector_coord1[n][:, y_idx, :]
        vector2_2d = mb_data_vector_coord2[n][:, y_idx, :]


        x = xmin + (np.arange(Nx) + 0.5)*(xmax - xmin)/Nx
        z = zmin + (np.arange(Nz) + 0.5)*(zmax - zmin)/Nz
        # 'ij' so both are (Nz, Nx) and line up with slice_2d
        Z, X = np.meshgrid(z, x, indexing='ij')

        # Keep this in a fresh local: rebinding `norm` would make the division
        # compound from one meshblock to the next (block k picking up r^(2k+2)).
        scale = norm/(X**2 + Z**2) if with_r else norm

        # 4. Draw patch using physical spatial bounds.  Rows of slice_2d are z,
        # columns are x, which is what imshow wants given this extent.
        # vmin/vmax go in here -- a trailing set_clim() would only ever reach
        # the last block's image, leaving every other block autoscaled.
        im = ax.imshow(
            slice_2d/scale,
            origin='lower',
            extent=[xmin, xmax, zmin, zmax],
            aspect='equal',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

            

         # 5. Overlay streamlines with subsampling (stride)
        ax.streamplot(
            X[::stride, ::stride],
            Z[::stride, ::stride],
            vector1_2d[::stride, ::stride],
            vector2_2d[::stride, ::stride],
            color='white',
            density=0.5
        )

    ax.set_xlabel('X', fontsize=desiredfontsize)
    ax.set_ylabel('Z', fontsize=desiredfontsize)
    ax.set_xlim(xmin_global, xmax_global)
    ax.set_ylim(ymin_global, ymax_global)

    if im is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #cbar.set_label(r'$\rho/\rho_{\rm floor}$',  fontsize=desiredfontsize)
    if save:
        fig.savefig(savename, dpi = dpi)
    #plt.show()
    #plt.show()


def plot_domain_slice_xy(mb_data_var, mb_geometry, n_mbs, z_target=0.0, cmap ='viridis', xmin_global=-10., xmax_global=10., ymin_global=-10., ymax_global=10.):
    fig, ax = plt.subplots()

    for n in range(0, n_mbs):
        # Extents: [xmin, xmax, ymin, ymax, zmin, zmax]
        xmin, xmax, ymin, ymax, zmin, zmax = mb_geometry[n]
        
        # 1. Check if the block intersects the slice plane
        if zmin <= z_target < zmax:
            data = mb_data_var[n]  # 3D array of shape (Nx, Ny, Nz)
            Nz = data.shape[1]
            print(n)
            # 2. Find local index along the Y-axis
            z_coords = np.linspace(zmin, zmax, Nz)
            z_idx = np.argmin(np.abs(z_coords - z_target))
            
            # 3. Extract 2D slice (Nx, Nz)
            slice_2d = data[:, :, z_idx]
            
            # 4. Draw patch using physical spatial bounds
            # Note: transpose (.T) so X maps to horizontal and Z to vertical
            ax.imshow(
                slice_2d, 
                origin='lower', 
                extent=[xmin, xmax, ymin, ymax],
                aspect='equal',
                cmap=cmap
            )

    ax.set_xlabel('X', fontsize=desiredfontsize)
    ax.set_ylabel('Y', fontsize=desiredfontsize)
    ax.set_xlim(xmin_global, xmax_global)
    ax.set_ylim(ymin_global, ymax_global)

    plt.show()


import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import glob, os, re, subprocess
##                      capture_output=True, text=True,
#                      executable="/bin/bash").stdout.strip().splitlines()[-1]
#os.environ["PATH"] = path
desiredfontsize=18
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # cmr10 has no minus sign, so tick labels need this
})

def plot_domain_slice(mb_data_var, mb_geometry, n_mbs, y_target=0.0, cmap ='viridis', xmin_global=-10., xmax_global=10., zmin_global=-10., zmax_global=10.):
    fig, ax = plt.subplots()

    for n in range(0, n_mbs):
        # Extents: [xmin, xmax, ymin, ymax, zmin, zmax]
        xmin, xmax, ymin, ymax, zmin, zmax = mb_geometry[n]
        
        # 1. Check if the block intersects the slice plane
        if ymin <= y_target < ymax:
            data = mb_data_var[n]  # 3D array of shape (Nx, Ny, Nz)
            Ny = data.shape[1]
            print(n)
            # 2. Find local index along the Y-axis
            y_coords = np.linspace(ymin, ymax, Ny)
            y_idx = np.argmin(np.abs(y_coords - y_target))
            
            # 3. Extract 2D slice (Nx, Nz)
            slice_2d = data[:, y_idx, :]
            
            # 4. Draw patch using physical spatial bounds
            # Note: transpose (.T) so X maps to horizontal and Z to vertical
            ax.imshow(
                slice_2d, 
                origin='lower', 
                extent=[xmin, xmax, zmin, zmax],
                aspect='equal',
                cmap=cmap
            )

    ax.set_xlabel('X', fontsize=desiredfontsize)
    ax.set_ylabel('Z', fontsize=desiredfontsize)
    ax.set_xlim(xmin_global, xmax_global)
    ax.set_ylim(zmin_global, zmax_global)

    plt.show()


def plot_domain_slice_with_vectors(mb_data_var, mb_data_vector_coord1, mb_data_vector_coord2, mb_geometry, n_mbs, y_target=0.0, cmap ='viridis', xmin_global=-10., xmax_global=10., ymin_global=-10., ymax_global=10., stride = 1, norm=1.e-8, save=False, savename="density.pdf", dpi = 100, with_r=True, vmin=0.5, vmax=10.):
    """x-z slice at y = y_target.  mb_data arrays are ordered (Nz, Ny, Nx)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = None

    for n in range(0, n_mbs):
        # Extents: [xmin, xmax, ymin, ymax, zmin, zmax]
        xmin, xmax, ymin, ymax, zmin, zmax = mb_geometry[n]
        
        # 1. Check if the block intersects the slice plane
        if not (ymin <= y_target < ymax):
            continue

        data = mb_data_var[n]  # 3D array of shape (Nz, Ny, Nx)
        Nz, Ny, Nx = data.shape

        # 2. Find local index along the Y-axis.  mb_geometry holds the block's
        # outer faces, so cell centers sit half a cell inside them --
        # linspace(ymin, ymax, Ny) would sample the faces instead.
        y = ymin + (np.arange(Ny) + 0.5)*(ymax - ymin)/Ny
        y_idx = np.argmin(np.abs(y - y_target))

        # 3. Extract 2D slice (Nz, Nx)
        slice_2d = data[:, y_idx, :]

        vector1_2d = mb_data_vector_coord1[n][:, y_idx, :]
        vector2_2d = mb_data_vector_coord2[n][:, y_idx, :]


        x = xmin + (np.arange(Nx) + 0.5)*(xmax - xmin)/Nx
        z = zmin + (np.arange(Nz) + 0.5)*(zmax - zmin)/Nz
        # 'ij' so both are (Nz, Nx) and line up with slice_2d
        Z, X = np.meshgrid(z, x, indexing='ij')

        # Keep this in a fresh local: rebinding `norm` would make the division
        # compound from one meshblock to the next (block k picking up r^(2k+2)).
        scale = norm/(X**2 + Z**2) if with_r else norm

        # 4. Draw patch using physical spatial bounds.  Rows of slice_2d are z,
        # columns are x, which is what imshow wants given this extent.
        # vmin/vmax go in here -- a trailing set_clim() would only ever reach
        # the last block's image, leaving every other block autoscaled.
        im = ax.imshow(
            slice_2d/scale,
            origin='lower',
            extent=[xmin, xmax, zmin, zmax],
            aspect='equal',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

            

         # 5. Overlay streamlines with subsampling (stride)
        ax.streamplot(
            X[::stride, ::stride],
            Z[::stride, ::stride],
            vector1_2d[::stride, ::stride],
            vector2_2d[::stride, ::stride],
            color='white',
            density=0.5
        )

    ax.set_xlabel('X', fontsize=desiredfontsize)
    ax.set_ylabel('Z', fontsize=desiredfontsize)
    ax.set_xlim(xmin_global, xmax_global)
    ax.set_ylim(ymin_global, ymax_global)

    if im is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #cbar.set_label(r'$\rho/\rho_{\rm floor}$',  fontsize=desiredfontsize)
    if save:
        fig.savefig(savename, dpi = dpi)


def plot_domain_slice_xy(mb_data_var, mb_geometry, n_mbs, z_target=0.0, cmap ='viridis', xmin_global=-10., xmax_global=10., ymin_global=-10., ymax_global=10.):
    fig, ax = plt.subplots()

    for n in range(0, n_mbs):
        # Extents: [xmin, xmax, ymin, ymax, zmin, zmax]
        xmin, xmax, ymin, ymax, zmin, zmax = mb_geometry[n]
        
        # 1. Check if the block intersects the slice plane
        if zmin <= z_target < zmax:
            data = mb_data_var[n]  # 3D array of shape (Nx, Ny, Nz)
            Nz = data.shape[1]
            print(n)
            # 2. Find local index along the Y-axis
            z_coords = np.linspace(zmin, zmax, Nz)
            z_idx = np.argmin(np.abs(z_coords - z_target))
            
            # 3. Extract 2D slice (Nx, Nz)
            slice_2d = data[:, :, z_idx]
            
            # 4. Draw patch using physical spatial bounds
            # Note: transpose (.T) so X maps to horizontal and Z to vertical
            ax.imshow(
                slice_2d, 
                origin='lower', 
                extent=[xmin, xmax, ymin, ymax],
                aspect='equal',
                cmap=cmap
            )

    ax.set_xlabel('X', fontsize=desiredfontsize)
    ax.set_ylabel('Y', fontsize=desiredfontsize)
    ax.set_xlim(xmin_global, xmax_global)
    ax.set_ylim(ymin_global, ymax_global)

    plt.show()


def plot_domain_slice_with_vectors_xy(mb_data_var, mb_data_vector_coord1, mb_data_vector_coord2, mb_geometry, n_mbs, z_target=0.0, cmap ='viridis', xmin_global=-10., xmax_global=10., ymin_global=-10., ymax_global=10., stride = 1, norm=1.e-8, save=False, savename="density.pdf", dpi = 100, with_r=True, vmin=0.5, vmax=10.):
    """x-y slice at z = z_target.  mb_data arrays are ordered (Nz, Ny, Nx)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = None

    for n in range(0, n_mbs):
        # Extents: [xmin, xmax, ymin, ymax, zmin, zmax]
        xmin, xmax, ymin, ymax, zmin, zmax = mb_geometry[n]

        # 1. Check if the block intersects the slice plane
        if not (zmin <= z_target < zmax):
            continue

        data = mb_data_var[n]                      # (Nz, Ny, Nx)

        # 2. Cell-center axes, via the same helper check_flux_analytic uses so
        # the two never disagree.  mb_geometry holds the block's outer faces,
        # so linspace(min, max, N) would sample the faces instead.
        Zc, Yc, Xc = mb_cell_centers(mb_geometry, n, data.shape)
        z, y, x = Zc.ravel(), Yc.ravel(), Xc.ravel()

        # 3. Find local index along the z-axis and extract the (Ny, Nx) slice
        z_idx = np.argmin(np.abs(z - z_target))
        slice_2d   = data[z_idx, :, :]
        vector1_2d = mb_data_vector_coord1[n][z_idx, :, :]
        vector2_2d = mb_data_vector_coord2[n][z_idx, :, :]

        # 'ij' so X and Y are both (Ny, Nx) and line up with slice_2d
        Y, X = np.meshgrid(y, x, indexing='ij')

        # 4. Keep this in a fresh local.  Assigning to `norm` here would rebind
        # the function argument, so block k would be divided by the running
        # product of blocks 0..k and pick up r^(2k+2) instead of r^2.
        scale = norm/(X**2 + Y**2) if with_r else norm

        # Rows of slice_2d are y, columns are x, which is what imshow wants
        # given this extent.  vmin/vmax go in here: a trailing set_clim() only
        # ever reaches the last block's image and leaves the rest autoscaled.
        im = ax.imshow(
            slice_2d/scale,
            origin='lower',
            extent=[xmin, xmax, ymin, ymax],
            aspect='equal',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # 5. Overlay streamlines with subsampling (stride)
        ax.streamplot(
            X[::stride, ::stride],
            Y[::stride, ::stride],
            vector1_2d[::stride, ::stride],
            vector2_2d[::stride, ::stride],
            color='white',
            density=0.5
        )

    ax.set_xlabel('X', fontsize=desiredfontsize)
    ax.set_ylabel('Y', fontsize=desiredfontsize)
    ax.set_xlim(xmin_global, xmax_global)
    ax.set_ylim(ymin_global, ymax_global)

    if im is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #cbar.set_label(r'$\rho/\rho_{\rm floor}$', fontsize=desiredfontsize)
    if save:
        fig.savefig(savename, dpi=dpi)
    #plt.show()

# code for computing surface integrals
    # ---- assemble the 8 meshblocks into one global array -------------------------
def stitch_uniform(mb_var, mb_geometry, n_glob=64, lims=(-10., 10.)):
    """mb_var: (n_mbs, nb1, nb2, nb3) ordered (x, y, z).  Returns (nx, ny, nz)."""
    n_mbs =  len(mb_var)
    nb1, nb2, nb3 = mb_var[0].shape

    lo, hi = lims
    h = (hi - lo) / n_glob
    out = np.full((n_glob, n_glob, n_glob), np.nan)
    for n in range(n_mbs):
        xmin, _, ymin, _, zmin, _ = mb_geometry[n]
        i0 = int(round((xmin - lo) / h))
        j0 = int(round((ymin - lo) / h))
        k0 = int(round((zmin - lo) / h))
        out[k0:k0 + nb3,  j0:j0 + nb2, i0:i0 + nb1] = mb_var[n]
    assert not np.isnan(out).any(), "meshblocks did not tile the domain"
    return out, h

def make_interp(mb_var, mb_geometry, n_glob=64, lims=(-10., 10.)):
    arr, h = stitch_uniform(mb_var, mb_geometry, n_glob, lims)
    c = lims[0] + (np.arange(n_glob) + 0.5) * h          # cell centers
    return RegularGridInterpolator((c, c, c), arr,       # axes are (z, y, x)
                                   bounds_error=True)

# ---- solid-angle quadrature: exact for spherical harmonics up to the order ---
def sphere_quad(n_theta=64, n_phi=128):
    mu, w_mu = np.polynomial.legendre.leggauss(n_theta)  # weights for ∫dmu
    theta = np.arccos(mu)
    phi = 2 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    W = np.broadcast_to(w_mu[:, None], TH.shape) * (2 * np.pi / n_phi)
    return TH, PH, np.ascontiguousarray(W) # W.sum() == 4*pi


# ---- the surface integral ---------------------------------------------------
def flux_through_sphere(fx, fy, fz, r, n_theta=64, n_phi=128):
    """fx, fy, fz: RegularGridInterpolators from make_interp.
    Returns total flux, hemisphere fluxes, and dF/dOmega on the (theta, phi) grid."""
    TH, PH, W = sphere_quad(n_theta, n_phi)
    st, ct = np.sin(TH), np.cos(TH)
    x, y, z = r * st * np.cos(PH), r * st * np.sin(PH), r * ct
    pts = np.stack([z, y, x], axis=-1)                   # match (x, y, z) order

    Fr = (fx(pts) * x + fy(pts) * y + fz(pts) * z) / r   # radial component
    dFdO = r**2 * Fr

    north = np.sum(np.where(TH < np.pi / 2, dFdO, 0.0) * W)
    south = np.sum(np.where(TH > np.pi / 2, dFdO, 0.0) * W)
    return north + south, north, south, TH, PH, dFdO, W


def calculate_momentum_flux(rho, egas, ux, uy, uz, bx, by, bz, gamma_ad = 4.0/3.0):
    u_sqr = ux**2+uy**2+uz**2

    P = egas*(gamma_ad-1.0)
    Gamma = np.sqrt(1.0+(u_sqr))
    sigma0=10.0
    B_sqr = bx**2+ by**2+bz**2
    vx=ux/Gamma
    vy=uy/Gamma
    vz=uz/Gamma
    v_wind=np.sqrt(u_sqr)/Gamma
    v_dot_B = vx*bx+vy*by+bz*vz
    #print(np.max(Gamma))
    #print(v_wind*(B_sqr)*(1.0)/(rho*Gamma**2*v_wind))
    f_x =(rho + gamma_ad/(gamma_ad-1.0)*P)*Gamma**2*vx - bx*(v_dot_B) + vx*(B_sqr)
    f_y =(rho + gamma_ad/(gamma_ad-1.0)*P)*Gamma**2*vy - by*(v_dot_B) + vy*(B_sqr)
    f_z =(rho + gamma_ad/(gamma_ad-1.0)*P)*Gamma**2*vz - bz*(v_dot_B) + vz*(B_sqr)
    
    return f_x, f_y, f_z

def calculate_sigma(rho, egas, ux, uy, uz, bx, by, bz, gamma_ad = 4.0/3.0):
    u_sqr = ux**2+uy**2+uz**2
    
    P = egas*(gamma_ad-1.0)
    Gamma = np.sqrt(1.0+(u_sqr))
    
    B_sqr = bx**2+ by**2+bz**2
    vx=ux/Gamma
    vy=uy/Gamma
    vz=uz/Gamma
    v_wind=np.sqrt(u_sqr)/Gamma
    v_dot_B = vx*bx+vy*by+bz*vz
    sigma = B_sqr/(rho*Gamma**2)
    return sigma

def calculate_momentum_flux_model(x, y, z, sigma0=10.0, rho_surf=1.0, Gamma_wind = 5.0, r_star = 3.0, theta0=0.1):
    v_wind = np.sqrt(1.0-1.0/Gamma_wind**2)
    B0=np.sqrt(rho_surf*sigma0*Gamma_wind**2) 
    r = np.sqrt(x**2+y**2+z**2)
    sintheta = np.sqrt(x**2+y**2)/r

    f_r = (1.0+sigma0)/sigma0*v_wind*B0**2*(r_star/r)**2*(sintheta**2+theta0)
    return f_r, 0.0, 0.0

def calculate_momentum_flux_model_rtheta(theta, r, sigma0=10.0, rho_surf=1.0, Gamma_wind = 5.0, r_star = 3.0, theta0=0.0):
    v_wind = np.sqrt(1.0-1.0/Gamma_wind**2)
    B0=np.sqrt(rho_surf*sigma0*Gamma_wind**2) 

    sintheta = np.sin(theta)
    
    f_r = (1.0+sigma0)/sigma0*v_wind*B0**2*(r_star/r)**2*(sintheta**2+theta0)
    return f_r, 0.0, 0.0

def calculate_sigma_model(theta, r, sigma0=10.0, chi=0.0):
    """sigma0 everywhere except the band |theta - pi/2| < chi, where it drops
    to 4 sigma0 arcsin^2(cot(theta) cot(chi))/pi^2.  Accepts scalar or array
    theta; returns the same shape."""
    theta = np.asarray(theta, dtype=float)
    sigma = np.full(theta.shape, sigma0, dtype=float)
    band = np.abs(theta - np.pi/2.0) < chi
    if chi > 0 and band.any():
        th = theta[band]
        arg = np.clip(np.cos(th)/np.sin(th)/np.tan(chi), -1.0, 1.0)
        sigma[band] = 4.0*sigma0*np.arcsin(arg)**2/np.pi**2
    return sigma if sigma.ndim else float(sigma)

def convert_spherical(vec_x, vec_y, vec_z, x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r_cyl = np.sqrt(x**2 + y**2 ) 

    costheta = z/r
    sintheta = r_cyl/r
    
    cosphi = x/r_cyl
    sinphi = y/r_cyl

    vec_r = vec_x*x/r+vec_y*y/r+ vec_z*z/r
    vec_theta = vec_x*costheta*cosphi +vec_y*costheta*sinphi-vec_z*sintheta
    vec_phi =  -vec_x*sinphi + vec_y*cosphi

    return vec_r, vec_theta, vec_phi

def get_stats(x):
    return np.mean(x), np.median(x), np.std(x), np.max(x), np.min(x)

def print_stats(stats, quantity_name):
    print(quantity_name+" mean error: {}".format(stats[0]))
    print(quantity_name+" median error: {}".format(stats[1]))
    print(quantity_name+" standard deviation of error: {}".format(stats[2]))
    print(quantity_name+" max error: {}".format(stats[3]))
    print(quantity_name+" min error: {}".format(stats[4]))

def compute_B_r(bx, by, bz, x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    B_r = bx*x/r +by*y/r+bz*z/r
    return B_r

def compute_B_theta(bx, by, bz, x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r_cyl = np.sqrt(x**2 + y**2 ) 
    costheta = z/r
    sintheta = r_cyl/r

    cosphi = x/r_cyl
    sinphi = y/r_cyl

    B_theta = bx*costheta*cosphi +by*costheta*sinphi-bz*sintheta
    return B_theta
    
def compute_B_phi(bx, by, bz, x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r_cyl = np.sqrt(x**2 + y**2 ) 
    costheta = z/r
    sintheta = r_cyl/r

    cosphi = x/r_cyl
    sinphi = y/r_cyl

    B_phi = -bx*sinphi + by*cosphi
    return B_phi

def mb_cell_centers(mb_geometry, n, shape):
    """Cell-center coords for meshblock n, ordered to match mb_data's (Nz, Ny, Nx).

    mb_geometry holds the block's outer faces, so centers are offset by half a
    cell -- np.linspace(xmin, xmax, Nx) would sample the faces instead, which
    lands points exactly on x=0/y=0 (block boundaries) and makes r_cyl vanish.
    sparse=True returns (Nz,1,1)/(1,Ny,1)/(1,1,Nx) views that broadcast for free.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = mb_geometry[n]
    Nz, Ny, Nx = shape
    x = xmin + (np.arange(Nx) + 0.5)*(xmax - xmin)/Nx
    y = ymin + (np.arange(Ny) + 0.5)*(ymax - ymin)/Ny
    z = zmin + (np.arange(Nz) + 0.5)*(zmax - zmin)/Nz
    return np.meshgrid(z, y, x, indexing='ij', sparse=True)

def check_flux_analytic(rho, egas, vx, vy, vz, bx, by, bz, n_mbs, mb_geometry, r_check=None, r_cyl_check=0.0, verbose=True,
                     r_star=3.0, gamma_wind=5.0, sigma0=10.0, rho_surf=1.0, theta0 = 0.0, gamma_ad=4.0/3.0):   

    """Compare simulated flux against the analytic expected flux.

    Errors are normalized by the monopole amplitude   e_* therefore reads as "error as a fraction
    of the local field strength".

    Cells failing the mask are dropped, not overwritten with the model value --
    filling them in would make their error exactly zero and dilute the stats.
    r_cyl_check exists because the spherical basis is singular on the whole
    z-axis (r_cyl -> 0), which a cut on R alone does not screen.

    r_check defaults to r_star: inside the star the pgen resets the field every
    step and the face/center stencils are inconsistent, so that region is not a
    meaningful test of the seeding.

    Returns dict of concatenated (all-block) normalized errors.
    """
    if r_check is None:
        r_check = r_star

    all_e = {"r": [], "theta": [], "phi": []}

    for n in range(0, n_mbs):
        Z, Y, X = mb_cell_centers(mb_geometry, n, rho[n].shape)

        R = np.sqrt(X**2 + Y**2 + Z**2)
        R_cyl = np.sqrt(X**2 + Y**2) + np.zeros_like(R)   # broadcast to full shape
        keep = (R > r_check) & (R_cyl > r_cyl_check)
        if not keep.any():
            continue
        f_r_model, f_theta_model, f_phi_model = calculate_momentum_flux_model(X, Y, Z, sigma0=sigma0, rho_surf=rho_surf, Gamma_wind = gamma_wind, r_star = r_star, theta0=theta0)

        f_x, f_y, f_z = calculate_momentum_flux(rho[n], egas[n], vx[n], vy[n], vz[n], bx[n], by[n], bz[n], gamma_ad = gamma_ad)

        f_r_sim, f_theta_sim, f_phi_sim = convert_spherical(f_x, f_y, f_z, X, Y, Z)

        e_r     = (np.abs(f_r_sim     - f_r_model    )/f_r_model)[keep]
        e_theta = (np.abs(f_theta_sim - f_theta_model)/f_r_model)[keep]
        e_phi   = (np.abs(f_phi_sim   - f_phi_model  )/f_r_model)[keep]
        #print(keep)
        #X_keep = X[keep]
        #Y_keep = Y[keep]
        #Z_keep = Z[keep]
        R_keep = R[keep]
        all_e["r"].append(e_r)
        all_e["theta"].append(e_theta)
        all_e["phi"].append(e_phi)

        if verbose:
            print("--- block {} ({} of {} cells kept) ---".format(n, keep.sum(), R.size))
            print_stats(get_stats(e_r),     "f_r")
            print_stats(get_stats(e_theta), "f_theta")
            print_stats(get_stats(e_phi),   "f_phi")

            print("Coords of largest deviation f_r is ({})".format(R_keep[e_r==np.max(e_r)]))

    all_e = {k: np.concatenate(v) for k, v in all_e.items()}
    print("=== all blocks, {} cells ===".format(all_e["r"].size))
    for k in ("r", "theta", "phi"):
        print_stats(get_stats(all_e[k]), "f_" + k)
    return all_e



def pngs_to_mp4(png_dir, name, out_name=None, out_file=None, fps=15, crf=18):
    """Stitch {png_dir}/{name}_{N}.png into an mp4, ordered by N."""
    pat = re.compile(re.escape(name) + r"(\d+)\.png$")
    frames = sorted(
        (f for f in glob.glob(os.path.join(png_dir, f"{name}*.png")) if pat.search(f)),
        key=lambda f: int(pat.search(f).group(1)),
    )
    if not frames:
        raise FileNotFoundError(f"No frames matching {name}*.png in {png_dir}")
    if out_name is None:
        out_name = name
    if out_file is None:
        out_file = os.path.join(os.path.dirname(os.path.normpath(png_dir)), f"{out_name}.mp4")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "image2pipe", "-framerate", str(fps), "-i", "-",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # libx264 needs even dimensions
        "-c:v", "libx264", "-pix_fmt", "yuv420p",      # yuv420p plays in QuickTime/browsers/PowerPoint
        "-crf", str(crf), "-preset", "slow",
        out_file,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        with open(f, "rb") as fh:
            proc.stdin.write(fh.read())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"Wrote {out_file} ({len(frames)} frames @ {fps} fps)")
    return out_file

# ---- quantities vs polar angle on spherical shells --------------------------
def shell_average_vs_theta(mb_q, mb_geometry, radii, n_theta=32, dr=None):
    """Phi-averaged profile <q>(theta) on thin spherical shells, computed
    straight from the meshblocks.

    mb_q: per-meshblock arrays ordered (Nz, Ny, Nx), e.g.
    filedata["mb_data"]["dens"].  A cell counts toward shell i if its center
    lies within dr/2 of radii[i].  Inside a shell, cells are binned by theta
    and averaged with cell-volume weights, so blocks at different refinement
    levels combine correctly (no global stitching, unlike stitch_uniform).

    dr defaults to the largest cell width among the blocks: a thinner shell
    leaves gaps where cells straddle it.

    Returns theta bin centers (n_theta,), <q> (n_r, n_theta), and the cell
    count per bin (n_r, n_theta).  Empty bins are NaN -- raise dr or lower
    n_theta if that happens near the poles, where the bins hold fewest cells.
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    geo = np.asarray(mb_geometry)
    n_mbs = len(mb_q)

    if dr is None:
        Nz, Ny, Nx = np.shape(mb_q[0])
        dr = np.max([(geo[:, 1] - geo[:, 0])/Nx,
                     (geo[:, 3] - geo[:, 2])/Ny,
                     (geo[:, 5] - geo[:, 4])/Nz])

    edges = np.linspace(0.0, np.pi, n_theta + 1)
    qsum  = np.zeros((radii.size, n_theta))
    vsum  = np.zeros((radii.size, n_theta))
    count = np.zeros((radii.size, n_theta), dtype=int)

    for n in range(n_mbs):
        q = np.asarray(mb_q[n])
        Z, Y, X = mb_cell_centers(mb_geometry, n, q.shape)
        R  = np.sqrt(X**2 + Y**2 + Z**2)
        TH = np.arccos(np.clip(Z/R, -1.0, 1.0))

        # skip blocks that no shell passes through
        if R.max() < radii.min() - 0.5*dr or R.min() > radii.max() + 0.5*dr:
            continue

        xmin, xmax, ymin, ymax, zmin, zmax = geo[n]
        dV = (xmax - xmin)*(ymax - ymin)*(zmax - zmin)/q.size

        for i, r in enumerate(radii):
            sel = np.abs(R - r) < 0.5*dr
            if not sel.any():
                continue
            th = TH[sel]
            qsum[i]  += np.histogram(th, bins=edges, weights=q[sel]*dV)[0]
            vsum[i]  += np.histogram(th, bins=edges)[0]*dV
            count[i] += np.histogram(th, bins=edges)[0]

    with np.errstate(invalid="ignore", divide="ignore"):
        q_avg = np.where(vsum > 0, qsum/vsum, np.nan)
    theta = 0.5*(edges[1:] + edges[:-1])
    return theta, q_avg, count


def energy_flux_vs_theta(rho, egas, ux, uy, uz, bx, by, bz, mb_geometry, radii,
                         n_theta=32, dr=None, gamma_ad=4.0/3.0):
    """dL/dOmega(theta) = r^2 <F_r>_phi on shells at each radius in radii.

    F is the lab-frame energy flux from calculate_momentum_flux (enthalpy +
    Poynting, rest mass included).  ux, uy, uz are the spatial 4-velocity
    components (velx/vely/velz in the SR bin output).

    Also returns L(r) = 2 pi * integral of dL/dOmega sin(theta) dtheta.  For a
    steady wind it should be the same at every radius, which makes it a quick
    sanity check.
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))

    f_r = []
    for n in range(len(rho)):
        Z, Y, X = mb_cell_centers(mb_geometry, n, np.shape(rho[n]))
        fx, fy, fz = calculate_momentum_flux(rho[n], egas[n], ux[n], uy[n], uz[n],
                                             bx[n], by[n], bz[n], gamma_ad=gamma_ad)
        R = np.sqrt(X**2 + Y**2 + Z**2)
        f_r.append((fx*X + fy*Y + fz*Z)/R)

    theta, Fr_avg, count = shell_average_vs_theta(f_r, mb_geometry, radii,
                                                  n_theta=n_theta, dr=dr)
    dLdO = radii[:, None]**2 * Fr_avg
    dtheta = np.pi/n_theta
    L = 2*np.pi*np.nansum(dLdO*np.sin(theta)*dtheta, axis=1)
    return theta, dLdO, L, count

def sigma_vs_theta(rho, egas, ux, uy, uz, bx, by, bz, mb_geometry, radii,
                         n_theta=32, dr=None, gamma_ad=4.0/3.0):
    """dL/dOmega(theta) = r^2 <F_r>_phi on shells at each radius in radii.

    F is the lab-frame energy flux from calculate_momentum_flux (enthalpy +
    Poynting, rest mass included).  ux, uy, uz are the spatial 4-velocity
    components (velx/vely/velz in the SR bin output).

    Also returns L(r) = 2 pi * integral of dL/dOmega sin(theta) dtheta.  For a
    steady wind it should be the same at every radius, which makes it a quick
    sanity check.
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))

    sigma = []
    for n in range(len(rho)):
        Z, Y, X = mb_cell_centers(mb_geometry, n, np.shape(rho[n]))
        sigma0 = calculate_sigma(rho[n], egas[n], ux[n], uy[n], uz[n],
                                             bx[n], by[n], bz[n], gamma_ad=gamma_ad)
        R = np.sqrt(X**2 + Y**2 + Z**2)
        sigma.append(sigma0)

    theta, sigma_avg, count = shell_average_vs_theta(sigma, mb_geometry, radii,
                                                  n_theta=n_theta, dr=dr)
    
    return theta, sigma_avg, count 

def rho_vs_theta(rho, egas, ux, uy, uz, bx, by, bz, mb_geometry, radii,
                         n_theta=32, dr=None, gamma_ad=4.0/3.0):
    """r^2 rho = r^2 rho on shells at each radius in radii.

    F is the lab-frame energy flux from calculate_momentum_flux (enthalpy +
    Poynting, rest mass included).  ux, uy, uz are the spatial 4-velocity
    components (velx/vely/velz in the SR bin output).

    Also returns L(r) = 2 pi * integral of dL/dOmega sin(theta) dtheta.  For a
    steady wind it should be the same at every radius, which makes it a quick
    sanity check.
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))


    theta, rho_avg, count = shell_average_vs_theta(rho, mb_geometry, radii,
                                                  n_theta=n_theta, dr=dr)
    rho_r_sqr = rho_avg*radii[:, None]**2
    return theta, rho_r_sqr, count


def plot_vs_theta(theta, profiles, radii, ylabel="", logy=False,
                  save=False, savename="vs_theta.pdf", dpi=100, ylim=None, model=None, t=None,
                  t_loc=(0.05, 0.95)):
    """One curve per radius; profiles is (n_r, n_theta)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(radii))) 
    for r, prof, c in zip(np.atleast_1d(radii), profiles, colors):
        ax.plot(theta, prof, color=c, label=r"$r = {:g}$".format(r))
        if model is not None:
            model_num = model(theta, r)
            ax.plot(theta, model_num, color=c, linestyle = "--")

    if t is not None:
        label = t if isinstance(t, str) else r"$t = {:.2f}$".format(t)
        ax.text(t_loc[0], t_loc[1], label, transform=ax.transAxes,
                ha="left", va="top", fontsize=desiredfontsize,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        
    ax.set_xlabel(r"$\theta$", fontsize=desiredfontsize)
    ax.set_ylabel(ylabel, fontsize=desiredfontsize)
    ax.set_xlim(0, np.pi)
    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    


    if logy:
        ax.set_yscale("log")

    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    if save:
        fig.savefig(savename, dpi=dpi)
    return fig, ax


    #plt.show()

# code for computing surface integrals
    # ---- assemble the 8 meshblocks into one global array -------------------------
def stitch_uniform(mb_var, mb_geometry, n_glob=64, lims=(-10., 10.)):
    """mb_var: (n_mbs, nb1, nb2, nb3) ordered (x, y, z).  Returns (nx, ny, nz)."""
    n_mbs =  len(mb_var)
    nb1, nb2, nb3 = mb_var[0].shape

    lo, hi = lims
    h = (hi - lo) / n_glob
    out = np.full((n_glob, n_glob, n_glob), np.nan)
    for n in range(n_mbs):
        xmin, _, ymin, _, zmin, _ = mb_geometry[n]
        i0 = int(round((xmin - lo) / h))
        j0 = int(round((ymin - lo) / h))
        k0 = int(round((zmin - lo) / h))
        out[k0:k0 + nb3,  j0:j0 + nb2, i0:i0 + nb1] = mb_var[n]
    assert not np.isnan(out).any(), "meshblocks did not tile the domain"
    return out, h

def make_interp(mb_var, mb_geometry, n_glob=64, lims=(-10., 10.)):
    arr, h = stitch_uniform(mb_var, mb_geometry, n_glob, lims)
    c = lims[0] + (np.arange(n_glob) + 0.5) * h          # cell centers
    return RegularGridInterpolator((c, c, c), arr,       # axes are (z, y, x)
                                   bounds_error=True)

# ---- solid-angle quadrature: exact for spherical harmonics up to the order ---
def sphere_quad(n_theta=64, n_phi=128):
    mu, w_mu = np.polynomial.legendre.leggauss(n_theta)  # weights for ∫dmu
    theta = np.arccos(mu)
    phi = 2 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    W = np.broadcast_to(w_mu[:, None], TH.shape) * (2 * np.pi / n_phi)
    return TH, PH, np.ascontiguousarray(W) # W.sum() == 4*pi


# ---- the surface integral ---------------------------------------------------
def flux_through_sphere(fx, fy, fz, r, n_theta=64, n_phi=128):
    """fx, fy, fz: RegularGridInterpolators from make_interp.
    Returns total flux, hemisphere fluxes, and dF/dOmega on the (theta, phi) grid."""
    TH, PH, W = sphere_quad(n_theta, n_phi)
    st, ct = np.sin(TH), np.cos(TH)
    x, y, z = r * st * np.cos(PH), r * st * np.sin(PH), r * ct
    pts = np.stack([z, y, x], axis=-1)                   # match (x, y, z) order

    Fr = (fx(pts) * x + fy(pts) * y + fz(pts) * z) / r   # radial component
    dFdO = r**2 * Fr

    north = np.sum(np.where(TH < np.pi / 2, dFdO, 0.0) * W)
    south = np.sum(np.where(TH > np.pi / 2, dFdO, 0.0) * W)
    return north + south, north, south, TH, PH, dFdO, W


def calculate_momentum_flux(rho, egas, ux, uy, uz, bx, by, bz, gamma_ad = 4.0/3.0):
    u_sqr = ux**2+uy**2+uz**2

    P = egas*(gamma_ad-1.0)
    Gamma = np.sqrt(1.0+(u_sqr))
    sigma0=10.0
    B_sqr = bx**2+ by**2+bz**2
    vx=ux/Gamma
    vy=uy/Gamma
    vz=uz/Gamma
    v_wind=np.sqrt(u_sqr)/Gamma
    v_dot_B = vx*bx+vy*by+bz*vz
    #print(np.max(Gamma))
    #print(v_wind*(B_sqr)*(1.0)/(rho*Gamma**2*v_wind))
    f_x =(rho + gamma_ad/(gamma_ad-1.0)*P)*Gamma**2*vx - bx*(v_dot_B) + vx*(B_sqr)
    f_y =(rho + gamma_ad/(gamma_ad-1.0)*P)*Gamma**2*vy - by*(v_dot_B) + vy*(B_sqr)
    f_z =(rho + gamma_ad/(gamma_ad-1.0)*P)*Gamma**2*vz - bz*(v_dot_B) + vz*(B_sqr)
    
    return f_x, f_y, f_z

def calculate_momentum_flux_model(x, y, z, sigma0=10.0, rho_surf=1.0, Gamma_wind = 5.0, r_star = 3.0, theta0=0.1):
    v_wind = np.sqrt(1.0-1.0/Gamma_wind**2)
    B0=np.sqrt(rho_surf*sigma0*Gamma_wind**2) 
    r = np.sqrt(x**2+y**2+z**2)
    sintheta = np.sqrt(x**2+y**2)/r

    f_r = (1.0+sigma0)/sigma0*v_wind*B0**2*(r_star/r)**2*(sintheta**2+theta0)
    return f_r, 0.0, 0.0

def convert_spherical(vec_x, vec_y, vec_z, x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r_cyl = np.sqrt(x**2 + y**2 ) 

    costheta = z/r
    sintheta = r_cyl/r
    
    cosphi = x/r_cyl
    sinphi = y/r_cyl

    vec_r = vec_x*x/r+vec_y*y/r+ vec_z*z/r
    vec_theta = vec_x*costheta*cosphi +vec_y*costheta*sinphi-vec_z*sintheta
    vec_phi =  -vec_x*sinphi + vec_y*cosphi

    return vec_r, vec_theta, vec_phi

def get_stats(x):
    return np.mean(x), np.median(x), np.std(x), np.max(x), np.min(x)

def print_stats(stats, quantity_name):
    print(quantity_name+" mean error: {}".format(stats[0]))
    print(quantity_name+" median error: {}".format(stats[1]))
    print(quantity_name+" standard deviation of error: {}".format(stats[2]))
    print(quantity_name+" max error: {}".format(stats[3]))
    print(quantity_name+" min error: {}".format(stats[4]))

def compute_B_r(bx, by, bz, x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    B_r = bx*x/r +by*y/r+bz*z/r
    return B_r

def compute_B_theta(bx, by, bz, x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r_cyl = np.sqrt(x**2 + y**2 ) 
    costheta = z/r
    sintheta = r_cyl/r

    cosphi = x/r_cyl
    sinphi = y/r_cyl

    B_theta = bx*costheta*cosphi +by*costheta*sinphi-bz*sintheta
    return B_theta
    
def compute_B_phi(bx, by, bz, x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r_cyl = np.sqrt(x**2 + y**2 ) 
    costheta = z/r
    sintheta = r_cyl/r

    cosphi = x/r_cyl
    sinphi = y/r_cyl

    B_phi = -bx*sinphi + by*cosphi
    return B_phi

def mb_cell_centers(mb_geometry, n, shape):
    """Cell-center coords for meshblock n, ordered to match mb_data's (Nz, Ny, Nx).

    mb_geometry holds the block's outer faces, so centers are offset by half a
    cell -- np.linspace(xmin, xmax, Nx) would sample the faces instead, which
    lands points exactly on x=0/y=0 (block boundaries) and makes r_cyl vanish.
    sparse=True returns (Nz,1,1)/(1,Ny,1)/(1,1,Nx) views that broadcast for free.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = mb_geometry[n]
    Nz, Ny, Nx = shape
    x = xmin + (np.arange(Nx) + 0.5)*(xmax - xmin)/Nx
    y = ymin + (np.arange(Ny) + 0.5)*(ymax - ymin)/Ny
    z = zmin + (np.arange(Nz) + 0.5)*(zmax - zmin)/Nz
    return np.meshgrid(z, y, x, indexing='ij', sparse=True)

def check_flux_analytic(rho, egas, vx, vy, vz, bx, by, bz, n_mbs, mb_geometry, r_check=None, r_cyl_check=0.0, verbose=True,
                     r_star=3.0, gamma_wind=5.0, sigma0=10.0, rho_surf=1.0, theta0 = 0.0, gamma_ad=4.0/3.0):   

    """Compare simulated flux against the analytic expected flux.

    Errors are normalized by the monopole amplitude   e_* therefore reads as "error as a fraction
    of the local field strength".

    Cells failing the mask are dropped, not overwritten with the model value --
    filling them in would make their error exactly zero and dilute the stats.
    r_cyl_check exists because the spherical basis is singular on the whole
    z-axis (r_cyl -> 0), which a cut on R alone does not screen.

    r_check defaults to r_star: inside the star the pgen resets the field every
    step and the face/center stencils are inconsistent, so that region is not a
    meaningful test of the seeding.

    Returns dict of concatenated (all-block) normalized errors.
    """
    if r_check is None:
        r_check = r_star

    all_e = {"r": [], "theta": [], "phi": []}

    for n in range(0, n_mbs):
        Z, Y, X = mb_cell_centers(mb_geometry, n, rho[n].shape)

        R = np.sqrt(X**2 + Y**2 + Z**2)
        R_cyl = np.sqrt(X**2 + Y**2) + np.zeros_like(R)   # broadcast to full shape
        keep = (R > r_check) & (R_cyl > r_cyl_check)
        if not keep.any():
            continue
        f_r_model, f_theta_model, f_phi_model = calculate_momentum_flux_model(X, Y, Z, sigma0=sigma0, rho_surf=rho_surf, Gamma_wind = gamma_wind, r_star = r_star, theta0=theta0)

        f_x, f_y, f_z = calculate_momentum_flux(rho[n], egas[n], vx[n], vy[n], vz[n], bx[n], by[n], bz[n], gamma_ad = gamma_ad)

        f_r_sim, f_theta_sim, f_phi_sim = convert_spherical(f_x, f_y, f_z, X, Y, Z)

        e_r     = (np.abs(f_r_sim     - f_r_model    )/f_r_model)[keep]
        e_theta = (np.abs(f_theta_sim - f_theta_model)/f_r_model)[keep]
        e_phi   = (np.abs(f_phi_sim   - f_phi_model  )/f_r_model)[keep]
        #print(keep)
        #X_keep = X[keep]
        #Y_keep = Y[keep]
        #Z_keep = Z[keep]
        R_keep = R[keep]
        all_e["r"].append(e_r)
        all_e["theta"].append(e_theta)
        all_e["phi"].append(e_phi)

        if verbose:
            print("--- block {} ({} of {} cells kept) ---".format(n, keep.sum(), R.size))
            print_stats(get_stats(e_r),     "f_r")
            print_stats(get_stats(e_theta), "f_theta")
            print_stats(get_stats(e_phi),   "f_phi")

            print("Coords of largest deviation f_r is ({})".format(R_keep[e_r==np.max(e_r)]))

    all_e = {k: np.concatenate(v) for k, v in all_e.items()}
    print("=== all blocks, {} cells ===".format(all_e["r"].size))
    for k in ("r", "theta", "phi"):
        print_stats(get_stats(all_e[k]), "f_" + k)
    return all_e       