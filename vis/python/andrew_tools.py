import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
desiredfontsize=18


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