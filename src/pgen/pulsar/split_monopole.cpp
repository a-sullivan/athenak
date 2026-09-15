//========================================================================================
// AthenaK pgen: Pgen for split monopole pulsar wind (Cartesian SR MHD)
//
// Magnetic field:
//   Split-monopole with equatorial current sheet of with delta: 
//   A = A_0 (r_star/r)((1-cos(theta))/sin(theta)) theta < pi/2-delta/2
//   A = -2A_0/delta (r_star/r)(1-((theta-pi/2)cos(theta)+sin(pi/2-delta/2)+delta/2)/sin(theta)) 
//                pi/2+delta/2 < theta < pi/2+delta/2
//   A = A_0 (r_star/r)((1+cos(theta))/sin(theta)) theta > pi/2+delta/2
//
//   Properties:
//     Equatorial current sheet with width delta
//     Outside -delta/2 < theta-pi/2 <delta/2, 
//          constant B with positive sign in northern hemisphere and negative sign in south 
//     
//
// 
//
// Parameters (<problem>):
//   x0, y0, z0      center (default 0)
//   r_star           neutron star radius (default 1.0)
//   Omega           inner solid-body angular velocity (default 0.0)
//  sigma0           initial  magnetization (default 10)
// theta0           regularization parameter for poles to ensure no numerical issues
//   R_corot         corotation radius; 0 = solid-body everywhere (default 0)
//   t_ramp          spin-up timescale; 0 = instant (default 0)
//   rho0            base density (default 1.0)
//   p0              base pressure (default 1e-4)

//
// Notes:
//   - Primitives in SR MHD: w(IVX/IVY/IVZ) = 4-velocity u^i = gamma*v^i.
//   - w(IEN) = internal energy density rho*eps = P/(gamma_ad - 1).
//   - u(IEN) = E - D  (AthenaK evolves total energy minus rest-mass density).
//   - All device kernels capture host params as scalars (no namespace static reads).
//   - CT seeding uses midpoint-rule A evaluations (second-order face fields).
//   - Parameters are read on both fresh start and restart so user_srcs_func works.
//========================================================================================

#include <cmath>
#include <cstdlib>
#include <iostream>
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"

namespace pw {
// ------ Parameters ------- //
struct Params {
    //Geometry
    Real x0, y0, z0; // center of the simulation domain

    // Pulsar 
    Real r_star; 
    Real Omega;
    Real t_ramp;

    // Magnetosphere
    Real A0;
    Real delta;
    Real sigma0;
    Real beta0;
    Real rho_surf;
    Real B0;

    //Regularization
    Real theta0;
    Real frac_star;
    Real r_interior;
    Real epsilon;
    Real theta_int;

    //track floors
    Real gamma_max;
    Real sigma_max;
    Real beta_min;
    Real dfloor;
    Real pfloor;

    // SNR (currently not implemented)
    Real rho0;
    Real p0;  

};
static Params P;

KOKKOS_INLINE_FUNCTION
Real F_radial(Real r, Real A0, Real r_star, Real r_i) {
  return (r > r_i) ? A0*r_star/(r*r)
                   : (A0*r_star/(r_i*r_i))*(2.0 - (r*r)/(r_i*r_i));
}



KOKKOS_INLINE_FUNCTION
void A_vec_split_monopole(Real x, Real y, Real z,
           Real x0, Real y0, Real z0, 
           Real theta0, Real delta, Real A0,  Real r_star, Real r_interior, Real epsilon, Real theta_int,
           Real &Ax, Real &Ay, Real &Az){
    const Real x_r = x - x0, y_r = y - y0, z_r = z - z0;

    const Real r_min = epsilon*r_interior;
    const Real r = sqrt(fmax(x_r*x_r+y_r*y_r+z_r*z_r, r_min*r_min));


    //const Real r_cyl = sqrt(x_r*x_r+y_r*y_r);  // cylindrical radius
    const Real costheta = fmax(-1.0, fmin(1.0, z_r/r));
    const Real sintheta = sqrt(fmax(1.0-costheta*costheta, 0.0)); // works for the branch I am in since theta < pi

    // Assumes vector potential is of the form 
    //  A_x = - F(r) g(theta) y_r
    //  A_y = F(r) g(theta) z_r
    // A_coeff = F(r) g(theta)
    

    // Now let me define the radial shape F(r)
    // outside r > r_interior, the radial component has the expected split monopole form
    // inside r < r_interior, the radial component is then regularized to be quadratic in r so there is no divergence at origin
    const Real F = F_radial(r, A0, r_star, r_interior);
    
    // Now let me define the polar shape g(theta)
    // first define some helper variables
    const Real sin_d2 = sin(0.5*delta);
    const Real cos_d2 = cos(0.5*delta);
    Real g;
    if (fabs(costheta)>sin_d2){
        g = 1.0/(1.0+fabs(costheta));
    } else {
        const Real K = cos_d2 +0.5*delta;
        g = (2.0/delta) * (K-asin(costheta)*costheta-sintheta)/(sintheta*sintheta);
    }
    
    // make the interior region flush as you get deeper into the star interior
    if (r < r_interior) {
        const Real t = r/r_interior;
        const Real lam = t*t*(3.0-2.0*t);
        g = lam * g + (1.0 - lam) * theta_int; // theta_int is a constant 
    }

    const Real A_coeff = F * g;
    Ax = -A_coeff* y_r;
    Ay = A_coeff* x_r;
    Az = 0.0;
    }

// Discrete face-centered curl (uniform Cartesian)
KOKKOS_INLINE_FUNCTION
Real Bx_from_A(Real Az_yP, Real Az_yM, Real Ay_zP, Real Ay_zM, Real dy, Real dz) {
  return (Az_yP - Az_yM)/dy - (Ay_zP - Ay_zM)/dz;
}
KOKKOS_INLINE_FUNCTION
Real By_from_A(Real Ax_zP, Real Ax_zM, Real Az_xP, Real Az_xM, Real dz, Real dx) {
  return (Ax_zP - Ax_zM)/dz - (Az_xP - Az_xM)/dx;
}
KOKKOS_INLINE_FUNCTION
Real Bz_from_A(Real Ay_xP, Real Ay_xM, Real Ax_yP, Real Ax_yM, Real dx, Real dy) {
  return (Ay_xP - Ay_xM)/dx - (Ax_yP - Ax_yM)/dy;
}

void Source(Mesh *pm, const Real dt) {
  auto *pack = pm->pmb_pack;
  if (!pack || !pack->pmhd) return;

  auto &w        = pack->pmhd->w0;
  auto &u        = pack->pmhd->u0;
  auto &size_src = pack->pmb->mb_size;
  auto &bf   = pack->pmhd->b0;
  auto &bcc0 = pack->pmhd->bcc0;
  auto &size = pack->pmb->mb_size; 

  const auto &ind = pm->mb_indcs;
  const int is=ind.is, ie=ind.ie, js=ind.js, je=ind.je, ks=ind.ks, ke=ind.ke;
  const int nx1=ind.nx1, nx2=ind.nx2, nx3=ind.nx3;

  // Capture all params as device scalars
  const Real x0        = P.x0, y0 = P.y0, z0 = P.z0;
  const Real A0         = P.A0;
  const Real r_star     = P.r_star;
  const Real r_interior = P.r_interior;
  const Real epsilon = P.epsilon;
  const Real rho0       = P.rho0;
  const Real p0         = P.p0;
  const Real delta      = P.delta;
  const Real theta0     = P.theta0;
  const Real B0 = P.B0;
  const Real sigma0 = P.sigma0;
  const Real beta0 = P.beta0;
  const Real gamma_ad = pack->pmhd->peos->eos_data.gamma;
  const Real e_0 = p0 / (gamma_ad - 1.0);

  const Real sigma_max = P.sigma_max;
  const Real beta_min = P.beta_min;
  const Real dfloor_original = P.dfloor;
  const Real pfloor_original = P.pfloor;

  const Real Om        = P.Omega;
  const Real t_ramp    = P.t_ramp;
  const Real t_cur     = pm->time;
  const Real gamma_max = P.gamma_max;

  const Real s = (t_ramp > 0.0) ? fmin((t_cur/t_ramp), 1.0) : 1.0; 
  const Real Om_t = Om * s*s*(3.0-2.0*s);

  par_for("monopole_reset_src", DevExeSpace(),
          0, pack->nmb_thispack-1, ks,ke, js,je, is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            const auto sz = size.d_view(m);
            Real xc = CellCenterX(i - is, nx1, sz.x1min, sz.x1max);
            Real yc = CellCenterX(j - js, nx2, sz.x2min, sz.x2max);
            Real zc = CellCenterX(k - ks, nx3, sz.x3min, sz.x3max);

            Real dx = xc - x0, dy = yc - y0, dz = zc -z0;

            Real rmin = epsilon*r_interior;
            Real r = sqrt(fmax(dx*dx + dy*dy + dz*dz, rmin*rmin));
            // assign SNR properties
            if(r <= r_star){
                
                // assign the floor based on the sigma floor, not on the actual density floor
                // B0 is normalized by the chosen surface density rho_surf and sigma0 
                Real B_sqr = SQR(pw::F_radial(r, A0, r_star, r_interior));
                Real dfloor = fmax(dfloor_original, B_sqr/sigma_max);
                Real pfloor = fmax(pfloor_original, B_sqr/2*beta_min);

                Real d_set = fmax(dfloor, B_sqr/sigma0);
                Real p_set = fmax(pfloor, B_sqr/2*beta0);

                Real dens = d_set;
                Real pgas = p_set;
                Real egas = pgas/(gamma_ad-1.0);
                
                // add in the rigid rotation of the star
                
                const Real r_cyl_sq = dx*dx+dy*dy;
                const Real u0_rot = 1.0/sqrt(fmax(1.0-r_cyl_sq*Om_t*Om_t, 1.0/(gamma_max*gamma_max))); // lorentz factor associated with rotation

                w(m,IDN,k,j,i) = dens;
                w(m,IVX,k,j,i) = -Om_t*dy*u0_rot;
                w(m,IVY,k,j,i) = Om_t*dx*u0_rot;
                w(m,IVZ,k,j,i) = 0.0;
                w(m,IEN,k,j,i) = egas;

            }
            
  }); 
  

        // ---- Reset cell-centered B and conserved vars, INSIDE THE STAR ONLY ----
        // Must run after the three face-field loops.  Cannot use the array-wide
        // PrimToCons(): u0 already carries this stage's flux divergence (see the note
        // on MHD::MHDSrcTerms), so an ungated call would discard the update over the
        // whole block and freeze the fluid everywhere.
        par_for("monopole_reset_cons", DevExeSpace(),
                0, pack->nmb_thispack-1, ks,ke, js,je, is,ie,
            KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            const auto sz = size.d_view(m);
            Real xc = CellCenterX(i - is, nx1, sz.x1min, sz.x1max);
            Real yc = CellCenterX(j - js, nx2, sz.x2min, sz.x2max);
            Real zc = CellCenterX(k - ks, nx3, sz.x3min, sz.x3max);

            Real dx = xc - x0, dy = yc - y0, dz = zc - z0;
            Real r = sqrt(dx*dx + dy*dy + dz*dz);
            if (r > r_star) return;          // leave the exterior entirely to the solver

            // cell-centered B from the just-reset face fields
            //Real bx = 0.5*(bf.x1f(m,k,j,i) + bf.x1f(m,k,j,i+1));
            //Real by = 0.5*(bf.x2f(m,k,j,i) + bf.x2f(m,k,j+1,i));
            //Real bz = 0.5*(bf.x3f(m,k,j,i) + bf.x3f(m,k+1,j,i));
            //bcc0(m,IBX,k,j,i) = bx;
            //bcc0(m,IBY,k,j,i) = by;
            //bcc0(m,IBZ,k,j,i) = bz;

            // read back the primitives the fluid loop just pinned, so there is a single
            // source of truth for the interior state
            MHDPrim1D w_cell;
            w_cell.d  = w(m,IDN,k,j,i);
            w_cell.vx = w(m,IVX,k,j,i);
            w_cell.vy = w(m,IVY,k,j,i);
            w_cell.vz = w(m,IVZ,k,j,i);
            w_cell.e  = w(m,IEN,k,j,i);
            w_cell.bx = bcc0(m,IBX,k,j,i);
            w_cell.by = bcc0(m,IBY,k,j,i);
            w_cell.bz = bcc0(m,IBZ,k,j,i);

            HydCons1D u_cell;
            SingleP2C_IdealSRMHD(w_cell, gamma_ad, u_cell);

            u(m,IDN,k,j,i) = u_cell.d;
            u(m,IM1,k,j,i) = u_cell.mx;
            u(m,IM2,k,j,i) = u_cell.my;
            u(m,IM3,k,j,i) = u_cell.mz;
            u(m,IEN,k,j,i) = u_cell.e;   // SR: E - D, handled inside SingleP2C
        });

  
    }

    void EfieldMask(Mesh *pm){
        auto *pack = pm->pmb_pack;
        if (!pack || !pack->pmhd) return;

        const auto &ind = pm->mb_indcs;
        const int is=ind.is, ie=ind.ie, js=ind.js, je=ind.je, ks=ind.ks, ke=ind.ke;
        const int nx1=ind.nx1, nx2=ind.nx2, nx3=ind.nx3;

        auto &size = pack->pmb->mb_size;
        auto e1 = pack->pmhd->efld.x1e;
        auto e2 = pack->pmhd->efld.x2e;
        auto e3 = pack->pmhd->efld.x3e;
        auto &b0 = pack->pmhd->b0;

        const Real x0=P.x0, y0=P.y0, z0=P.z0, r_star=P.r_star;
        const Real Om        = P.Omega;
        const Real t_ramp    = P.t_ramp;
        const Real t_cur     = pm->time;
        const Real s = (t_ramp > 0.0) ? fmin((t_cur/t_ramp), 1.0) : 1.0; 
        const Real Om_t = Om * s*s*(3.0-2.0*s);

        par_for("emf_e1", DevExeSpace(), 0,pack->nmb_thispack-1, ks,ke+1, js,je+1, is,ie,
            KOKKOS_LAMBDA(int m, int k, int j, int i) {
                const auto sz = size.d_view(m);
                const Real dx = CellCenterX(i-is, nx1, sz.x1min, sz.x1max) - x0;
                const Real dy = LeftEdgeX  (j-js, nx2, sz.x2min, sz.x2max) - y0;
                const Real dz = LeftEdgeX  (k-ks, nx3, sz.x3min, sz.x3max) - z0;
                if (sqrt(dx*dx + dy*dy + dz*dz) > r_star) return;
                const Real bz = 0.5*(b0.x3f(m,k,j-1,i) + b0.x3f(m,k,j,i));
                e1(m,k,j,i) = -Om_t*dx*bz;
        });

        par_for("emf_e2", DevExeSpace(), 0,pack->nmb_thispack-1, ks,ke+1, js,je+1, is,ie,
            KOKKOS_LAMBDA(int m, int k, int j, int i) {
                const auto sz = size.d_view(m);
                const Real dx = CellCenterX(i-is, nx1, sz.x1min, sz.x1max) - x0;
                const Real dy = LeftEdgeX  (j-js, nx2, sz.x2min, sz.x2max) - y0;
                const Real dz = LeftEdgeX  (k-ks, nx3, sz.x3min, sz.x3max) - z0;
                if (sqrt(dx*dx + dy*dy + dz*dz) > r_star) return;
                const Real bz = 0.5*(b0.x3f(m,k,j,i-1) + b0.x3f(m,k,j,i));
                e2(m,k,j,i) = -Om_t*dy*bz;
        });

        par_for("emf_e3", DevExeSpace(), 0,pack->nmb_thispack-1, ks,ke+1, js,je+1, is,ie,
            KOKKOS_LAMBDA(int m, int k, int j, int i) {
                const auto sz = size.d_view(m);
                const Real dx = CellCenterX(i-is, nx1, sz.x1min, sz.x1max) - x0;
                const Real dy = LeftEdgeX  (j-js, nx2, sz.x2min, sz.x2max) - y0;
                const Real dz = LeftEdgeX  (k-ks, nx3, sz.x3min, sz.x3max) - z0;
                if (sqrt(dx*dx + dy*dy + dz*dz) > r_star) return;
                const Real bx = 0.5*(b0.x1f(m,k,j-1,i) + b0.x2f(m,k,j,i));
                const Real by = 0.5*(b0.x2f(m,k,j,i-1) + b0.x2f(m,k,j,i));
                e3(m,k,j,i) = Om_t*(dy*by+dx*bx);
        });

    }





} //namespace pw

// define the problem generator
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart){

    user_srcs = pin->GetOrAddBoolean("problem", "user_srcs", true);
    user_esrcs = pin->GetOrAddBoolean("problem","user_esrcs",false); 

    // read the parameters from inside the input file

    // neutron star paremeters
    pw::P.r_star = pin->GetOrAddReal("problem", "r_star", 10.0);
    pw::P.Omega = pin->GetOrAddReal("problem", "Omega", 0.0);
    pw::P.t_ramp = pin->GetOrAddReal("problem", "t_ramp", 20.0);

    if (pw::P.Omega*pw::P.r_star >= 1.0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "Omega*r_star = " << pw::P.Omega*pw::P.r_star
            << " >= 1: light cylinder is inside the stellar surface" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // magnetosphere
    pw::P.delta = pin->GetOrAddReal("problem", "delta", 1.0);
    pw::P.sigma0 = pin->GetOrAddReal("problem", "sigma0", 10.0);
    pw::P.beta0      = pin->GetOrAddReal("problem", "beta0",  1.0e-03);
    pw::P.rho_surf = pin->GetOrAddReal("problem", "rho_surf", 1.0); // set the density at the surface of neutron star
    pw::P.A0 = pw::P.r_star*sqrt(pw::P.rho_surf*pw::P.sigma0);
    pw::P.B0 = sqrt(pw::P.rho_surf*pw::P.sigma0);

    //Regularization
    pw::P.theta0 = pin->GetOrAddReal("problem", "theta0", 0.1);
    pw::P.frac_star = pin->GetOrAddReal("problem", "frac_star", 0.5);
    pw::P.epsilon = pin->GetOrAddReal("problem", "epsilon", 1e-8);
    pw::P.theta_int = pin->GetOrAddReal("problem", "theta_int", 0.5);
    pw::P.r_interior = pw::P.frac_star * pw::P.r_star;

    //geometry
    pw::P.x0 = pin->GetOrAddReal("problem", "x0", 0.0);
    pw::P.y0 = pin->GetOrAddReal("problem", "y0", 0.0);
    pw::P.z0 = pin->GetOrAddReal("problem", "z0", 0.0);

    // mhd floors
    pw::P.dfloor        = pin->GetOrAddReal("mhd", "dfloor",    1.0e-08);
    pw::P.pfloor        = pin->GetOrAddReal("mhd", "pfloor",    1.0e-15);
    pw::P.beta_min      = pin->GetOrAddReal("mhd", "beta_min",  1.0e-05);
    pw::P.sigma_max     = pin->GetOrAddReal("mhd", "sigma_max", 1000.0);
    pw::P.gamma_max     = pin->GetOrAddReal("mhd", "gamma_max", 1000.0);

    // SNR parameters 
    // right now these are not needed, will become relevant when SNR is created
    pw::P.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    pw::P.p0 = pin->GetOrAddReal("problem", "p0", 1.e-4);

    

    auto *pmbp = pmy_mesh_->pmb_pack;
    if (!pmbp || !pmbp->pmhd) return; //i.e. do not go any further if class variables are not defined

    // may want to use rhd first when setting this up, we shall see

    if (!restart){
        auto &w0 = pmbp->pmhd->w0;
        auto &u0 = pmbp->pmhd->u0;
        auto &bf = pmbp->pmhd->b0; // fields defined at the faces
        auto &bcc0 = pmbp->pmhd->bcc0; // fields at cell centers
        auto &size = pmbp->pmb->mb_size; 

        const auto &ind = pmy_mesh_->mb_indcs;
        const int is = ind.is, ie=ind.ie, js=ind.js, je=ind.je, ks=ind.ks, ke=ind.ke;
        const int nx1=ind.nx1, nx2 = ind.nx2, nx3 = ind.nx3;

        const Real gamma_ad = pmbp->pmhd->peos->eos_data.gamma;

        const Real x0 = pw::P.x0, y0 = pw::P.y0, z0 = pw::P.z0;
        const Real rho0 = pw::P.rho0; 
        const Real p0 = pw::P.p0;
        const Real e_0 = p0 / (gamma_ad - 1.0);

        const Real r_star = pw::P.r_star;
        
        
        

        const Real theta0 = pw::P.theta0;
        const Real sigma0 = pw::P.sigma0;
        const Real beta0 = pw::P.beta0;
        const Real A0 = pw::P.A0;
        const Real delta = pw::P.delta;
        const Real B0 = pw::P.B0;
        const Real rho_surf = pw::P.rho_surf;
        const Real r_interior = pw::P.r_interior;
        const Real epsilon = pw::P.epsilon;
        const Real theta_int = pw::P.theta_int;

        const Real sigma_max = pw::P.sigma_max;
        const Real beta_min = pw::P.beta_min;
        const Real dfloor_original = pw::P.dfloor;
        const Real pfloor_original = pw::P.pfloor;
        
        

        
        
        // First do loop for fluid quantities
        par_for("pgen_fluid",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
        KOKKOS_LAMBDA(int m,int k,int j,int i) {
            const auto sz = size.d_view(m);
            Real xc = CellCenterX(i - is, nx1, sz.x1min, sz.x1max);
            Real yc = CellCenterX(j - js, nx2, sz.x2min, sz.x2max);
            Real zc = CellCenterX(k - ks, nx3, sz.x3min, sz.x3max);

            Real dx = xc - x0, dy = yc - y0, dz = zc -z0;
            Real rmin = epsilon*r_interior;
            Real r = sqrt(fmax(dx*dx + dy*dy + dz*dz, rmin*rmin));
            // assign SNR properties

            // assign the floor based on the sigma floor, not on the actual density floor
            // B0 is normalized by the chosen surface density rho_surf and sigma0 
            //Real emag_star = SQR(B0*r_star*r_star);
            Real B_sqr = SQR(pw::F_radial(r, A0, r_star, r_interior));
            Real dfloor = fmax(dfloor_original, B_sqr/sigma_max);
            Real pfloor = fmax(pfloor_original, B_sqr/2*beta_min);

            Real d_set = fmax(dfloor, B_sqr/sigma0);
            Real p_set = fmax(pfloor, B_sqr/2*beta0);

            Real dens = d_set;
            Real pgas = p_set;
            Real egas = pgas/(gamma_ad-1.0);


            w0(m,IDN,k,j,i) = dens;
            w0(m,IVX,k,j,i) = 0.0;
            w0(m,IVY,k,j,i) = 0.0;
            w0(m,IVZ,k,j,i) = 0.0;
            w0(m,IEN,k,j,i) = egas;


        
        });
        // Now do a set of loops for the magnetic field quantities
        // start with x fields
        par_for("pgen_b_x1", DevExeSpace(), 0,(pmbp->nmb_thispack-1), ks,ke, js,je, is,(ie+1),
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int ifc) {
            const auto sz = size.d_view(m);
            const auto dx1 = (sz.x1max-sz.x1min)/nx1;
            const auto dx2 = (sz.x2max-sz.x2min)/nx2;
            const auto dx3 = (sz.x3max-sz.x3min)/nx3;

            const Real xf = sz.x1min + (ifc - is)*dx1; // x faces
            const Real yc = sz.x2min + ((j - js) +0.5)*dx2; // y centers
            const Real zc = sz.x3min + ((k - ks) +0.5)*dx3; // z centers

            Real dx = xf - x0, dy = yc - y0, dz = zc -z0;

            Real r = sqrt(dx*dx + dy*dy + dz*dz);

            if(r >= r_star){
            }
            Real Ax, Ay, Az, Ay_zP, Ay_zM, Az_yP, Az_yM;

            pw::A_vec_split_monopole(xf, yc + 0.5*dx2, zc, x0, y0, z0, theta0, delta, A0, r_star, r_interior, epsilon, theta_int, Ax, Ay, Az); Az_yP = Az;
            pw::A_vec_split_monopole(xf, yc - 0.5*dx2, zc, x0, y0, z0, theta0, delta, A0, r_star, r_interior, epsilon, theta_int, Ax, Ay, Az); Az_yM = Az;
            pw::A_vec_split_monopole(xf, yc, zc + 0.5*dx3, x0, y0, z0, theta0, delta, A0, r_star, r_interior, epsilon, theta_int, Ax, Ay, Az); Ay_zP = Ay;
            pw::A_vec_split_monopole(xf, yc, zc - 0.5*dx3, x0, y0, z0, theta0, delta, A0, r_star, r_interior, epsilon, theta_int, Ax, Ay, Az); Ay_zM = Ay;

            bf.x1f(m, k, j, ifc) = pw::Bx_from_A(Az_yP, Az_yM, Ay_zP, Ay_zM, dx2, dx3);
               
   

        });

        // now y 
        par_for("pgen_b_x2", DevExeSpace(), 0,(pmbp->nmb_thispack-1), ks,ke, js,(je+1), is,ie,
        KOKKOS_LAMBDA(const int m, const int k, const int jfc, const int i) {
            const auto sz = size.d_view(m);
            const auto dx1 = (sz.x1max-sz.x1min)/nx1;
            const auto dx2 = (sz.x2max-sz.x2min)/nx2;
            const auto dx3 = (sz.x3max-sz.x3min)/nx3;

            const Real xc = sz.x1min + ((i - is) +0.5)*dx1; // x centers
            const Real yf = sz.x2min + (jfc - js)*dx2; // y centers
            const Real zc = sz.x3min + ((k - ks) +0.5)*dx3; // z centers

            Real dx = xc - x0, dy = yf - y0, dz = zc -z0;

            Real r = sqrt(dx*dx + dy*dy + dz*dz);

            if(r > r_star){
            }

            Real Ax, Ay, Az, Ax_zP, Ax_zM, Az_xP, Az_xM;

            pw::A_vec_split_monopole(xc + 0.5*dx1, yf, zc, x0, y0, z0, theta0, delta, A0, r_star, r_interior, epsilon, theta_int, Ax, Ay, Az); Az_xP = Az;
            pw::A_vec_split_monopole(xc - 0.5*dx1, yf, zc, x0, y0, z0, theta0, delta, A0, r_star, r_interior, epsilon, theta_int, Ax, Ay, Az); Az_xM = Az;
            pw::A_vec_split_monopole(xc, yf, zc + 0.5*dx3, x0, y0, z0, theta0, delta, A0, r_star, r_interior, epsilon, theta_int, Ax, Ay, Az); Ax_zP = Ax;
            pw::A_vec_split_monopole(xc, yf, zc - 0.5*dx3, x0, y0, z0, theta0, delta, A0, r_star, r_interior, epsilon, theta_int, Ax, Ay, Az); Ax_zM = Ax;

            bf.x2f(m, k, jfc, i) = pw::By_from_A(Ax_zP, Ax_zM, Az_xP, Az_xM, dx3, dx1);
            
               
            
        

        });

        // now z 
        par_for("pgen_b_x3", DevExeSpace(), 0,(pmbp->nmb_thispack-1), ks,(ke+1), js,je, is,ie,
        KOKKOS_LAMBDA(const int m, const int kfc, const int j, const int i) {
            const auto sz = size.d_view(m);
            const auto dx1 = (sz.x1max-sz.x1min)/nx1;
            const auto dx2 = (sz.x2max-sz.x2min)/nx2;
            const auto dx3 = (sz.x3max-sz.x3min)/nx3;

            const Real xc = sz.x1min + ((i - is) +0.5)*dx1; // x centers
            const Real yc = sz.x2min + ((j - js) +0.5)*dx2; // y centers
            const Real zf = sz.x3min + ((kfc - ks))*dx3; // z centers

            Real dx = xc - x0, dy = yc - y0, dz = zf -z0;

            Real r = sqrt(dx*dx + dy*dy + dz*dz);



            Real Ax, Ay, Az, Ax_yP, Ax_yM, Ay_xP, Ay_xM;

            pw::A_vec_split_monopole(xc + 0.5*dx1, yc, zf, x0, y0, z0, theta0, delta, A0, r_star, r_interior, epsilon, theta_int, Ax, Ay, Az); Ay_xP = Ay;
            pw::A_vec_split_monopole(xc - 0.5*dx1, yc, zf, x0, y0, z0, theta0, delta, A0, r_star, r_interior, epsilon, theta_int, Ax, Ay, Az); Ay_xM = Ay;
            pw::A_vec_split_monopole(xc, yc + 0.5*dx2, zf , x0, y0, z0, theta0, delta, A0, r_star, r_interior, epsilon, theta_int, Ax, Ay, Az); Ax_yP = Ax;
            pw::A_vec_split_monopole(xc, yc - 0.5*dx2, zf , x0, y0, z0, theta0, delta, A0, r_star, r_interior, epsilon, theta_int, Ax, Ay, Az); Ax_yM = Ax;

            bf.x3f(m, kfc, j, i)  = pw::Bz_from_A(Ay_xP, Ay_xM, Ax_yP, Ax_yM, dx1, dx2);
               
            
     

        });

        // ---- Cell-centered B (average of faces) ----
        par_for("bcc", DevExeSpace(), 0,(pmbp->nmb_thispack-1), ks,ke, js,je, is,ie,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            bcc0(m,IBX,k,j,i) = 0.5*(bf.x1f(m,k,j,i) + bf.x1f(m,k,j,i+1));
            bcc0(m,IBY,k,j,i) = 0.5*(bf.x2f(m,k,j,i) + bf.x2f(m,k,j+1,i));
            bcc0(m,IBZ,k,j,i) = 0.5*(bf.x3f(m,k,j,i) + bf.x3f(m,k+1,j,i));
        });

        // now let us convert the ptimitives to conservative variables
        pmbp->pmhd->peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);


    }

    if (user_srcs) user_srcs_func = &pw::Source;

    if (user_esrcs) user_esrcs_func = &pw::EfieldMask;
} 