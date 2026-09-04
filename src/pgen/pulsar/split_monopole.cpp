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
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
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
    Real rho0;
    Real p0; 

    // Magnetosphere
    Real A0;
    Real delta;
    Real sigma0;

    //Regularization
    Real theta0;
    Real frac_star;
    Real r_interior;

   

};
static Params P;




KOKKOS_INLINE_FUNCTION
void A_vec_split_monopole(Real x, Real y, Real z,
           Real x0, Real y0, Real z0, 
           Real theta0, Real delta, Real A0, Real r_star, Real r_interior,
           Real &Ax, Real &Ay, Real &Az){
    const Real x_r = x - x0, y_r = y - y0, z_r = z - z0;
    const Real r = sqrt(x_r*x_r+y_r*y_r+z_r*z_r);
    const Real r_cyl = sqrt(x_r*x_r+y_r*y_r);  // cylindrical radius
    const Real costheta = z_r/r;
    const Real sintheta = sqrt(y_r*y_r+x_r*x_r)/r;
    const Real theta = acos(costheta);

   
    if (theta < (M_PI-delta)/2.0){
        Real A_phi = A0*(r_star/fmax(r, r_interior))/(fmax(r+fabs(z_r), r_interior));
        Ax = -A_phi*y_r;
        Ay =  A_phi*x_r;
    } 
    else if (theta >= (M_PI-delta)/2.0 && theta < (M_PI+delta)/2.0 ){
        Real A_phi = -2*A0/delta *(r_star/(fmax(r, r_interior)))*(1.0-((theta-M_PI/2.0)*costheta+sin((M_PI-delta)/2.0)+delta/2.0)/sintheta);
        Ax = -A_phi*y_r/r_cyl;
        Ay =  A_phi*x_r/r_cyl;
    } 
    else if (theta >= (M_PI+delta)/2.0){
        Real A_phi = A0*(r_star/fmax(r, r_interior))/(fmax(r+fabs(z_r), r_interior));
        Ax = -A_phi*y_r;
        Ay =  A_phi*x_r;
    }

    Az = 0.0;
}

namespace pw {

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

} //namespace pw

// define the problem generator
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart){

    // read the parameters from inside the input file

    // neutron star paremeters
    pw::P.r_star = pin->GetOrAddReal("problem", "r_star", 10.0);
    pw::P.Omega = pin->GetOrAddReal("problem", "Omega", 0.0);
    pw::P.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    pw::P.p0 = pin->GetOrAddReal("problem", "p0", 1.e-4);

    // magnetosphere
    pw::P.delta = pin->GetOrAddReal("problem", "delta", 1.0);
    pw::P.sigma0 = pin->GetOrAddReal("problem", "sigma0", 10.0);
    pw::P.A0 = pw::P.r_star*sqrt(pw::P.rho0*pw::P.sigma0);

    //Regularization
    pw::P.theta0 = pin->GetOrAddReal("problem", "theta0", 0.1);
    pw::P.frac_star = pin->GetOrAddReal("problem", "frac_star", 0.5);
    pw::P.r_interior = pw::P.frac_star * pw::P.r_star;

    //geometry
    pw::P.x0 = pin->GetOrAddReal("problem", "x0", 0.0);
    pw::P.y0 = pin->GetOrAddReal("problem", "y0", 0.0);
    pw::P.z0 = pin->GetOrAddReal("problem", "z0", 0.0);

    

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
        const Real r_star = pw::P.r_star;
        
        const Real e_0 = p0 / (gamma_ad - 1.0);

        const Real theta0 = pw::P.theta0;
        const Real sigma0 = pw::P.sigma0;
        const Real A0 = pw::P.A0;
        const Real delta = pw::P.delta;
        const Real r_interior = pw::P.r_interior;

        
        
        // First do loop for fluid quantities
        par_for("pgen_fluid",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
        KOKKOS_LAMBDA(int m,int k,int j,int i) {
            const auto sz = size.d_view(m);
            Real xc = CellCenterX(i - is, nx1, sz.x1min, sz.x1max);
            Real yc = CellCenterX(j - js, nx2, sz.x2min, sz.x2max);
            Real zc = CellCenterX(k - ks, nx3, sz.x3min, sz.x3max);

            Real dx = xc - x0, dy = yc - y0, dz = zc -z0;

            Real r = sqrt(dx*dx + dy*dy + dz*dz);
            // assign SNR properties
            if(r >= r_star){

            }
            w0(m,IDN,k,j,i) = rho0;
            w0(m,IVX,k,j,i) = 0.0;
            w0(m,IVY,k,j,i) = 0.0;
            w0(m,IVZ,k,j,i) = 0.0;
            w0(m,IEN,k,j,i) = e_0;


        
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

            pw::A_vec_split_monopole(xf, yc + 0.5*dx2, zc, x0, y0, z0, theta0, delta, A0, r_star, r_interior, Ax, Ay, Az); Az_yP = Az;
            pw::A_vec_split_monopole(xf, yc - 0.5*dx2, zc, x0, y0, z0, theta0, delta, A0, r_star, r_interior, Ax, Ay, Az); Az_yM = Az;
            pw::A_vec_split_monopole(xf, yc, zc + 0.5*dx3, x0, y0, z0, theta0, delta, A0, r_star, r_interior, Ax, Ay, Az); Ay_zP = Ay;
            pw::A_vec_split_monopole(xf, yc, zc - 0.5*dx3, x0, y0, z0, theta0, delta, A0, r_star, r_interior, Ax, Ay, Az); Ay_zM = Ay;

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

            pw::A_vec_split_monopole(xc + 0.5*dx1, yf, zc, x0, y0, z0, theta0, delta, A0, r_star, r_interior, Ax, Ay, Az); Az_xP = Az;
            pw::A_vec_split_monopole(xc - 0.5*dx1, yf, zc, x0, y0, z0, theta0, delta, A0, r_star, r_interior, Ax, Ay, Az); Az_xM = Az;
            pw::A_vec_split_monopole(xc, yf, zc + 0.5*dx3, x0, y0, z0, theta0, delta, A0, r_star, r_interior, Ax, Ay, Az); Ax_zP = Ax;
            pw::A_vec_split_monopole(xc, yf, zc - 0.5*dx3, x0, y0, z0, theta0, delta, A0, r_star, r_interior, Ax, Ay, Az); Ax_zM = Ax;

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

            pw::A_vec_split_monopole(xc + 0.5*dx1, yc, zf, x0, y0, z0, theta0, delta, A0, r_star, r_interior, Ax, Ay, Az); Ay_xP = Ay;
            pw::A_vec_split_monopole(xc - 0.5*dx1, yc, zf, x0, y0, z0, theta0, delta, A0, r_star, r_interior, Ax, Ay, Az); Ay_xM = Ay;
            pw::A_vec_split_monopole(xc, yc + 0.5*dx2, zf , x0, y0, z0, theta0, delta, A0, r_star, r_interior, Ax, Ay, Az); Ax_yP = Ax;
            pw::A_vec_split_monopole(xc, yc - 0.5*dx2, zf , x0, y0, z0, theta0, delta, A0, r_star, r_interior, Ax, Ay, Az); Ax_yM = Ax;

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
} 