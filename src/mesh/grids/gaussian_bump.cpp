#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <Sacado.hpp>
#include "gaussian_bump.h"

namespace PHiLiP {
namespace Grids {

template<int dim>
void gaussian_bump(
    dealii::parallel::distributed::Triangulation<dim> &grid,
    const std::vector<unsigned int> n_subdivisions,
    const double channel_length,
    const double channel_height,
    const double bump_height,
    const double channel_width)
{
    const double x_start = channel_length * 0.5;

    assert(n_subdivisions.size() == dim);

    dealii::Point<dim> p1, p2;
    for (int d=0;d<dim;++d) {
        if (d==0) {
            p1[d] = -x_start;
            p2[d] = x_start;
        }
        if (d==1) {
            p1[d] = 0.0;
            p2[d] = channel_height;
        }
        if (d==2) {
            p1[d] = 0.0;
            p2[d] = channel_width;
        }
    }
    const bool colorize = true;
    dealii::GridGenerator::subdivided_hyper_rectangle (grid, n_subdivisions, p1, p2, colorize);

    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                int current_id = cell->face(face)->boundary_id();
                if (current_id == 4 || current_id == 5) cell->face(face)->set_boundary_id (-9999); // Side walls wall
                if (current_id == 2 || current_id == 3) cell->face(face)->set_boundary_id (1001); // Bottom and top wall
                if (current_id == 1) cell->face(face)->set_boundary_id (1002); // Outflow with supersonic or back_pressure
                if (current_id == 0) cell->face(face)->set_boundary_id (1003); // Inflow

                if (current_id == 2) {
                    cell->face(face)->set_user_index(1); // Bottom wall
                } else {
                    cell->face(face)->set_user_index(-1); // All other boundaries.
                }
            }
        }
    }

    const BumpManifold<dim> bump_manifold(channel_height, bump_height);

    // Warp grid to be a gaussian bump
    //dealii::GridTools::transform (&(BumpManifold<dim>::warp), grid);
    dealii::GridTools::transform (
        [&bump_manifold](const dealii::Point<dim> &chart_point) {
          return bump_manifold.push_forward(chart_point);}, grid);
    
    // Assign a manifold to have curved geometry
    unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
    grid.reset_all_manifolds();
    grid.set_all_manifold_ids(manifold_id);
    //grid.set_manifold ( manifold_id, bump_manifold );
   
    // // Set Flat manifold on the domain, but not on the boundary.
    grid.set_manifold(0, dealii::FlatManifold<dim>());
    grid.set_all_manifold_ids_on_boundary(1001,1);
    grid.set_manifold(1, bump_manifold);

    // Set boundary type and design type
    for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                int current_id = cell->face(face)->boundary_id();
                if (current_id == -9999) cell->face(face)->set_boundary_id (1001); // Side walls wall
            }
        }
    }
}

template<int dim>
template<typename real>
dealii::Point<dim,real> BumpManifold<dim>::mapping(const dealii::Point<dim,real> &chart_point) const 
{
    const real x_ref = chart_point[0];
    const real y_ref = chart_point[1];
    const real z_ref = (dim==3) ? chart_point[2] : 0;

    const real x_phys = x_ref;//-1.5+x_ref*3.0;
    const real z_phys = z_ref;//-1.5+x_ref*3.0;

    //const real y_phys = channel_height*y_ref + exp(coeff_expy*y_ref*y_ref)*bump_height*exp(coeff_expx*x_phys*x_phys) * (1.0+0.7*x_phys);

    real y_phys;

    if (dim == 2) {
        const double coeff2 = 2; // Increase for more aggressive INITIAL exponential spacing.
        real y_scaled = channel_height;
        y_scaled *= (exp(std::pow(y_ref,coeff2))-1.0);
        y_scaled /= (exp(std::pow(channel_height,coeff2))-1.0); // [0,channel_height]
        const real y_lower = bump_height*exp(coeff_expx*x_ref*x_ref);
        const real perturbation = y_lower * exp(coeff_expy*y_scaled*y_scaled);
        y_phys = y_scaled + perturbation;
    } else if (dim==3) {
        const real y_lower = bump_height*exp(coeff_expx*x_ref*x_ref);
        const real y_higher = channel_height;
        const real y_prop = std::pow(y_ref/channel_height,1);
        y_phys = y_lower + (y_higher - y_lower) * y_prop;
    }


    //std::cout << x_ref << " " << y_ref << " " << x_phys << " " << y_phys << std::endl;
    dealii::Point<dim,real> phys_point;
    phys_point[0] = x_phys;
    phys_point[1] = y_phys;
    if (dim==3) phys_point[2] = z_phys;
    return phys_point;
}

template<int dim>
dealii::Point<dim> BumpManifold<dim>::pull_back(const dealii::Point<dim> &space_point) const {
    const double x_phys = space_point[0];
    const double target_y_phys = space_point[1];
    const double z_phys = (dim==3) ? space_point[2] : 0;

    double x_ref = x_phys + 0.001;
    double y_ref = target_y_phys;
    double z_ref = z_phys + 0.001;

    using ADtype = Sacado::Fad::DFad<double>;
    ADtype x_ref_ad = x_ref;
    ADtype y_ref_ad = y_ref;
    ADtype z_ref_ad = z_ref;

    // // Perform a few bisection iterations.
    // double lower_y = -0.1, upper_y = 1.01;
    // double midpoint;
    // double fun_mid = 1.0;
    // int max_bisection = 1000;
    // int it_bisection = 0;
    // std::cout << "Starting bisection..." << std::endl;
    // while (std::abs(fun_mid) >= 1e-15 && it_bisection < max_bisection) {
    //     it_bisection++;
    //     // Find middle point
    //     midpoint = (lower_y+upper_y)/2;

    //     dealii::Point<dim,double> chart_point;
    //     chart_point[0] = x_ref;
    //     if (dim==3) chart_point[2] = z_ref;

    //     chart_point[1] = midpoint;
    //     dealii::Point<dim,double> new_midpoint = mapping<double>(chart_point);
    //     fun_mid = new_midpoint[1] - target_y_phys;

    //     chart_point[1] = lower_y;
    //     dealii::Point<dim,double> new_lowerpoint = mapping<double>(chart_point);
    //     double fun_low = new_lowerpoint[1] - target_y_phys;

    //     std::cout << midpoint << " " << fun_mid << " " << fun_low << std::endl;

    //     y_ref_ad.val() = midpoint;
    //     // Check if middle point is root
    //     if (std::abs(fun_mid) <= 1e-15) {
    //       break;
    //     // Decide the side to repeat the steps
    //     } else if (fun_mid*fun_low <= 0) {
    //       upper_y = midpoint;
    //     } else {
    //       lower_y = midpoint;
    //     }
    // }
    // std::cout << "Found last point was... " << y_ref_ad.val() << " for a target_y_phys= " << target_y_phys <<  std::endl;
    // if (it_bisection > 900) {
    //     dealii::Point<dim,double> chart_point;
    //     chart_point[0] = x_ref;
    //     if (dim==3) chart_point[2] = z_ref;
    //     double low = -1, high = 1;
    //     int n = 1000;
    //     std::cout << "Failed to find point..." << std::endl;
    //     std::cout << "Last point was... " << y_ref_ad.val() << " for a target_y_phys= " << target_y_phys <<  std::endl;
    //     for (int i = 0; i < n; ++i) {
    //         chart_point[1] = low + i * (high-low) / n;
    //         dealii::Point<dim,double> new_midpoint = mapping<double>(chart_point);
    //         std::cout << chart_point[1] << " " << new_midpoint[1] - target_y_phys << std::endl;
    //     }
    // }
    // Perform a Newton iterations.
    int it_newton = 0;
    // std::cout << "Starting Newton: " << std::endl;
    // for (it_newton=0; it_newton<200; it_newton++) {
    //     dealii::Point<dim,ADtype> chart_point_ad;
    //     y_ref_ad.diff(0,1);
    //     chart_point_ad[0] = x_ref_ad;
    //     chart_point_ad[1] = y_ref_ad;
    //     if (dim==3) chart_point_ad[2] = z_ref_ad;
    //     dealii::Point<dim,ADtype> new_point = mapping<ADtype>(chart_point_ad);

    //     const double fun = new_point[1].val() - target_y_phys;
    //     std::cout
    //         << " y_ref " << y_ref_ad.val()
    //         << " yref_to_yphys " << new_point[1].val()
    //         << " target_y_phys " << target_y_phys
    //         << " diff " << fun
    //         << std::endl;
    //     const double derivative = new_point[1].dx(0);
    //     if(std::abs(fun) < 1e-15) break;
    //     y_ref_ad = y_ref_ad - fun/derivative;
    // }

    for (it_newton=0; it_newton<200; it_newton++) {
        dealii::Point<dim,ADtype> chart_point_ad;
        x_ref_ad.diff(0,dim);
        y_ref_ad.diff(1,dim);
        if (dim==3) z_ref_ad.diff(2,dim);
        chart_point_ad[0] = x_ref_ad;
        chart_point_ad[1] = y_ref_ad;
        if (dim==3) chart_point_ad[2] = z_ref_ad;
        dealii::Point<dim,ADtype> new_point = mapping<ADtype>(chart_point_ad);

        dealii::Tensor<1,dim,double> fun;
        fun[0] = new_point[0].val() - space_point[0];
        fun[1] = new_point[1].val() - space_point[1];
        if (dim==3) fun[2] = new_point[2].val() - space_point[2];
        dealii::Tensor<2,dim,double> jac;
        for (int id = 0; id < dim; ++id) {
            for (int jd = 0; jd < dim; ++jd) {
                jac[id][jd] = new_point[id].dx(jd);
            }
        }
        dealii::Tensor<2,dim,double> inv_jac = invert(jac);
        // std::cout << fun << std::endl;
        // std::cout << jac << std::endl;
        // std::cout << inv_jac << std::endl;
        // std::cout
        //     << " y_ref " << y_ref_ad.val()
        //     << " yref_to_yphys " << new_point[1].val()
        //     << " target_y_phys " << target_y_phys
        //     << " diff " << fun
        //     << std::endl;
        if(fun.norm() < 1e-15) break;

        dealii::Tensor<1,dim,double> old_ref;
        old_ref[0] = x_ref_ad.val();
        old_ref[1] = y_ref_ad.val();
        if (dim==3) old_ref[2] = z_ref_ad.val();

        dealii::Tensor<1,dim,double> new_ref;
        new_ref = old_ref - inv_jac*fun;

        x_ref_ad.val() = new_ref[0];
        y_ref_ad.val() = new_ref[1];
        if(dim==3) z_ref_ad.val() = new_ref[2];
    }

    if (it_newton > 198) {
        dealii::Point<dim,double> chart_point;
        chart_point[0] = x_ref;
        if (dim==3) chart_point[2] = 0.0;//z_ref;
        double low = -0.1, high = 0.1;
        int n = 1000;
        std::cout << "Failed to find point..." << std::endl;
        std::cout << "Last point was... " << y_ref_ad.val() << " for a target_y_phys= " << target_y_phys <<  std::endl;
        for (int i = 0; i < n; ++i) {
            chart_point[1] = low + i * (high-low) / n;
            dealii::Point<dim,double> new_midpoint = mapping<double>(chart_point);
            std::cout << chart_point[1] << " " << new_midpoint[1] - target_y_phys << std::endl;
        }
    }

    // double x1=y_ref_ad.val()-0.1, x2=y_ref_ad.val(), E=1e-14;
    // double n = 0, x0, c = 1.0;
    // while (std::abs(c) >= E) {
    //     dealii::Point<dim,double> chart_point;
    //     chart_point[0] = x_ref;
    //     if (dim==3) chart_point[2] = z_ref;

    //     chart_point[1] = x0;
    //     dealii::Point<dim,double> new_midpoint = mapping<double>(chart_point);
    //     double fx0 = new_midpoint[1] - target_y_phys;

    //     chart_point[1] = x1;
    //     new_midpoint = mapping<double>(chart_point);
    //     double fx1 = new_midpoint[1] - target_y_phys;

    //     chart_point[1] = x2;
    //     new_midpoint = mapping<double>(chart_point);
    //     double fx2 = new_midpoint[1] - target_y_phys;

    //     // calculate the intermediate value
    //     x0 = (x1 * fx2 - x2 * fx1) / (fx2 - fx1);

    //     // check if x0 is root of equation or not
    //     c = fx1 * fx0;

    //     // update the value of interval
    //     x1 = x2;
    //     x2 = x0;

    //     // update number of iteration
    //     n++;

    //     // if x0 is the root of equation then break the loop
    //     if (std::abs(c) <= E)
    //         break;
    // }
    // y_ref_ad.val() = x0;
    // std::cout << "Root of the given equation=" << x0 << std::endl;
    // std::cout << "No. of iterations = " << n << std::endl;

    dealii::Point<dim,double> chart_point;
    chart_point[0] = x_ref_ad.val();
    chart_point[1] = y_ref_ad.val();
    if (dim==3) chart_point[2] = z_ref_ad.val();

    dealii::Point<dim,double> new_point = mapping<double>(chart_point);
    const double yref_to_yphys = new_point[1];
    const double error = std::abs(yref_to_yphys - target_y_phys);
    if (error > 1e-13) {
        std::cout << "Large error " << error << std::endl;
        std::cout << " xref " << chart_point[0]
                  << " yref " << chart_point[1]
                  << " zref " << ((dim==3) ? chart_point[2] : 0)
                  << " target_y_phys " << target_y_phys
                  << " yref_to_yphys " << yref_to_yphys
                  << " error " << error << std::endl;
    }

    return chart_point;
}

template<int dim>
dealii::Point<dim> BumpManifold<dim>::push_forward(const dealii::Point<dim> &chart_point) const 
{
    return mapping<double>(chart_point);
}

template<int dim>
dealii::DerivativeForm<1,dim,dim> BumpManifold<dim>::push_forward_gradient(const dealii::Point<dim> &chart_point) const
{
    using ADtype = Sacado::Fad::DFad<double>;
    ADtype x_ref = chart_point[0];
    ADtype y_ref = chart_point[1];
    ADtype z_ref = (dim==3) ? chart_point[2] : 0;
    x_ref.diff(0,dim);
    y_ref.diff(1,dim);
    if(dim==3) z_ref.diff(2,dim);
    dealii::Point<dim,ADtype> chart_point_ad(x_ref,y_ref);
    dealii::Point<dim,ADtype> new_point = mapping<ADtype>(chart_point_ad);

    dealii::DerivativeForm<1, dim, dim> dphys_dref;
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            dphys_dref[i][j] = new_point[i].dx(j);
        }
    }


    return dphys_dref;
}

template<int dim>
std::unique_ptr<dealii::Manifold<dim,dim> > BumpManifold<dim>::clone() const
{
    return std::make_unique<BumpManifold<dim>>(channel_height,bump_height);
}

#if PHILIP_DIM==1
#else
template void gaussian_bump<PHILIP_DIM>(
    dealii::parallel::distributed::Triangulation<PHILIP_DIM> &grid,
    const std::vector<unsigned int> n_subdivisions,
    const double channel_length,
    const double channel_height,
    const double bump_height,
    const double channel_width);
#endif
} // namespace Grids
} // namespace PHiLiP
