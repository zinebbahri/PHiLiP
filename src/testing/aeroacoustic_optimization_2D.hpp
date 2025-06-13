#ifndef __2D_AEROACOUSTIC_OPTIMIZATION_H__
#define __2D_AEROACOUSTIC_OPTIMIZATION_H__

#include <deal.II/grid/manifold_lib.h>

#include "ROL_Bounds.hpp"
#include "ROL_BoundConstraint_SimOpt.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_Reduced_Constraint_SimOpt.hpp"
#include "ROL_Constraint_Partitioned.hpp"

// for generate_grid
#include <deal.II/base/function.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>
#include <stdlib.h>

#include "testing/tests.h"
#include "dg/dg_base.hpp"
#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include "optimization/flow_constraints_physics_model.hpp"
#include <deal.II/base/parameter_handler.h>

#include <iostream>

#include "dg/dg_factory.hpp"
#include "physics/physics_factory.h"
//#include "ode_solver/runge_kutta_ode_solver.h"
#include "ode_solver/runge_kutta_ode_solver.h"
#include "ode_solver/ode_solver_factory.h"
#include <deal.II/base/table_handler.h>
#include <string>
#include <vector>

namespace PHiLiP {
namespace Tests {
// using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class AeroAcousticOptimization2D: public TestsBase
{
public:
    /// Constructor. Deleted the default constructor since it should not be used
    AeroAcousticOptimization2D () = delete;
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    AeroAcousticOptimization2D(const std::vector<Parameters::AllParameters*> &parameters_input);

    /// Parameter handler for storing the .prm file being ran
    // const dealii::ParameterHandler &parameter_handler;  
  
    /// Grid convergence on Euler Gaussian Bump
    /** Will run the a grid convergence test for various p
     *  on multiple grids to determine the order of convergence.
     *
     *  Expecting the solution to converge at p+1. and output to converge at 2p+1.
     *  Note that the output solution currently convergens slightly suboptimally
     *  depending on the case (around 2p). The implementation of the boundary conditions
     *  play a large role on this adjoint consistency.
     *  
     *  Want to see entropy go to 0.
     */
    int run_test () const;

    /// Pointer to sub dg so it can be accessed externally.
    // std::shared_ptr<DGBase<dim, double>> dg;
    /// Pointer to ode solver so it can be accessed externally.
    // std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver;
    /// Pointer to sub dg so it can be accessed externally.
    // std::shared_ptr<DGBase<dim, double>> sub_dg;
    /// Pointer to sub ode solver so it can be accessed externally.
    // std::shared_ptr<ODE::ODESolverBase<dim, double>> sub_ode_solver = nullptr;


protected:
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    const int mpi_rank; ///< MPI rank.
    const int n_mpi; ///< Number of MPI processes.
    /// ConditionalOStream.
    /** Used as std::cout, but only prints if mpi_rank == 0
     */
    dealii::ConditionalOStream pcout;
    const Parameters::AllParameters all_param; ///< All parameters
    Parameters::AllParameters sub_all_param; ///< Sub all parameters
    const Parameters::FlowSolverParam flow_solver_param; ///< Flow solver parameters
    Parameters::FlowSolverParam sub_flow_solver_param; ///< Sub flow solver parameters
    const Parameters::ODESolverParam ode_param; ///< ODE solver parameters
    Parameters::ODESolverParam sub_ode_param; ///< Sub ODE solver parameters
    const unsigned int poly_degree; ///< Polynomial order
    unsigned int sub_poly_degree; ///< Sub polynomial order
    const unsigned int grid_degree; ///< Polynomial order of the grid
    unsigned int sub_grid_degree; ///< Sub polynomial order of the grid
    const double final_time; ///< Final time of solution

    /// Name of the reference copy of inputted parameters file; for restart purposes
    const std::string input_parameters_file_reference_copy_filename;

    const bool do_output_solution_at_fixed_times; ///< Flag for outputting solution at fixed times
    const unsigned int number_of_fixed_times_to_output_solution; ///< Number of fixed times to output the solution
    const bool output_solution_at_exact_fixed_times;///< Flag for outputting the solution at exact fixed times by decreasing the time step on the fly


    /// Performs mesh adaptation.
    /** Currently implemented for steady state flows.
     */
    void perform_steady_state_mesh_adaptation(std::shared_ptr<DGBase<dim, double>> dg, std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver) const;

    /// Actual test for which the number of design variables can be inputted.
    int optimize (const unsigned int nx_ffd, const unsigned int poly_degree/*, 
                  std::shared_ptr<DGBase<dim, double>> dg, std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver,
                  std::shared_ptr<DGBase<dim, double>> sub_dg, std::shared_ptr<ODE::ODESolverBase<dim, double>> sub_ode_solver*/) const;

    ROL::Ptr<ROL::Vector<double>> getDesignVariables(
        ROL::Ptr<ROL::Vector<double>> simulation_variables,
        ROL::Ptr<ROL::Vector<double>> control_variables,
        const bool is_reduced_space) const;

    ROL::Ptr<ROL::Objective<double>> getObjective(
        const ROL::Ptr<ROL::Objective_SimOpt<double>> objective_simopt,
        const ROL::Ptr<ROL::Constraint_SimOpt<double>> flow_constraints,
        const ROL::Ptr<ROL::Vector<double>> simulation_variables,
        const ROL::Ptr<ROL::Vector<double>> control_variables,
        const bool is_reduced_space) const;

    ROL::Ptr<ROL::BoundConstraint<double>> getDesignBoundConstraint(
        ROL::Ptr<ROL::Vector<double>> simulation_variables,
        ROL::Ptr<ROL::Vector<double>> control_variables,
        const bool is_reduced_space) const;

    ROL::Ptr<ROL::Constraint<double>> getEqualityConstraint(void) const;
    ROL::Ptr<ROL::Vector<double>> getEqualityMultiplier(void) const;
    std::vector<ROL::Ptr<ROL::Constraint<double>>> getInequalityConstraint(
		const std::vector<ROL::Ptr<ROL::Objective_SimOpt<double>>> constraints,
		const ROL::Ptr<ROL::Constraint_SimOpt<double>> flow_constraints,
		const ROL::Ptr<ROL::Vector<double>> simulation_variables,
		const ROL::Ptr<ROL::Vector<double>> control_variables,
		const bool is_reduced_space
		) const;

    std::vector<ROL::Ptr<ROL::Vector<double>>> getInequalityMultiplier(std::vector<double>& nonlinear_inequality_targets) const;
    std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> getSlackBoundConstraint(
        const std::vector<double>& nonlinear_targets,
        const std::vector<double>& lower_bound_dx,
        const std::vector<double>& upper_bound_dx) const;

    int check_flow_constraints(
        const unsigned int nx_ffd,
        ROL::Ptr<FlowConstraintsPhysicsModel<dim>> flow_constraints,
        ROL::Ptr<ROL::Vector<double>> design_simulation,
        ROL::Ptr<ROL::Vector<double>> design_control,
        ROL::Ptr<ROL::Vector<double>> dual_equality_state);
    int check_objective(
        ROL::Ptr<ROL::Objective_SimOpt<double>> objective_simopt,
        ROL::Ptr<FlowConstraintsPhysicsModel<dim>> flow_constraints,
        ROL::Ptr<ROL::Vector<double>> design_simulation,
        ROL::Ptr<ROL::Vector<double>> design_control,
        ROL::Ptr<ROL::Vector<double>> dual_equality_state);
    int check_reduced_constraint(
        const unsigned int nx_ffd,
        ROL::Ptr<ROL::Constraint<double>> reduced_constraint,
        ROL::Ptr<ROL::Vector<double>> control_variables,
        ROL::Ptr<ROL::Vector<double>> lift_residual_dual);

};




} // Tests namespace
} // PHiLiP namespace
#endif

