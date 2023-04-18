#ifndef __FLOW_SOLVER_H__
#define __FLOW_SOLVER_H__

// for FlowSolver class:
#include "flow_solver_cases/flow_solver_case_base.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

// for generate_grid
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/base/function.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include "dg/dg_factory.hpp"
//#include "ode_solver/runge_kutta_ode_solver.h"
#include "ode_solver/explicit_ode_solver.h"
#include "ode_solver/ode_solver_factory.h"
#include <deal.II/base/table_handler.h>
#include <string>
#include <vector>
#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace FlowSolver {

#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

/// Base class of all the flow solvers.
/** Needed for main.cpp
 *  Generated by the FlowSolverFactory.
 */
class FlowSolverBase
{
public:
    /// Constructor
    FlowSolverBase () {};

    /// Destructor
    virtual ~FlowSolverBase() {};

    /// Basically the main and only function of this class.
    /** This will get overloaded by the derived flow solver class.
     */
    virtual int run() const = 0;
};


/// Selects which flow case to simulate.
template <int dim, int nstate, int sub_nstate = 1>
class FlowSolver : public FlowSolverBase
{
public:
    /// Constructor.
    FlowSolver(
        const Parameters::AllParameters *const parameters_input, 
        std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Constructor with a vector of parameters input.
    FlowSolver(
        const std::vector<Parameters::AllParameters*> &parameters_input, 
        std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case_input,
        std::shared_ptr<FlowSolverCaseBase<dim, sub_nstate>> sub_flow_solver_case_input,
        const std::vector<dealii::ParameterHandler> &parameter_handler_input);
    
    /// Destructor
    ~FlowSolver() {};

    /// Pointer to Flow Solver Case
    std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case;

    /// Pointer to Sub Flow Solver Case
    std::shared_ptr<FlowSolverCaseBase<dim, sub_nstate>> sub_flow_solver_case;

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// Simply runs the flow solver and returns 0 upon completion
    int run () const override;

    /// Setup for main flow solver
    void main_flow_solver_setup ();

    /// Setup for sub flow solver
    void sub_flow_solver_setup ();

    /// Initializes the data table from an existing file
    void initialize_data_table_from_file(
        std::string data_table_filename_with_extension,
        const std::shared_ptr <dealii::TableHandler> data_table) const;

    /// Returns the restart filename without extension given a restart index (adds padding appropriately)
    std::string get_restart_filename_without_extension(const int restart_index_input) const;

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
    
public:
    /// Pointer to dg so it can be accessed externally.
    std::shared_ptr<DGBase<dim, double>> dg;

    /// Pointer to ode solver so it can be accessed externally.
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver;

private:
    /** Returns the column names of a dealii::TableHandler object
     *  given the first line of the file */
    std::vector<std::string> get_data_table_column_names(const std::string string_input) const;

    /// Writes a parameter file (.prm) for restarting the computation with
    void write_restart_parameter_file(const int restart_index_input,
                                      const double constant_time_step_input) const;

    /// Converts a double to a string with scientific format and with full precision
    std::string double_to_string(const double value_input) const;

#if PHILIP_DIM>1
    /// Outputs all the necessary restart files
    void output_restart_files(
        const int current_restart_index,
        const double constant_time_step,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table) const;
#endif

    /// Performs mesh adaptation.
    /** Currently implemented for steady state flows.
     */
    void perform_steady_state_mesh_adaptation() const;

public:
    /// Pointer to sub dg so it can be accessed externally.
    std::shared_ptr<DGBase<dim, double>> sub_dg = nullptr;

    /// Pointer to sub ode solver so it can be accessed externally.
    std::shared_ptr<ODE::ODESolverBase<dim, double>> sub_ode_solver = nullptr;

};

} // FlowSolver namespace
} // PHiLiP namespace
#endif
