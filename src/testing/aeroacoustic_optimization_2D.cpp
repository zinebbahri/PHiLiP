#include <fenv.h> // catch nan
#include <stdlib.h>     /* srand, rand */
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include "reduced_order/pod_basis_offline.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "mesh/mesh_adaptation/mesh_adaptation.h"
#include <deal.II/base/timer.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/optimization/rol/vector_adaptor.h>

#include "Teuchos_GlobalMPISession.hpp"
#include "ROL_Bounds.hpp"
#include "ROL_BoundConstraint_SimOpt.hpp"
#include "ROL_Algorithm.hpp"
#include "optimization/rol_modified/ROL_Reduced_Objective_SimOpt_FailSafe.hpp"
#include "ROL_Reduced_Constraint_SimOpt.hpp"

#include "ROL_Constraint_Partitioned.hpp"

#include "ROL_OptimizationSolver.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"

#include "ROL_SingletonVector.hpp"
#include <ROL_AugmentedLagrangian_SimOpt.hpp>

#include "aeroacoustic_optimization_2D.hpp"

#include "physics/euler.h"
#include "physics/negative_spalart_allmaras_rans_model.h" // for FreeStreamInitialConditions_RANS_SA_negative
#include "physics/initial_conditions/initial_condition_function.h"
#include "dg/dg_factory.hpp"
// #include "ode_solver/ode_solver_base.h"
#include "ode_solver/ode_solver_factory.h"

#include "functional/target_boundary_functional.h"

#include "mesh/grids/gaussian_bump.h"
#include "mesh/free_form_deformation.h"

#include "optimization/rol_to_dealii_vector.hpp"
#include "optimization/flow_constraints.hpp"
#include "optimization/flow_constraints_physics_model.hpp"
#include "optimization/rol_objective_acoustic.hpp"
#include "optimization/rol_objective.hpp"
#include "optimization/constraintfromobjective_simopt.hpp"

#include "optimization/primal_dual_active_set.hpp"
#include "optimization/full_space_step.hpp"
#include "optimization/sequential_quadratic_programming.hpp"

#include "mesh/gmsh_reader.hpp"
// #include "mesh/grids/half_cylinder.hpp"
#include "mesh/grids/naca_airfoil_grid.hpp"
#include "functional/lift_drag.hpp"
#include "functional/moment.hpp"
#include "functional/geometric_volume.hpp"
#include "functional/target_wall_pressure.hpp"
#include "optimization/design_parameterization/ffd_parameterization.hpp"
#include "functional/extraction_functional.hpp"
#include "functional/amiet_model.hpp"
#include "functional/acoustic_adjoint.hpp"

#include "global_counter.hpp"

#include "mesh/grids/naca_airfoil_grid.hpp"
#include "mesh/high_order_grid.h"
#include "physics/initial_conditions/set_initial_condition.h"

//#define CREATE_RST
//#define REMOVE_BOUND

namespace {
const bool USE_LIFT_CONSTRAINT   = true;
const bool USE_MOMENT_CONSTRAINT = false;
const bool USE_VOLUME_CONSTRAINT = true;
const bool USE_DESIGN_CONSTRAINT = true;

enum class OptimizationAlgorithm { full_space_birosghattas, full_space_composite_step, reduced_space_bfgs, reduced_space_newton, reduced_sqp };
enum class Preconditioner { P2, P2A, P4, P4A, identity };
enum class GridType {naca0012, cylinder};
enum class OptimizationProblemType { drag_minimization, lift_target, inverse_pressure_design };

//const GridType grid_type = GridType::cylinder;
const GridType grid_type = GridType::naca0012;
const OptimizationProblemType optimization_problem_type = OptimizationProblemType::drag_minimization;
//const OptimizationProblemType optimization_problem_type = OptimizationProblemType::lift_target;
//const OptimizationProblemType optimization_problem_type = OptimizationProblemType::inverse_pressure_design;

const std::vector<Preconditioner> precond_list { Preconditioner::P4A };
//const std::vector<Preconditioner> precond_list { Preconditioner::P2, Preconditioner::P2A, Preconditioner::P4, Preconditioner::P4A };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::full_space_birosghattas, OptimizationAlgorithm::reduced_space_bfgs, OptimizationAlgorithm::reduced_space_newton };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::reduced_space_bfgs };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::reduced_sqp };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::reduced_space_newton };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::full_space_birosghattas };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::reduced_space_bfgs };
const std::vector<OptimizationAlgorithm> opt_list {
    //OptimizationAlgorithm::reduced_space_newton,
    // OptimizationAlgorithm::full_space_birosghattas,
    OptimizationAlgorithm::reduced_space_bfgs,
    };

const unsigned int POLY_START = 1;
const unsigned int POLY_END = 1; // Can do until at least P2

//const unsigned int n_des_var_start = 10;//20;
//const unsigned int n_des_var_end   = 40;//100;
//const unsigned int n_des_var_step  = 10;//20;
//const std::vector<unsigned int> n_des_var_list { 10, 20, 40, 80, 160, 320};//20;
//const std::vector<unsigned int> n_des_var_list { 40, 80, 160, 320};//20;
//const std::vector<unsigned int> n_des_var_list { 80, 160, 320};//20;
//const std::vector<unsigned int> n_des_var_list { 320};//20;
//const std::vector<unsigned int> n_des_var_list { 5, 10, 20, 40 };//20;
//const std::vector<unsigned int> n_des_var_list { 15, 25, 30, 35 };//20;
//const std::vector<unsigned int> n_des_var_list { 40 };//20;
//const std::vector<unsigned int> n_des_var_list { 5, 10, 15, 20, 25, 30, 35, 40 };//20;
//const std::vector<unsigned int> n_des_var_list { 30, 35, 40 };//20;
//const std::vector<unsigned int> n_des_var_list { 5 };
const std::vector<unsigned int> n_des_var_list { 10 };
//const std::vector<unsigned int> n_des_var_list { 160, 320};//20;
//const std::vector<unsigned int> n_des_var_list { 80, 160};//20;
//const std::vector<unsigned int> n_des_var_list { 20, 40};//20;
//const std::vector<unsigned int> n_des_var_list { 40, 80, 160};//20;
//const std::vector<unsigned int> n_des_var_list { 160, 320};//20;
//const std::vector<unsigned int> n_des_var_list { 10, 20 } ;//20;

const int max_design_cycle = 1500;
const double GRAD_CONVERGENCE = 1e-7;

const double FD_TOL = 1e-6;
const double CONSISTENCY_ABS_TOL = 1e-10;

const int LINESEARCH_MAX_ITER = 5;
const double BACKTRACKING_RATE = 0.5;
const int PDAS_MAX_ITER = 1;

const double LINEAR_SOLVER_ABS_TOL = 1e-12;
const double LINEAR_SOLVER_REL_TOL = 1e-4;
const int LINEAR_SOLVER_MAX_ITS = 500;

const std::string line_search_curvature = "Null Curvature Condition";
const std::string line_search_method = "Backtracking";
}

namespace PHiLiP {
namespace Tests {


template <int dim, int nstate>
AeroAcousticOptimization2D<dim,nstate>::
AeroAcousticOptimization2D(const std::vector<Parameters::AllParameters*> &parameters_input
                           /*const std::vector<dealii::ParameterHandler> &parameter_handler_input*/)
    :TestsBase::TestsBase(parameters_input)
    // , parameter_handler(parameter_handler_input[0])
    , mpi_communicator(MPI_COMM_WORLD)
    , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , n_mpi(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank==0)
    , all_param(*parameters_input[0])
    , sub_all_param(*parameters_input[1])
    , flow_solver_param(all_param.flow_solver_param)
    , sub_flow_solver_param(sub_all_param.flow_solver_param)
    , ode_param(all_param.ode_solver_param)
    , sub_ode_param(sub_all_param.ode_solver_param)
    , poly_degree(flow_solver_param.poly_degree)
    , sub_poly_degree(sub_flow_solver_param.poly_degree)
    , grid_degree(flow_solver_param.grid_degree)
    , sub_grid_degree(sub_flow_solver_param.grid_degree)
    , final_time(flow_solver_param.final_time)
    , input_parameters_file_reference_copy_filename(flow_solver_param.restart_files_directory_name + std::string("/") + std::string("input_copy.prm"))
    , do_output_solution_at_fixed_times(ode_param.output_solution_at_fixed_times)
    , number_of_fixed_times_to_output_solution(ode_param.number_of_fixed_times_to_output_solution)
    , output_solution_at_exact_fixed_times(ode_param.output_solution_at_exact_fixed_times)
    // , dg(DGFactory<dim,double>::create_discontinuous_galerkin(&all_param,
    //                                                           &sub_all_param, 
    //                                                           poly_degree,
    //                                                           flow_solver_param.max_poly_degree_for_adaptation, 
    //                                                           grid_degree, 
    //                                                           flow_solver_case->generate_grid()))
    // , sub_dg(DGFactory<dim,double>::create_discontinuous_galerkin(&sub_all_param, 
    //                                                               sub_poly_degree, 
    //                                                               sub_flow_solver_param.max_poly_degree_for_adaptation, 
    //                                                               sub_grid_degree, 
    //                                                               sub_flow_solver_case->generate_grid()))
    {
        pcout << "A sub flow solver detected." << std::endl;
        // pcout << "Set up for main and sub flow solver:" << std::endl;
        // pcout << "Set up main flow solver." << std::endl;
        // main_flow_solver_setup();
        // pcout << "Done." << std::endl;
        // pcout << "Set up sub flow solver." << std::endl;
        // sub_flow_solver_setup();
        // pcout << "Done." << std::endl;
    }


namespace {
    double check_maximum_relative_error(std::vector<std::vector<double>> rol_check_results) {
        double max_rel_err = 999999;
        for (unsigned int i = 0; i < rol_check_results.size(); ++i) {
            const double abs_val_ad = std::abs(rol_check_results[i][1]);
            const double abs_val_fd = std::abs(rol_check_results[i][2]);
            const double abs_err    = std::abs(rol_check_results[i][3]);
            const double rel_err    = abs_err / std::max(abs_val_ad,abs_val_fd);
            max_rel_err = std::min(max_rel_err, rel_err);
        }
        return max_rel_err;
    }

#ifndef CREATE_RST
    std::string get_restart_filename_without_extension(const int restart_index_input) {
        // returns the restart file index as a string with appropriate padding
        std::string restart_index_string = std::to_string(restart_index_input);
        const unsigned int length_of_index_with_padding = 5;
        const int number_of_zeros = length_of_index_with_padding - restart_index_string.length();
        restart_index_string.insert(0, number_of_zeros, '0');

        const std::string prefix = "restart-";
        const std::string restart_filename_without_extension = prefix+restart_index_string;

        return restart_filename_without_extension;
    }
#endif
}

template<int dim, int nstate>
int AeroAcousticOptimization2D<dim,nstate>
::check_flow_constraints(
    const unsigned int nx_ffd,
    ROL::Ptr<FlowConstraintsPhysicsModel<dim>> flow_constraints,
    ROL::Ptr<ROL::Vector<double>> design_simulation,
    ROL::Ptr<ROL::Vector<double>> design_control,
    ROL::Ptr<ROL::Vector<double>> dual_equality_state)
{
    // Physics::Euler<dim,nstate,double> euler_physics_double
    //     = Physics::Euler<dim, nstate, double>(
    //             1,
    //             1.4,
    //             0.8,
    //             1.25,
    //             0.0);
    // FreeStreamInitialConditions<dim,nstate,double> initial_conditions(euler_physics_double);

    int test_error = 0;
    // Temporary vectors
    const auto temp_sim = design_simulation->clone();
    const auto temp_ctl = design_control->clone();
    const auto v1 = temp_sim->clone();
    const auto v2 = temp_ctl->clone();

    const auto jv1 = temp_sim->clone();
    const auto jv2 = temp_sim->clone();

    v1->zero();
    v1->setScalar(1.0);
    v2->zero();
    v2->setScalar(1.0);

    std::vector<double> steps;
    for (int i = -2; i > -12; i--) {
        steps.push_back(std::pow(10,i));
    }
    const int order = 2;

    const ROL::Ptr<ROL::Vector_SimOpt<double>> des_var_rol_p = ROL::makePtr<ROL::Vector_SimOpt<double>>(design_simulation, design_control);

    Teuchos::RCP<std::ostream> outStream;
    ROL::nullstream bhs; // outputs nothing
    const unsigned int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    std::filebuf filebuffer;
    if (mpi_rank == 0) filebuffer.open ("flow_constraints_check"+std::to_string(nx_ffd)+".log",std::ios::out);
    std::ostream ostr(&filebuffer);
    if (mpi_rank == 0) outStream = ROL::makePtrFromRef(ostr);
    else if (mpi_rank == 1) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);

    *outStream << "flow_constraints->checkApplyJacobian_1..." << std::endl;
    *outStream << "Checks dRdW * v1 against R(w+h*v1,x)/h  ..." << std::endl;
    {
        std::vector<std::vector<double>> results
            = flow_constraints->checkApplyJacobian_1(*temp_sim, *temp_ctl, *v1, *jv1, steps, true, *outStream, order);

        const double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed flow_constraints->checkApplyJacobian_1..." << std::endl;
        }
    }

    *outStream << "flow_constraints->checkApplyJacobian_2..." << std::endl;
    *outStream << "Checks dRdX * v2 against R(w,x+h*v2)/h  ..." << std::endl;
    {
        std::vector<std::vector<double>> results
            = flow_constraints->checkApplyJacobian_2(*temp_sim, *temp_ctl, *v2, *jv2, steps, true, *outStream, order);

        const double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed flow_constraints->checkApplyJacobian_2..." << std::endl;
        }
    }

    *outStream << "flow_constraints->checkInverseJacobian_1..." << std::endl;
    *outStream << "Checks || v - Jinv J v || == 0  ..." << std::endl;
    {
        const double v_minus_Jinv_J_v = flow_constraints->checkInverseJacobian_1(*jv1, *v1, *temp_sim, *temp_ctl, true, *outStream);
        const double normalized_v_minus_Jinv_J_v = v_minus_Jinv_J_v / v1->norm();
        if (normalized_v_minus_Jinv_J_v > CONSISTENCY_ABS_TOL) {
            test_error++;
            *outStream << "Failed flow_constraints->checkInverseJacobian_1..." << std::endl;
        }
    }

    *outStream << "flow_constraints->checkInverseAdjointJacobian_1..." << std::endl;
    *outStream << "Checks || v - Jtinv Jt v || == 0  ..." << std::endl;
    {
        const double v_minus_Jinv_J_v = flow_constraints->checkInverseAdjointJacobian_1(*jv1, *v1, *temp_sim, *temp_ctl, true, *outStream);
        const double normalized_v_minus_Jinv_J_v = v_minus_Jinv_J_v / v1->norm();
        if (normalized_v_minus_Jinv_J_v > CONSISTENCY_ABS_TOL) {
            test_error++;
            *outStream << "Failed flow_constraints->checkInverseAdjointJacobian_1..." << std::endl;
        }

    }

    *outStream << "flow_constraints->checkAdjointConsistencyJacobian..." << std::endl;
    *outStream << "Checks (w J v) versus (v Jt w)  ..." << std::endl;
    {
        const auto w = dual_equality_state->clone();
        const auto v = des_var_rol_p->clone();
        const auto x = des_var_rol_p->clone();
        const auto temp_Jv = dual_equality_state->clone();
        const auto temp_Jtw = des_var_rol_p->clone();
        const bool printToStream = true;
        const double wJv_minus_vJw = flow_constraints->checkAdjointConsistencyJacobian (*w, *v, *x, *temp_Jv, *temp_Jtw, printToStream, *outStream);
        if (wJv_minus_vJw > CONSISTENCY_ABS_TOL) {
            test_error++;
            *outStream << "Failed flow_constraints->checkAdjointConsistencyJacobian..." << std::endl;
        }
    }

    *outStream << "flow_constraints->checkApplyAdjointHessian..." << std::endl;
    *outStream << "Checks (w H v) versus FD approximation  ..." << std::endl;
    {
        const auto dual = design_simulation->clone();
        const auto temp_sim_ctl = des_var_rol_p->clone();
        const auto v3 = des_var_rol_p->clone();
        const auto hv3 = des_var_rol_p->clone();

        std::vector<std::vector<double>> results
            = flow_constraints->checkApplyAdjointHessian(*des_var_rol_p, *dual, *v3, *hv3, steps, true, *outStream, order);

        const double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed flow_constraints->checkApplyAdjointHessian..." << std::endl;
        }
    }
    filebuffer.close();

    return test_error;
}

template<int dim, int nstate>
int AeroAcousticOptimization2D<dim,nstate>
::check_objective(
    ROL::Ptr<ROL::Objective_SimOpt<double>> objective_simopt,
    ROL::Ptr<FlowConstraintsPhysicsModel<dim>> flow_constraints,
    ROL::Ptr<ROL::Vector<double>> design_simulation,
    ROL::Ptr<ROL::Vector<double>> design_control,
    ROL::Ptr<ROL::Vector<double>> dual_equality_state)
{
    int test_error = 0;
    const bool storage = false;
    const bool useFDHessian = false;
    auto robj = ROL::makePtr<ROL::Reduced_Objective_SimOpt_FailSafe<double>>( objective_simopt, flow_constraints, design_simulation, design_control, dual_equality_state, storage, useFDHessian);

    auto des_var_p = ROL::makePtr<ROL::Vector_SimOpt<double>>(design_simulation, design_control);

    Teuchos::RCP<std::ostream> outStream;
    ROL::nullstream bhs; // outputs nothing
    const unsigned int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    std::filebuf filebuffer;
    static int objective_count = 0;
    if (mpi_rank == 0) filebuffer.open ("objective_simopt"+std::to_string(objective_count)+"_check"+std::to_string(999)+".log",std::ios::out);
    objective_count++;
    std::ostream ostr(&filebuffer);
    if (mpi_rank == 0) outStream = ROL::makePtrFromRef(ostr);
    else if (mpi_rank == 1) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);

    std::vector<double> steps;
    for (int i = -2; i > -9; i--) {
        steps.push_back(std::pow(10,i));
    }
    const int order = 2;
    {
        const auto direction = des_var_p->clone();
        *outStream << "objective_simopt->checkGradient..." << std::endl;
        std::vector<std::vector<double>> results
            = objective_simopt->checkGradient( *des_var_p, *direction, steps, true, *outStream, order);

        const double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) test_error++;
    }
    {
        const auto direction_1 = des_var_p->clone();
        auto direction_2 = des_var_p->clone();
        direction_2->scale(0.5);
        *outStream << "objective_simopt->checkHessVec..." << std::endl;
        std::vector<std::vector<double>> results
            = objective_simopt->checkHessVec( *des_var_p, *direction_1, steps, true, *outStream, order);

        const double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) test_error++;

        *outStream << "objective_simopt->checkHessSym..." << std::endl;
        std::vector<double> results_HessSym = objective_simopt->checkHessSym( *des_var_p, *direction_1, *direction_2, true, *outStream);
        double wHv       = std::abs(results_HessSym[0]);
        double vHw       = std::abs(results_HessSym[1]);
        double abs_error = std::abs(wHv - vHw);
        double rel_error = abs_error / std::max(wHv, vHw);
        if (rel_error > FD_TOL) test_error++;
    }

    {
        const auto direction_ctl = design_control->clone();
        *outStream << "robj->checkGradient..." << std::endl;
        std::vector<std::vector<double>> results
            = robj->checkGradient( *design_control, *direction_ctl, steps, true, *outStream, order);

        const double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) test_error++;

    }
    filebuffer.close();

    return test_error;
}

template<int dim, int nstate>
int AeroAcousticOptimization2D<dim,nstate>
::check_reduced_constraint(
    const unsigned int nx_ffd,
    ROL::Ptr<ROL::Constraint<double>> reduced_constraint,
    ROL::Ptr<ROL::Vector<double>> control_variables,
    ROL::Ptr<ROL::Vector<double>> lift_residual_dual)
{
    int test_error = 0;

    std::vector<double> steps;
    for (int i = -2; i > -12; i--) {
        steps.push_back(std::pow(10,i));
    }
    const int order = 2;


    Teuchos::RCP<std::ostream> outStream;
    ROL::nullstream bhs; // outputs nothing
    const unsigned int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    std::filebuf filebuffer;
    if (mpi_rank == 0) filebuffer.open ("flow_constraints_check"+std::to_string(nx_ffd)+".log",std::ios::out);
    std::ostream ostr(&filebuffer);
    if (mpi_rank == 0) outStream = ROL::makePtrFromRef(ostr);
    else if (mpi_rank == 1) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);

    *outStream << "reduced_constraint->checkApplyJacobian..." << std::endl;
    *outStream << "Checks dRdW * v1 against R(w+h*v1,x)/h  ..." << std::endl;
    {
        const auto temp_ctl = control_variables->clone();
        *outStream << "After temp_ctl declaration ..." << std::endl;
        const auto v1 = control_variables->clone();
        const auto jv1 = lift_residual_dual->clone();
        *outStream << "Right after v1->setScalar(1.0)  ..." << std::endl;
        v1->setScalar(1.0);
        jv1->setScalar(1.0);

        *outStream << "Right before checkApplyJac  ..." << std::endl;
        std::vector<std::vector<double>> results
            = reduced_constraint->checkApplyJacobian(*temp_ctl, *v1, *jv1, steps, true, *outStream, order);


        double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed reduced_constraint->checkApplyJacobian..." << std::endl;

        }

        jv1->setScalar(1.0);
        *outStream << "Right before checkApplyAdjointJac  ..." << std::endl;
        auto c_temp = lift_residual_dual->clone();
        results = reduced_constraint->checkApplyAdjointJacobian(*temp_ctl, *jv1, *c_temp, *v1, true, *outStream, 10);

        max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed reduced_constraint->checkApplyAdjointJacobian..." << std::endl;
        }
    }

    *outStream << "reduced_constraint->checkAdjointConsistencyJacobian..." << std::endl;
    *outStream << "Checks (w J v) versus (v Jt w)  ..." << std::endl;
    const ROL::Ptr<ROL::Vector<double>> des_var_rol_p = control_variables->clone();
    {
        const auto w = lift_residual_dual->clone(); w->setScalar(1.0);
        const auto v = des_var_rol_p->clone();
        const auto x = des_var_rol_p->clone();
        const auto temp_Jv = lift_residual_dual->clone(); temp_Jv->setScalar(1.0);
        const auto temp_Jtw = des_var_rol_p->clone();
        const bool printToStream = true;
        const double wJv_minus_vJw = reduced_constraint->checkAdjointConsistencyJacobian (*w, *v, *x, *temp_Jv, *temp_Jtw, printToStream, *outStream);
        if (wJv_minus_vJw > CONSISTENCY_ABS_TOL) {
            test_error++;
            *outStream << "Failed reduced_constraint->checkAdjointConsistencyJacobian..." << std::endl;
        }
    }

    *outStream << "reduced_constraint->checkApplyAdjointHessian..." << std::endl;
    *outStream << "Checks (w H v) versus FD approximation  ..." << std::endl;
    {
        const auto dual = lift_residual_dual->clone(); dual->setScalar(1.0);
        const auto temp_sim_ctl = des_var_rol_p->clone();
        const auto v3 = des_var_rol_p->clone();
        const auto hv3 = des_var_rol_p->clone();

        std::vector<std::vector<double>> results
            = reduced_constraint->checkApplyAdjointHessian(*des_var_rol_p, *dual, *v3, *hv3, steps, true, *outStream, order);

        const double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed reduced_constraint->checkApplyAdjointHessian..." << std::endl;
        }
    }
    filebuffer.close();

    return test_error;
}


template <int dim, int nstate>
ROL::Ptr<ROL::Vector<double>> 
AeroAcousticOptimization2D<dim,nstate>::
getDesignVariables(
    ROL::Ptr<ROL::Vector<double>> simulation_variables,
    ROL::Ptr<ROL::Vector<double>> control_variables,
    const bool is_reduced_space) const
{
    if (is_reduced_space) {
        return control_variables;
    }
    ROL::Ptr<ROL::Vector<double>> design_variables_full = ROL::makePtr<ROL::Vector_SimOpt<double>>(simulation_variables, control_variables);
    return design_variables_full;
}


template <int dim, int nstate>
ROL::Ptr<ROL::Objective<double>> 
AeroAcousticOptimization2D<dim,nstate>::
getObjective(
    const ROL::Ptr<ROL::Objective_SimOpt<double>> objective,
    const ROL::Ptr<ROL::Constraint_SimOpt<double>> flow_constraints,
    const ROL::Ptr<ROL::Vector<double>> simulation_variables,
    const ROL::Ptr<ROL::Vector<double>> control_variables,
    const bool is_reduced_space) const
{
    const auto state_constraints = ROL::makePtrFromRef<PHiLiP::FlowConstraintsPhysicsModel<PHILIP_DIM>>(dynamic_cast<PHiLiP::FlowConstraintsPhysicsModel<PHILIP_DIM>&>(*flow_constraints));
    ROL::Ptr<ROL::Vector<double>> drag_adjoint = simulation_variables->clone();
    // int objective_check_error = check_objective<PHILIP_DIM,PHILIP_DIM+2>( objective, state_constraints, simulation_variables, control_variables, drag_adjoint);
    // (void) objective_check_error;

    if (!is_reduced_space) return objective;

    const bool storage = true;
    const bool useFDHessian = false;

    return ROL::makePtr<ROL::Reduced_Objective_SimOpt_FailSafe<double>>( objective, flow_constraints, simulation_variables, control_variables, drag_adjoint, storage, useFDHessian);
}

template <int dim, int nstate>
ROL::Ptr<ROL::BoundConstraint<double>>
AeroAcousticOptimization2D<dim,nstate>::
getDesignBoundConstraint(
    ROL::Ptr<ROL::Vector<double>> simulation_variables,
    ROL::Ptr<ROL::Vector<double>> control_variables,
    const bool is_reduced_space) const
{
    (void) simulation_variables;

    struct setUpper : public ROL::Elementwise::UnaryFunction<double> {
        private:
            double zero_;
        public:
            setUpper() : zero_(0) {}
            double apply(const double &x) const {
                if (grid_type == GridType::cylinder) {
                    if(x>zero_) { return ROL::ROL_INF<double>(); }
                    else        { return ROL::ROL_INF<double>(); }
                    //else { return zero_+0.1; }
                } else if (grid_type == GridType::naca0012) {
                    if (USE_DESIGN_CONSTRAINT) {
                    if(x>zero_) { return 0.1; }//ROL::ROL_INF<double>(); }
                    else { return zero_; }
                    } else {
                        if(x>zero_) { return ROL::ROL_INF<double>(); }
                        else        { return ROL::ROL_INF<double>(); }
                    }
                }
            }
    } setupper;
    struct setLower : public ROL::Elementwise::UnaryFunction<double> {
        private:
            double zero_;
        public:
            setLower() : zero_(0) {}
            double apply(const double &x) const {
                if (grid_type == GridType::cylinder) {
                    //if(x>zero_) { return zero_-0.1; }
                    //else { return -ROL::ROL_INF<double>(); }
                    if(x>zero_) { return -ROL::ROL_INF<double>(); }
                    else        { return -ROL::ROL_INF<double>(); }
                } else if (grid_type == GridType::naca0012) {
                    if (USE_DESIGN_CONSTRAINT) {
                    if(x<zero_) { return -1.0*0.1; }//ROL::ROL_INF<double>(); }
                    else { return zero_; }
                    } else {
                        if(x>zero_) { return -ROL::ROL_INF<double>(); }
                        else        { return -ROL::ROL_INF<double>(); }
                    }
                }
            }
    } setlower;

    ROL::Ptr<ROL::Vector<double>> l = control_variables->clone();
    ROL::Ptr<ROL::Vector<double>> u = control_variables->clone();

    l->applyUnary(setlower);
    u->applyUnary(setupper);

    double scale = 1;
    double feasTol = 1e-8;
    ROL::Ptr<ROL::BoundConstraint<double>> control_bounds = ROL::makePtr<ROL::Bounds<double>>(l,u, scale, feasTol);

    if (is_reduced_space) return control_bounds;

    ROL::Ptr<ROL::BoundConstraint<double>> simulation_bounds = ROL::makePtr<ROL::BoundConstraint<double>>(*simulation_variables);
    simulation_bounds->deactivate();
    ROL::Ptr<ROL::BoundConstraint<double>> design_bounds = ROL::makePtr<ROL::BoundConstraint_SimOpt<double>> (simulation_bounds, control_bounds);

    return design_bounds;
}


template <int dim, int nstate>
ROL::Ptr<ROL::Constraint<double>>
AeroAcousticOptimization2D<dim,nstate>::getEqualityConstraint(void) const
{
    return ROL::nullPtr;
}

template <int dim, int nstate>
ROL::Ptr<ROL::Vector<double>> 
AeroAcousticOptimization2D<dim,nstate>::getEqualityMultiplier(void) const
{
    return ROL::nullPtr;
}

template <int dim, int nstate>
std::vector<ROL::Ptr<ROL::Constraint<double>>>
AeroAcousticOptimization2D<dim,nstate>::
getInequalityConstraint(
    const std::vector<ROL::Ptr<ROL::Objective_SimOpt<double>>> constraints_as_objective,
    const ROL::Ptr<ROL::Constraint_SimOpt<double>> flow_constraints,
    const ROL::Ptr<ROL::Vector<double>> simulation_variables,
    const ROL::Ptr<ROL::Vector<double>> control_variables,
    const bool is_reduced_space
    ) const
{
    std::vector<ROL::Ptr<ROL::Constraint<double> > > cvec;
    if (is_reduced_space) {
		const bool storage = true;
		const bool useFDHessian = false;
		for (unsigned int i = 0; i < constraints_as_objective.size(); ++i) {
			ROL::Ptr<ROL::Vector<double>> adjoint = simulation_variables->clone();
			adjoint->setScalar(1.0);
			auto reduced_objective = ROL::makePtr<ROL::Reduced_Objective_SimOpt_FailSafe<double>>(
				constraints_as_objective[i], flow_constraints, simulation_variables, control_variables, adjoint, storage, useFDHessian);
			//const auto state_constraints = ROL::makePtrFromRef<PHiLiP::FlowConstraints<PHILIP_DIM>>(dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(*flow_constraints));
			//int objective_check_error = check_objective<PHILIP_DIM,PHILIP_DIM+2>( constraints_as_objective[i], state_constraints, simulation_variables, control_variables, adjoint);
			//(void) objective_check_error;
			ROL::Ptr<ROL::Constraint<double>> reduced_constraint = ROL::makePtr<ROL::ConstraintFromObjective<double>> (reduced_objective, 0.0);

			cvec.push_back(reduced_constraint);
		}
    } else {
		for (unsigned int i = 0; i < constraints_as_objective.size(); ++i) {
			ROL::Ptr<ROL::Constraint<double>> constraint = ROL::makePtr<PHiLiP::ConstraintFromObjective_SimOpt<double>> (constraints_as_objective[i], 0.0);
			cvec.push_back(constraint);
		}
    }
	return cvec;


}

template <int dim, int nstate>
std::vector<ROL::Ptr<ROL::Vector<double>>> 
AeroAcousticOptimization2D<dim,nstate>::
getInequalityMultiplier(std::vector<double>& nonlinear_inequality_targets) const
{
    std::vector<ROL::Ptr<ROL::Vector<double>>> emul;
    const unsigned int size = nonlinear_inequality_targets.size();
    for (unsigned int i = 0; i < size; ++i) {
        const ROL::Ptr<ROL::SingletonVector<double>> lift_constraint_dual = ROL::makePtr<ROL::SingletonVector<double>> (1.0);
        emul.push_back(lift_constraint_dual);
    }
    return emul;
}

template <int dim, int nstate>
std::vector<ROL::Ptr<ROL::BoundConstraint<double>>>
AeroAcousticOptimization2D<dim,nstate>::
getSlackBoundConstraint(
    const std::vector<double>& nonlinear_targets,
    const std::vector<double>& lower_bound_dx,
    const std::vector<double>& upper_bound_dx) const
{

    double scale = 1;
    double feasTol = 1e-4;

    std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> bcon;

    for (unsigned int i = 0; i < nonlinear_targets.size(); ++i) {
        const ROL::Ptr<ROL::SingletonVector<double>> nonlinear_lower_bound = ROL::makePtr<ROL::SingletonVector<double>> (nonlinear_targets[i]+lower_bound_dx[i]);
        const ROL::Ptr<ROL::SingletonVector<double>> nonlinear_upper_bound = ROL::makePtr<ROL::SingletonVector<double>> (nonlinear_targets[i]+upper_bound_dx[i]);
        auto nonlinear_bounds = ROL::makePtr<ROL::Bounds<double>> (nonlinear_lower_bound, nonlinear_upper_bound, scale, feasTol);
        bcon.push_back(nonlinear_bounds);
    }
    return bcon;
}

template<int dim, int nstate>
void AeroAcousticOptimization2D<dim,nstate>::perform_steady_state_mesh_adaptation(std::shared_ptr<DGBase<dim, double>> dg, std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver) const
{
    std::unique_ptr<MeshAdaptation<dim,double>> meshadaptation = std::make_unique<MeshAdaptation<dim,double>>(dg, &(this->all_param.mesh_adaptation_param));
    const int total_adaptation_cycles = this->all_param.mesh_adaptation_param.total_mesh_adaptation_cycles;
    double residual_norm = dg->get_residual_l2norm();
    
    pcout<<"Running mesh adaptation cycles..."<<std::endl;
    while (meshadaptation->current_mesh_adaptation_cycle < total_adaptation_cycles)
    {
        // Check if steady state solution is being used.
        if(residual_norm > ode_param.nonlinear_steady_residual_tolerance)
        {
            pcout<<"Mesh adaptation is currently implemented for steady state flows and the current residual norm isn't sufficiently low. "
                 <<"The solution has not converged. If p or hp adaptation is being used, issues with convergence might occur when integrating face terms with lower quad points at " 
                 <<"the face of adjacent elements with different p. Try increasing overintegration in the parameters file to fix it."<<std::endl;
            std::abort();
        }
        
        meshadaptation->adapt_mesh();
        ode_solver->steady_state();
        residual_norm = ode_solver->residual_norm;
        // flow_solver_case->steady_state_postprocessing(dg); 
    }

    pcout<<"Finished running mesh adaptation cycles."<<std::endl; 
}


template<int dim, int nstate>
int AeroAcousticOptimization2D<dim,nstate>
::run_test () const
{
    int test_error = 0;
    int design_space = 1;
    std::filebuf filebuffer;
    if (this->mpi_rank == 2) filebuffer.open ("optimization.log", std::ios::out);
    if (this->mpi_rank == 0) filebuffer.close();

    // for (unsigned int poly_degree = POLY_START; poly_degree <= POLY_END; ++poly_degree) {
        // for (const unsigned int n_des_var : n_des_var_list) {
            // const unsigned int nx_ffd = n_des_var + 2;
            const unsigned int nx_ffd = 10 + 2;
            if (design_space == 0)
            test_error += optimize(nx_ffd, poly_degree/*, this->header_dg, this->ode_solver, this->header_sub_dg, this->sub_ode_solver*/);
            else
            OASPL_design_space(nx_ffd);
        // }
    // }
    return test_error;
}

template<int dim, int nstate>
void AeroAcousticOptimization2D<dim,nstate>::OASPL_design_space(const unsigned int nx_ffd) const
{

    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
    using VectorAdaptor = dealii::Rol::VectorAdaptor<DealiiVector>;
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix;

    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr <Triangulation> grid = std::make_shared<Triangulation> (
    this->mpi_communicator,
    typename dealii::Triangulation<dim>::MeshSmoothing(
        dealii::Triangulation<dim>::smoothing_on_refinement |
        dealii::Triangulation<dim>::smoothing_on_coarsening));

    unsigned int n_design_variables = 0;
    dealii::Point<dim> ffd_origin;
    std::array<double,dim> ffd_rectangle_lengths;
    std::array<unsigned int,dim> ffd_ndim_control_pts;
    std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim;
    if constexpr (dim == 2) {
        if (grid_type == GridType::naca0012) {
        //// Coordinates for NACA deall II grid
        ffd_origin = dealii::Point<dim> (-0.025,-0.035);
        ffd_rectangle_lengths = std::array<double,dim> {{0.45,0.07}};
        }

        ffd_ndim_control_pts = {{nx_ffd,3}};

    }
    FreeFormDeformation<dim> ffd( ffd_origin, ffd_rectangle_lengths, ffd_ndim_control_pts);
    if constexpr (dim == 2) {
        // Vector of ijk indices and dimension.
        // Each entry in the vector points to a design variable's ijk ctl point and its acting dimension.
        for (unsigned int i_ctl = 0; i_ctl < ffd.n_control_pts; ++i_ctl) {

            const std::array<unsigned int,dim> ijk = ffd.global_to_grid ( i_ctl );
            for (unsigned int d_ffd = 0; d_ffd < dim; ++d_ffd) {

                if (   ijk[0] == 0 // Constrain first column of FFD points.
                    || ijk[0] == ffd_ndim_control_pts[0] - 1  // Constrain last column of FFD points.
                    || ijk[1] == 1 // Constrain middle row of FFD points.
                    || d_ffd == 0 // Constrain x-direction of FFD points.
                ) {
                    continue;
                }
                ++n_design_variables;
                ffd_design_variables_indices_dim.push_back(std::make_pair(i_ctl, d_ffd));
            }
        }
    }

    const dealii::IndexSet row_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, n_design_variables);
    dealii::IndexSet ghost_row_part(n_design_variables);
    ghost_row_part.add_range(0,n_design_variables);
    DealiiVector ffd_design_variables(row_part,ghost_row_part,MPI_COMM_WORLD);

    ffd.get_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    const auto initial_design_variables = ffd_design_variables;

    ffd_design_variables = initial_design_variables;
    ffd_design_variables.update_ghost_values();
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    // Initial optimization point
    grid->clear();
    dealii::GridGenerator::hyper_cube(*grid);

                // Exploring the design spaces of the OASPL reduction by deforming the NACA0012 mesh and computing the acoustic signature
            DealiiVector target_solution_ffd;
        
            std::ofstream outfile_init_FFD_coords;
            outfile_init_FFD_coords.open("FFD_init_coordinates.dat");
            std::ofstream outfile_final_FFD_coords;
            outfile_final_FFD_coords.open("FFD_final_coordinates.dat");
            for (unsigned int i_ctl = 0; i_ctl < ffd.n_control_pts; ++i_ctl) {

                const std::array<unsigned int,dim> ijk = ffd.global_to_grid ( i_ctl );
                if (   ijk[0] == 0 // Constrain first column of FFD points.
                    || ijk[0] == ffd_ndim_control_pts[0] - 1  // Constrain last column of FFD points.
                    || ijk[1] == 1 // Constrain middle row of FFD points.
                ) continue;


                outfile_init_FFD_coords << i_ctl << "  " << ffd.control_pts[i_ctl] << "\n";
                if(i_ctl == 1) { 
                    double dy = 0.0250632704918033;//0.0300759245901639;//0.0436817;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 2) { 
                    double dy = -0.0163610081967213;//-0.0196332098360656;//-0.0285149;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 3) { 
                    double dy = 0.00166123770491803;//0.00199348524590164;//0.0028953;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 4) { 
                    double dy = 0.0106013852459016;//0.012721662295082;//0.0184767;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 5) { 
                    double dy = 0.00692936885245902;//0.00831524262295082;//0.0120769;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 6) { 
                    double dy = 0;//-0.00337156721311476;//0.00489680000000001/2;//0;//0.00489680000000001;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 7) { 
                    double dy = 0;//-0.0153401901639344;//0.0222798/2;//0;//0.0222798;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 8) { 
                    double dy = 0;//-0.0189099147540984;//0.0274644/2;//0;//0.0274644;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 9) { 
                    double dy = 0;//-0.00990800655737705;//0.0143902/2;//0;//0.0143902;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 10) { 
                    double dy = 0;//0.00249624590163934;//-0.0036255/2;//0;//-0.0036255;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 25) { 
                    double dy = -0.0250298196721311;//-0.0300357836065574;//-0.0436234;//0;//-0.06107276;//-0.04798574;//0.0436234;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 26) { 
                    double dy =  0.0162540573770492;//0.019504868852459;//0.0283285;// 0;//0.0396599;//0.03116135;//-0.0283285;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 27) { 
                    double dy =  -0.00176767213114754;//-0.00212120655737705;//-0.0030808;// 0;//-0.00431312;//-0.00338888;// 0.0030808;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 28) { 
                    double dy =  -0.0107664016393443;//-0.0129196819672131;//-0.0187643;//0;//-0.02627002;//-0.02064073;// 0.0187643;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 29) { 
                    double dy = -0.00709358196721312;//-0.00851229836065574;//-0.0123631;//0;//-0.01730834;//-0.01359941;// 0.0123631;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 30) { 
                    double dy = 0;//0.00326326229508197;//-0.00473950000000001/2;//0;//0.00663530000000001;//0.00521345000000001;// -0.00473950000000001;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 31) { 
                    double dy = 0;//0.0151570426229508;//-0.0220138/2;//0;//0.03081932;//0.02421518;//-0.0220138;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 32) { 
                    double dy = 0;//0.0184158983606557;//-0.0267469/2;//0;//0.03744566;//0.02942159;//-0.0267469;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 33) { 
                    double dy = 0;//0.00925893442622951;//-0.0134475/2;//0;//0.0188265;//0.01479225;// -0.0134475;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                else if(i_ctl == 34) { 
                    double dy = 0;//-0.00269385245901639;//0.0039125/2;//0;//-0.0054775; //-0.00430375;//0.0039125;
                    ffd.control_pts[i_ctl][1] += dy;
                    }
                outfile_final_FFD_coords << ffd.control_pts[i_ctl] << "\n";
            }
            
            outfile_init_FFD_coords.close();
            outfile_final_FFD_coords.close();

    // using dealii Grid Generator
    std::shared_ptr<Triangulation> naca0012_mesh = std::make_shared<Triangulation> (
    #if dim!=1
        this->mpi_communicator
    #endif
        );

    dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
    airfoil_data.airfoil_type = "NACA";
    airfoil_data.naca_id      = "0012";
    airfoil_data.airfoil_length = all_param.flow_solver_param.airfoil_length;
    airfoil_data.height         = all_param.flow_solver_param.height;
    airfoil_data.length_b2      = all_param.flow_solver_param.length_b2;
    airfoil_data.incline_factor = all_param.flow_solver_param.incline_factor;
    airfoil_data.bias_factor    = all_param.flow_solver_param.bias_factor; 
    airfoil_data.refinements    = all_param.flow_solver_param.refinements;

    airfoil_data.n_subdivision_x_0 = all_param.flow_solver_param.n_subdivision_x_0;
    airfoil_data.n_subdivision_x_1 = all_param.flow_solver_param.n_subdivision_x_1;
    airfoil_data.n_subdivision_x_2 = all_param.flow_solver_param.n_subdivision_x_2;
    airfoil_data.n_subdivision_y = all_param.flow_solver_param.n_subdivision_y;
    airfoil_data.airfoil_sampling_factor = all_param.flow_solver_param.airfoil_sampling_factor; 

    dealii::GridGenerator::Airfoil::create_triangulation(*naca0012_mesh, airfoil_data);

        // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = naca0012_mesh->begin_active(); cell != naca0012_mesh->end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0 || current_id == 1 || current_id == 4 || current_id == 5) {
                    cell->face(face)->set_boundary_id (1005); // farfield
                } else {
                    cell->face(face)->set_boundary_id (1001); // wall
                }
            }
        }
    }
    const int poly_degree = 1;

std::shared_ptr < DGBase<dim, double> > dg_target = DGFactory<dim,double>::create_discontinuous_galerkin(&all_param, &sub_all_param, poly_degree,flow_solver_param.max_poly_degree_for_adaptation, grid_degree, naca0012_mesh);
std::shared_ptr < DGBase<dim, double> > sub_dg_target = DGFactory<dim,double>::create_discontinuous_galerkin(&sub_all_param, sub_poly_degree, sub_flow_solver_param.max_poly_degree_for_adaptation, sub_grid_degree, naca0012_mesh);

ffd.deform_mesh (*(dg_target->high_order_grid));
ffd.deform_mesh (*(sub_dg_target->high_order_grid));
ffd.output_ffd_vtu(2025);

// main flow solver set up
    if (ode_param.allocate_matrix_dRdW) {
        pcout << "Note: Allocating DG with AD matrix dRdW and dRdX only." << std::endl;
        dg_target->allocate_system(true,true,false); // FlowSolver only requires dRdW to be allocated
    } else {
        pcout << "Note: Allocating DG without AD matrices." << std::endl;
        dg_target->allocate_system(false,false,false);
    }

    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg_target);


    // Initialize solution
    SetInitialCondition<dim,nstate,double>::set_initial_condition(InitialConditionFactory<dim, nstate, double>::create_InitialConditionFunction(&all_param), dg_target, &all_param);

    dg_target->solution.update_ghost_values();
    dg_target->sub_solution.update_ghost_values();
    
    // Allocate ODE solver after initializing DG
    ode_solver->allocate_ode_system();

    
    // sub flow solver set up
    sub_dg_target->allocate_system();

    std::shared_ptr<ODE::ODESolverBase<dim, double>> sub_ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(sub_dg_target);

    pcout << "Initializing sub solution with initial condition function... " << std::flush;
    SetInitialCondition<dim,1,double>::set_initial_condition(InitialConditionFactory<dim, 1, double>::create_InitialConditionFunction(&sub_all_param), sub_dg_target, &sub_all_param);

    sub_dg_target->solution.update_ghost_values();
    pcout << "done." << std::endl;
    sub_ode_solver->allocate_ode_system();

// Solve sub and main flow solvers, steady state
        //----------------------------------------------------
        // Steady-state solution
        //----------------------------------------------------
        using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
        if(flow_solver_param.steady_state_polynomial_ramping && (ode_param.ode_solver_type != ODEEnum::pod_galerkin_solver && ode_param.ode_solver_type != ODEEnum::pod_petrov_galerkin_solver)) {
            ode_solver->initialize_steady_polynomial_ramping(poly_degree);
        }

        if(sub_dg_target){
            pcout << "Start calculation for the sub ODE solver..." << std::endl;
            sub_ode_solver->steady_state();
            // sub_flow_solver_case->steady_state_postprocessing(sub_dg);
            pcout << "End calculation for the sub ODE solver..." << std::endl;

            if(sub_dg_target->max_degree!=dg_target->max_degree){
                pcout << "Detect different polynomial degree between sub dg (poly_degree = " << sub_dg_target->max_degree << ") and main dg (poly_degree = " << dg_target->max_degree << ")..." << std::endl;
                pcout << "Interpolate solution from current polynomial degree " << sub_dg_target->max_degree << " to desired polynomial degree " << dg_target->max_degree << " ..." << std::endl;
                sub_dg_target->output_results_vtk(8888);
                sub_ode_solver->interpolate_solution_polynomial_degree(dg_target->max_degree);
                sub_dg_target->output_results_vtk(9999);
            }

            pcout << "Transfer the solution from sub dg to main dg..." << std::endl;
            dg_target->import_sub_solution(sub_dg_target);
        }else{
            pcout << "No calculation for the sub ODE solver..." << std::endl;
        }

        pcout << "Start calculation for the main ODE solver..." << std::endl;
        
        ode_solver->steady_state();
        // flow_solver_case->steady_state_postprocessing(dg);
        
        const bool use_isotropic_mesh_adaptation = (all_param.mesh_adaptation_param.total_mesh_adaptation_cycles > 0) 
                                        && (all_param.mesh_adaptation_param.mesh_adaptation_type != Parameters::MeshAdaptationParam::MeshAdaptationType::anisotropic_adaptation);
        
        if(use_isotropic_mesh_adaptation)
        {
            perform_steady_state_mesh_adaptation(dg_target, ode_solver);
        }


        //Compute lift, total drag, pressure drag, and acoustic noise
        LiftDragFunctional<dim,nstate,double> lift_functional( dg_target, LiftDragFunctional<dim,nstate,double>::Functional_types::lift);
        LiftDragFunctional<dim,nstate,double> pressure_drag_functional( dg_target, LiftDragFunctional<dim,nstate,double>::Functional_types::pressure_drag );
        LiftDragFunctional<dim,nstate,double> total_drag_functional( dg_target, LiftDragFunctional<dim,nstate,double>::Functional_types::total_drag );
        dealii::Point<dim,double> initial_extraction_point;
        if constexpr(dim==2){
            initial_extraction_point[0] = all_param.boundary_layer_extraction_param.extraction_point_x;
            initial_extraction_point[1] = all_param.boundary_layer_extraction_param.extraction_point_y;
        } else if constexpr(dim==3){
            initial_extraction_point[0] = all_param.boundary_layer_extraction_param.extraction_point_x;
            initial_extraction_point[1] = all_param.boundary_layer_extraction_param.extraction_point_y;
            initial_extraction_point[2] = 0;
        }
        int number_of_sampling = all_param.boundary_layer_extraction_param.number_of_sampling;

        //Update location of extraction point
        dealii::Point<dim,double> new_extraction_point = ffd.new_point_location(initial_extraction_point);

        std::cout << "New extraction point location: " << new_extraction_point[0] << " , " << new_extraction_point[1] << std::endl;

        ExtractionFunctional<dim,nstate,double,Triangulation> boundary_layer_extraction(dg_target, new_extraction_point, number_of_sampling);

        dealii::Point<3,double> observer_coord_ref;
        observer_coord_ref[0] = all_param.amiet_param.observer_coord_ref_x;
        observer_coord_ref[1] = all_param.amiet_param.observer_coord_ref_y;
        observer_coord_ref[2] = all_param.amiet_param.observer_coord_ref_z;

        AmietModelFunctional<dim,nstate,double,Triangulation> acoustic_functional = AmietModelFunctional<dim,nstate,double,Triangulation>(dg_target,boundary_layer_extraction,observer_coord_ref);

        std::cout << " Current lift = " << lift_functional.evaluate_functional()
                << ". Current pressure drag = " << pressure_drag_functional.evaluate_functional()
                  << ". Current total drag = " << total_drag_functional.evaluate_functional()
                << ". Current OASPL = " << acoustic_functional.evaluate_functional(true,true,false)
                << std::endl;
    
    return;

}

template<int dim, int nstate>
int AeroAcousticOptimization2D<dim,nstate>
::optimize (const unsigned int nx_ffd, const unsigned int level/*,
            std::shared_ptr<DGBase<dim, double>> dg, std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver,
            std::shared_ptr<DGBase<dim, double>> sub_dg, std::shared_ptr<ODE::ODESolverBase<dim, double>> sub_ode_solver*/) const
{
    int test_error = 0;

    for (auto const opt_type : opt_list) {
    for (auto const precond_type : precond_list) {

    std::string opt_output_name = "";
    std::string descent_method = "";
    std::string preconditioner_string = "";
    switch(opt_type) {
        case OptimizationAlgorithm::full_space_birosghattas: {
            opt_output_name = "full_space";
            switch(precond_type) {
                case Preconditioner::P2: {
                    opt_output_name += "_p2";
                    preconditioner_string = "P2";
                    break;
                }
                case Preconditioner::P2A: {
                    opt_output_name += "_p2a";
                    preconditioner_string = "P2A";
                    break;
                }
                case Preconditioner::P4: {
                    opt_output_name += "_p4";
                    preconditioner_string = "P4";
                    break;
                }
                case Preconditioner::P4A: {
                    opt_output_name += "_p4a";
                    preconditioner_string = "P4A";
                    break;
                }
                case Preconditioner::identity: {
                    opt_output_name += "_identity";
                    preconditioner_string = "identity";
                    break;
                }
            }
            break;
        }
        case OptimizationAlgorithm::full_space_composite_step: {
            opt_output_name = "full_space_composite_step";
            break;
        }
        case OptimizationAlgorithm::reduced_space_bfgs: {
            opt_output_name = "reduced_space_bfgs";
            descent_method = "Quasi-Newton Method";
            break;
        }
        case OptimizationAlgorithm::reduced_sqp: {
        }
        case OptimizationAlgorithm::reduced_space_newton: {
            opt_output_name = "reduced_space_newton";
            descent_method = "Newton-Krylov";
            break;
        }
    }
    opt_output_name = opt_output_name + "_" + "P" + std::to_string(level);

    // Output stream
    ROL::nullstream bhs; // outputs nothing
    std::filebuf filebuffer;
    if (this->mpi_rank == 0) filebuffer.open ("optimization_"+opt_output_name+"_"+std::to_string(nx_ffd-2)+".log", std::ios::out);
    if (this->mpi_rank == 2) filebuffer.open ("optimization.log", std::ios::out|std::ios::app);
    std::ostream ostr(&filebuffer);

    Teuchos::RCP<std::ostream> outStream;
    if (this->mpi_rank == 0 || this->mpi_rank == 2) outStream = ROL::makePtrFromRef(ostr);
    else if (this->mpi_rank == 1) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);


    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
    using VectorAdaptor = dealii::Rol::VectorAdaptor<DealiiVector>;
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix;
    Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    Assert(param.pde_type == param.PartialDifferentialEquation::navier_stokes, dealii::ExcNotImplemented());

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;

    n_vmult = 0;
    dRdW_form = 0;
    dRdW_mult = 0;
    dRdX_mult = 0;
    d2R_mult = 0;

    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr <Triangulation> grid = std::make_shared<Triangulation> (
        this->mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));


    unsigned int n_design_variables = 0;
    dealii::Point<dim> ffd_origin;
    std::array<double,dim> ffd_rectangle_lengths;
    std::array<unsigned int,dim> ffd_ndim_control_pts;
    std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim;
    if constexpr (dim == 2) {
    if (grid_type == GridType::cylinder) {
        ffd_origin = dealii::Point<dim> (-0.60,-0.51);
        ffd_rectangle_lengths = std::array<double,dim> {{1.0+0.2,1.0+0.02}};
    } else if (grid_type == GridType::naca0012) {
        //// Coordinates for NACA deall II grid
        // ffd_origin = dealii::Point<dim> (0.0,-0.061);
        // ffd_rectangle_lengths = std::array<double,dim> {{0.999,0.122}};
        //    ffd_origin = dealii::Point<dim> (-0.1,-0.1);
        ffd_origin = dealii::Point<dim> (-0.025,-0.035);
        //    ffd_rectangle_lengths = std::array<double,dim> {{0.6,0.2}};
        ffd_rectangle_lengths = std::array<double,dim> {{0.45,0.07}};
            //ffd_rectangle_lengths = std::array<double,dim> {{1.0,0.122}};
        }

        ffd_ndim_control_pts = {{nx_ffd,3}};

    }
    FreeFormDeformation<dim> ffd( ffd_origin, ffd_rectangle_lengths, ffd_ndim_control_pts);
    if constexpr (dim == 2) {
        // Vector of ijk indices and dimension.
        // Each entry in the vector points to a design variable's ijk ctl point and its acting dimension.
        for (unsigned int i_ctl = 0; i_ctl < ffd.n_control_pts; ++i_ctl) {

            const std::array<unsigned int,dim> ijk = ffd.global_to_grid ( i_ctl );
            for (unsigned int d_ffd = 0; d_ffd < dim; ++d_ffd) {

                if (   ijk[0] == 0 // Constrain first column of FFD points.
                    || ijk[0] == ffd_ndim_control_pts[0] - 1  // Constrain last column of FFD points.
                    || ijk[1] == 1 // Constrain middle row of FFD points.
                    || d_ffd == 0 // Constrain x-direction of FFD points.
                ) {
                    continue;
                }
                ++n_design_variables;
                ffd_design_variables_indices_dim.push_back(std::make_pair(i_ctl, d_ffd));
            }
        }
    }

    const dealii::IndexSet row_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, n_design_variables);
    dealii::IndexSet ghost_row_part(n_design_variables);
    ghost_row_part.add_range(0,n_design_variables);
    DealiiVector ffd_design_variables(row_part,ghost_row_part,MPI_COMM_WORLD);

    ffd.get_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    const auto initial_design_variables = ffd_design_variables;

    ffd_design_variables = initial_design_variables;
    ffd_design_variables.update_ghost_values();
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    // Initial optimization point
    grid->clear();
    dealii::GridGenerator::hyper_cube(*grid);

    if constexpr (dim == 2) {
    if (grid_type == GridType::cylinder) {
        //int n_cells_circle = 45;
        //int n_cells_radial = 45;
        //PHiLiP::Grids::cylinder(*grid, n_cells_circle, n_cells_radial);

        const dealii::Point<dim> center(0.0,0.0);
        const double        inner_radius = 0.5;
        const double        outer_radius = 40;
        const unsigned int  n_shells = 40;
        const double        skewness = 3.0;
        const unsigned int  n_cells_per_shell = 40;
        const bool          colorize = true;
        dealii::GridGenerator::concentric_hyper_shells(*grid, center, inner_radius, outer_radius, n_shells, skewness, n_cells_per_shell, colorize);

        grid->set_all_manifold_ids(0);
        grid->set_manifold(0, dealii::SphericalManifold<2>(center));

        // Assign BC
        for (auto cell = grid->begin_active(); cell != grid->end(); ++cell) {
            //if (!cell->is_locally_owned()) continue;
            for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) {
                    unsigned int current_id = cell->face(face)->boundary_id();
                    if (current_id == 0) {
                        cell->face(face)->set_boundary_id (1001); // Wall
                    } else if (current_id == 1) {
                        cell->face(face)->set_boundary_id (1004); // Farfield
                    } else {
                        std::abort();
                    }
                }
            }
        }

        }
    }

    // if (grid_type == GridType::naca0012) {
        //const double farfield_length = 20.0;
        //dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
        //airfoil_data.airfoil_type = "NACA";
        //airfoil_data.naca_id      = "0012";
        //airfoil_data.airfoil_length = 1.0;
        //airfoil_data.height         = farfield_length;
        //airfoil_data.length_b2      = farfield_length;
        //airfoil_data.incline_factor = 0.35;
        //airfoil_data.bias_factor    = 3.5; // default good enough?
        //airfoil_data.refinements    = 0;


        //const double multiplier = 1.0;
        //const int n_cells_leading_edge = 10 * multiplier;
        //const int n_cells_trailing_edge = 10 * multiplier;
        //const int n_cells_normal_to_airfoil = 10 * multiplier;
        //const int n_cells_downstream = 10 * multiplier;
        //airfoil_data.n_subdivision_x_0 = n_cells_leading_edge;
        //airfoil_data.n_subdivision_x_1 = n_cells_trailing_edge;
        //airfoil_data.n_subdivision_x_2 = n_cells_downstream;
        //airfoil_data.n_subdivision_y = n_cells_normal_to_airfoil;
        //airfoil_data.airfoil_sampling_factor = 3; // default 2
        //PHiLiP::Grids::naca_airfoil(*grid, airfoil_data);

        //std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, grid);

        // if (dim==2) {
            //std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh <dim, dim> ("naca0012.msh",1);
            //std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh <dim, dim> ("naca0012_hopw_ref"+std::to_string(level)+".msh",1);
            // std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh <dim, dim> ("naca0012_hopw_ref3.msh",1);
            // std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh <dim, dim> ("naca0012_hopw_ref1.msh", 1);
            // std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh <dim, dim> ("naca0012_hopw_ref1.msh", true, 1, false);
            // std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh <dim, dim> ("naca0012_hopw_ref2.msh", 1);
            // std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh <dim, dim> ("naca0012_hopw_ref4.msh", 1);
            //naca0012_mesh->refine_global();

            // using dealii Grid Generator
            std::shared_ptr<Triangulation> naca0012_mesh = std::make_shared<Triangulation> (
        #if dim!=1
            this->mpi_communicator
        #endif
            );

            dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
            airfoil_data.airfoil_type = "NACA";
            airfoil_data.naca_id      = "0012";
            airfoil_data.airfoil_length = all_param.flow_solver_param.airfoil_length;
            airfoil_data.height         = all_param.flow_solver_param.height;
            airfoil_data.length_b2      = all_param.flow_solver_param.length_b2;
            airfoil_data.incline_factor = all_param.flow_solver_param.incline_factor;
            airfoil_data.bias_factor    = all_param.flow_solver_param.bias_factor; 
            airfoil_data.refinements    = all_param.flow_solver_param.refinements;

            airfoil_data.n_subdivision_x_0 = all_param.flow_solver_param.n_subdivision_x_0;
            airfoil_data.n_subdivision_x_1 = all_param.flow_solver_param.n_subdivision_x_1;
            airfoil_data.n_subdivision_x_2 = all_param.flow_solver_param.n_subdivision_x_2;
            airfoil_data.n_subdivision_y = all_param.flow_solver_param.n_subdivision_y;
            airfoil_data.airfoil_sampling_factor = all_param.flow_solver_param.airfoil_sampling_factor; 

            dealii::GridGenerator::Airfoil::create_triangulation(*naca0012_mesh, airfoil_data);

                // Set boundary type and design type
            for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = naca0012_mesh->begin_active(); cell != naca0012_mesh->end(); ++cell) {
                for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
                    if (cell->face(face)->at_boundary()) {
                        unsigned int current_id = cell->face(face)->boundary_id();
                        if (current_id == 0 || current_id == 1 || current_id == 4 || current_id == 5) {
                            cell->face(face)->set_boundary_id (1005); // farfield
                        } else {
                            cell->face(face)->set_boundary_id (1001); // wall
                        }
                    }
                }
            }
            const int poly_degree = level;

   std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&all_param, &sub_all_param, poly_degree,flow_solver_param.max_poly_degree_for_adaptation, grid_degree, naca0012_mesh);
   std::shared_ptr < DGBase<dim, double> > sub_dg = DGFactory<dim,double>::create_discontinuous_galerkin(&sub_all_param, sub_poly_degree, sub_flow_solver_param.max_poly_degree_for_adaptation, sub_grid_degree, naca0012_mesh);
            // dg->set_high_order_grid(std::make_shared<HighOrderGrid<dim,double,dealii::parallel::distributed::Triangulation<2>>>(4, naca0012_mesh));
            // sub_dg->set_high_order_grid(std::make_shared<HighOrderGrid<dim,double,dealii::parallel::distributed::Triangulation<2>>>(4, naca0012_mesh));
        // }
        //if (dim==3) {
        //    std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh <dim, dim> ("naca0012_wing_unstructured_cutoff.msh", true, 1, false);
        //    dg->set_high_order_grid(naca0012_mesh);
        //}
    // }

// main flow solver set up
    if (ode_param.allocate_matrix_dRdW) {
        pcout << "Note: Allocating DG with AD matrix dRdW and dRdX only." << std::endl;
        dg->allocate_system(true,true,false); // FlowSolver only requires dRdW to be allocated
    } else {
        pcout << "Note: Allocating DG without AD matrices." << std::endl;
        dg->allocate_system(false,false,false);
    }

    // if(ode_param.ode_solver_type == Parameters::ODESolverParam::pod_galerkin_solver || ode_param.ode_solver_type == Parameters::ODESolverParam::pod_petrov_galerkin_solver){
    //     std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(dg);
    //     /*std::shared_ptr<ODE::ODESolverBase<dim, double>>*/ ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg, pod);
    // }
    // else{
        std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    // }
    // flow_solver_case->display_flow_solver_setup(dg);

    if(flow_solver_param.restart_computation_from_file == true) {
        if(dim == 1) {
            pcout << "Error: restart_computation_from_file is not possible for 1D. Set to false." << std::endl;
            std::abort();
        }

        if (flow_solver_param.steady_state == true) {
            pcout << "Error: Restart capability has not been fully implemented / tested for steady state computations." << std::endl;
            std::abort();
        }

        // Initialize solution from restart file
        pcout << "Initializing solution from restart file..." << std::flush;
        const std::string restart_filename_without_extension = get_restart_filename_without_extension(flow_solver_param.restart_file_index);
    #if PHILIP_DIM>1
        dg->triangulation->load(flow_solver_param.restart_files_directory_name + std::string("/") + restart_filename_without_extension);
        
        // Note: Future development with hp-capabilities, see section "Note on usage with DoFHandler with hp-capabilities"
        // ----- Ref: https://www.dealii.org/current/doxygen/deal.II/classparallel_1_1distributed_1_1SolutionTransfer.html
        dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
        solution_no_ghost.reinit(dg->locally_owned_dofs, this->mpi_communicator);
        dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
        solution_transfer.deserialize(solution_no_ghost);
        dg->solution = solution_no_ghost; //< assignment
        dg->sub_solution = solution_no_ghost; //< assignment
#endif
        pcout << "done." << std::endl;
    } else {
        // Initialize solution
        SetInitialCondition<dim,nstate,double>::set_initial_condition(InitialConditionFactory<dim, nstate, double>::create_InitialConditionFunction(&all_param), dg, &all_param);
    }
    dg->solution.update_ghost_values();
    dg->sub_solution.update_ghost_values();
    
    // Allocate ODE solver after initializing DG
    ode_solver->allocate_ode_system();

    
 // sub flow solver set up
    sub_dg->allocate_system();

    // if(sub_ode_param.ode_solver_type == Parameters::ODESolverParam::pod_galerkin_solver || sub_ode_param.ode_solver_type == Parameters::ODESolverParam::pod_petrov_galerkin_solver){
    //     std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(sub_dg);
    //     /*std::shared_ptr<ODE::ODESolverBase<dim, double>>*/ sub_ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(sub_dg, pod);
    // }
    // else{
        std::shared_ptr<ODE::ODESolverBase<dim, double>> sub_ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(sub_dg);
    // }

    pcout << "Initializing sub solution with initial condition function... " << std::flush;
    SetInitialCondition<dim,1,double>::set_initial_condition(InitialConditionFactory<dim, 1, double>::create_InitialConditionFunction(&sub_all_param), sub_dg, &sub_all_param);

    sub_dg->solution.update_ghost_values();
    pcout << "done." << std::endl;
    sub_ode_solver->allocate_ode_system();



#ifndef CREATE_RST
    DealiiVector target_solution;
    if (OptimizationProblemType::inverse_pressure_design == optimization_problem_type) {
        const std::string restart_filename_without_extension = get_restart_filename_without_extension(99299);
        dg->triangulation->load(std::string("./") + restart_filename_without_extension);
        dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);

        dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
        solution_no_ghost.reinit(dg->locally_owned_dofs, this->mpi_communicator);
        solution_transfer.deserialize(solution_no_ghost);

        dg->solution = solution_no_ghost; //< assignment
        dg->solution.update_ghost_values();
        target_solution = dg->solution;
    }
    // TargetWallPressure<dim,nstate,double> target_wall_pressure_functional(dg, target_solution);
    // TargetWallPressure<dim,dim+2,double> target_wall_pressure_functional(dg, target_solution);
#endif

// Solve sub and main flow solvers, steady state
{
        //----------------------------------------------------
        // Steady-state solution
        //----------------------------------------------------
        using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
        if(flow_solver_param.steady_state_polynomial_ramping && (ode_param.ode_solver_type != ODEEnum::pod_galerkin_solver && ode_param.ode_solver_type != ODEEnum::pod_petrov_galerkin_solver)) {
            ode_solver->initialize_steady_polynomial_ramping(poly_degree);
        }

        if(sub_dg){
            pcout << "Start calculation for the sub ODE solver..." << std::endl;
            sub_ode_solver->steady_state();
            // sub_flow_solver_case->steady_state_postprocessing(sub_dg);
            pcout << "End calculation for the sub ODE solver..." << std::endl;

            if(sub_dg->max_degree!=dg->max_degree){
                pcout << "Detect different polynomial degree between sub dg (poly_degree = " << sub_dg->max_degree << ") and main dg (poly_degree = " << dg->max_degree << ")..." << std::endl;
                pcout << "Interpolate solution from current polynomial degree " << sub_dg->max_degree << " to desired polynomial degree " << dg->max_degree << " ..." << std::endl;
                sub_dg->output_results_vtk(8888);
                sub_ode_solver->interpolate_solution_polynomial_degree(dg->max_degree);
                sub_dg->output_results_vtk(9999);
            }

            pcout << "Transfer the solution from sub dg to main dg..." << std::endl;
            dg->import_sub_solution(sub_dg);
        }else{
            pcout << "No calculation for the sub ODE solver..." << std::endl;
        }

        pcout << "Start calculation for the main ODE solver..." << std::endl;
        
        ode_solver->steady_state();
        // flow_solver_case->steady_state_postprocessing(dg);
        
        const bool use_isotropic_mesh_adaptation = (all_param.mesh_adaptation_param.total_mesh_adaptation_cycles > 0) 
                                        && (all_param.mesh_adaptation_param.mesh_adaptation_type != Parameters::MeshAdaptationParam::MeshAdaptationType::anisotropic_adaptation);
        
        if(use_isotropic_mesh_adaptation)
        {
            perform_steady_state_mesh_adaptation(dg, ode_solver);
        }

}

    /// Reset to initial_grid
    DealiiVector des_var_sim = dg->solution;
    DealiiVector des_var_ctl = initial_design_variables;
    DealiiVector des_var_adj = dg->dual;
    des_var_adj.add(0.1);

    const bool has_ownership = false;
    VectorAdaptor des_var_sim_rol(Teuchos::rcp(&des_var_sim, has_ownership));
    VectorAdaptor des_var_ctl_rol(Teuchos::rcp(&des_var_ctl, has_ownership));
    VectorAdaptor des_var_adj_rol(Teuchos::rcp(&des_var_adj, has_ownership));

    ROL::Ptr<ROL::Vector<double>> simulation_variables = ROL::makePtr<VectorAdaptor>(des_var_sim_rol);
    ROL::Ptr<ROL::Vector<double>> control_variables = ROL::makePtr<VectorAdaptor>(des_var_ctl_rol);
    auto des_var_p = ROL::makePtr<ROL::Vector_SimOpt<double>>(simulation_variables, control_variables);


    ROL::OptimizationProblem<double> opt;
    Teuchos::ParameterList parlist;
    LiftDragFunctional<dim,nstate,double> lift_functional( dg, LiftDragFunctional<dim,nstate,double>::Functional_types::lift );
    LiftDragFunctional<dim,nstate,double> drag_functional( dg, LiftDragFunctional<dim,nstate,double>::Functional_types::total_drag );
    // LiftDragFunctional<dim,nstate,double> pressure_drag_functional( dg, LiftDragFunctional<dim,nstate,double>::Functional_types::pressure_drag );
    ZMomentFunctional<dim,nstate,double> moment_functional( dg, {0.25, 0.0} );
    GeometricVolume<dim,nstate,double> volume_functional( dg );


    std::ofstream outfile_pressure_drag;
    outfile_pressure_drag.open("pressure_drag.dat");
    std::ofstream outfile_total_drag;
    outfile_total_drag.open("total_drag.dat");
    std::ofstream outfile_acoustic;
    outfile_acoustic.open("sound_level.dat");

    dealii::Point<dim,double> extraction_point;
    if constexpr(dim==2){
            extraction_point[0] = 0.36;
            extraction_point[1] = 0.00546019;
        } else if constexpr(dim==3){
            extraction_point[0] = 0.36;
            extraction_point[1] = 0.00546019;
            extraction_point[2] = 0;
        }
        int number_of_sampling = 200;

    ExtractionFunctional<dim,nstate,double,Triangulation> boundary_layer_extraction(dg, extraction_point, number_of_sampling);

    dealii::Point<3,double> observer_coord_ref;
    observer_coord_ref[0] = 0.0;
    observer_coord_ref[1] = 0.0;
    observer_coord_ref[2] = 2.0;

    AmietModelFunctional<dim,nstate,double,Triangulation> acoustic_functional = AmietModelFunctional<dim,nstate,double,Triangulation>(dg,boundary_layer_extraction,observer_coord_ref);

    std::cout << " Current lift = " << lift_functional.evaluate_functional()
              << ". Current drag = " << drag_functional.evaluate_functional()
            //   << ". Current pressure drag = " << pressure_drag_functional.evaluate_functional()
              << ". Current OASPL = " << acoustic_functional.evaluate_functional(true,true,false)
              << ". Current Z-moment = " << moment_functional.evaluate_functional()
              << std::endl;

    double lift_target;
    double volume_target;
    double moment_target;

    if (optimization_problem_type == OptimizationProblemType::drag_minimization) {
        lift_target = lift_functional.evaluate_functional() * 1.0;
        volume_target = volume_functional.evaluate_functional() * 1.0;
        moment_target = moment_functional.evaluate_functional() * 1.0;
    } else if (optimization_problem_type == OptimizationProblemType::lift_target) {
        lift_target = lift_functional.evaluate_functional() * 2.0;
        volume_target = volume_functional.evaluate_functional() * 1.0;
        moment_target = std::abs(moment_functional.evaluate_functional() * 1.0) * 9.0;
    } else if (optimization_problem_type == OptimizationProblemType::inverse_pressure_design) {
        lift_target = lift_functional.evaluate_functional() * 1.1;
        volume_target = volume_functional.evaluate_functional() * 0.9;
        moment_target = moment_functional.evaluate_functional() * 1.0;
    }
    if (grid_type == GridType::cylinder) {
        lift_target = 0.3;
        volume_target = 0.08;
        moment_target = 0.04;
    }



/// WAS COMMENTED OUT BEFORE (Zineb May 30)
//    const std::string restart_filename_without_extension = get_restart_filename_without_extension(flow_solver_param.restart_file_index);
//#if PHILIP_DIM>1
//    dg->triangulation->load(flow_solver_param.restart_files_directory_name + std::string("/") + restart_filename_without_extension);
//    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
//    solution_transfer.deserialize(solution_no_ghost);
//#endif


    ffd.output_ffd_vtu(8999);

    // std::shared_ptr<BaseParameterization<dim>> design_parameterization = 
    std::shared_ptr<FreeFormDeformationParameterization<dim>> design_parameterization = 
                        std::make_shared<FreeFormDeformationParameterization<dim>>(dg->high_order_grid, ffd, ffd_design_variables_indices_dim);

    auto flow_constraints  = ROL::makePtr<FlowConstraintsPhysicsModel<dim>>(sub_dg,dg,design_parameterization);
    // auto flow_constraints  = ROL::makePtr<FlowConstraints<dim>>(dg,design_parameterization);
    std::shared_ptr<MatrixType> precomputed_dXvdXp = std::make_shared<MatrixType> ();
    precomputed_dXvdXp->reinit(flow_constraints->dXvdXp);
    precomputed_dXvdXp->copy_from(flow_constraints->dXvdXp);

    // WAS COMMENTED OUT BEFORE (Zineb May 30)
    // auto volume_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( volume_functional, design_parameterization, precomputed_dXvdXp );
    // auto moment_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( moment_functional, design_parameterization, precomputed_dXvdXp );


    auto volume_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( volume_functional, design_parameterization, precomputed_dXvdXp );
    auto moment_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( moment_functional, design_parameterization, precomputed_dXvdXp );
	auto constraint1 = volume_objective;

    const double constraint1_lower_bound_dx = USE_VOLUME_CONSTRAINT ? -1e-4 : -ROL::ROL_INF<double>();
    const double constraint1_upper_bound_dx = USE_VOLUME_CONSTRAINT ? 1e-4  : ROL::ROL_INF<double>();

	auto constraint2 = moment_objective;
    
    /// WAS COMMENTED OUT BEFORE (Zineb May 30)  
//#ifdef CREATE_RST
//    const double constraint2_lower_bound_dx = -1e-3;
//    const double constraint2_upper_bound_dx = 1e-3;
//#else
//    const double constraint2_lower_bound_dx = -ROL::ROL_INF<double>();
//    const double constraint2_upper_bound_dx = ROL::ROL_INF<double>();
//#endif



    const double constraint2_lower_bound_dx = USE_MOMENT_CONSTRAINT ? -1e-3 : -ROL::ROL_INF<double>();
    const double constraint2_upper_bound_dx = USE_MOMENT_CONSTRAINT ?  1e-3 :  ROL::ROL_INF<double>();


    /// WAS COMMENTED OUT BEFORE (Zineb May 30)
    //ROL::Ptr<ROL::Vector<double>> drag_adjoint = ROL::makePtr<VectorAdaptor>(des_var_adj_rol);
    // int flow_constraints_check_error
    //     = check_flow_constraints<dim,nstate>( nx_ffd,
    //                                            flow_constraints,
    //                                            simulation_variables,
    //                                            control_variables,
    //                                            drag_adjoint);
    // (void) flow_constraints_check_error;


    ROL::Ptr<ROL::Objective_SimOpt<double>> objective;
    std::vector<ROL::Ptr<ROL::Objective_SimOpt<double>>> nonlinear_inequalities_as_objectives {constraint1, constraint2};//, constraint2};
    std::vector<double> nonlinear_inequality_targets {volume_target, moment_target};
    std::vector<double> constraint_lower_bound_dx {constraint1_lower_bound_dx, constraint2_lower_bound_dx};
    std::vector<double> constraint_upper_bound_dx {constraint1_upper_bound_dx, constraint2_upper_bound_dx};

    if (optimization_problem_type == OptimizationProblemType::drag_minimization) {
        // Objective
        auto drag_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( drag_functional, design_parameterization, precomputed_dXvdXp );
        // auto drag_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,dim+2>>( drag_functional, design_parameterization, precomputed_dXvdXp );
        // objective = drag_objective;

        auto acoustic_objective = ROL::makePtr<ROLAcousticObjectiveSimOpt<dim,nstate>>( dg, design_parameterization, precomputed_dXvdXp );
        objective = acoustic_objective;

        // Additional lift constraint
        // auto lift_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( lift_functional, design_parameterization, precomputed_dXvdXp );
        auto lift_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( lift_functional, design_parameterization, precomputed_dXvdXp );


        nonlinear_inequalities_as_objectives.push_back(lift_objective);
        nonlinear_inequality_targets.push_back(lift_target);
        if (USE_LIFT_CONSTRAINT) {
        constraint_lower_bound_dx.push_back(-lift_target*0.05);
        constraint_upper_bound_dx.push_back(ROL::ROL_INF<double>());
    } else {
            constraint_lower_bound_dx.push_back(-ROL::ROL_INF<double>());
            constraint_upper_bound_dx.push_back(ROL::ROL_INF<double>());
        }

    } else if (optimization_problem_type == OptimizationProblemType::lift_target) {

        // Constraint lift-target minimization objective
        // auto lift_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( lift_functional, design_parameterization, precomputed_dXvdXp );
        auto lift_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( lift_functional, design_parameterization, precomputed_dXvdXp );
        auto lift_target_constraint = ROL::makePtr<PHiLiP::ConstraintFromObjective_SimOpt<double>>( lift_objective, lift_target );
        const ROL::Ptr<ROL::SingletonVector<double>> lift_constraint_dual = ROL::makePtr<ROL::SingletonVector<double>> (0.0);
        const ROL::Ptr<ROL::SingletonVector<double>> lift_constraint_value = ROL::makePtr<ROL::SingletonVector<double>> (1.0);
        const double penaltyParameter = 1.0;

        auto lift_target_quadratic_objective = ROL::makePtr<ROL::QuadraticPenalty_SimOpt<double>>(lift_target_constraint,
                                                                                                  *lift_constraint_dual,
                                                                                                  penaltyParameter,
                                                                                                  *simulation_variables,
                                                                                                  *control_variables,
                                                                                                  *lift_constraint_value);
        objective = lift_target_quadratic_objective;

    } else if (optimization_problem_type == OptimizationProblemType::inverse_pressure_design) {

        // Additional lift constraint
        auto lift_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( lift_functional, design_parameterization, precomputed_dXvdXp );
        // auto lift_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,dim+2>>( lift_functional, design_parameterization, precomputed_dXvdXp );
        nonlinear_inequalities_as_objectives.push_back(lift_objective);
        nonlinear_inequality_targets.push_back(lift_target);
        constraint_lower_bound_dx.push_back(-lift_target*0.005);
        //constraint_lower_bound_dx.push_back(-ROL::ROL_INF<double>());
        constraint_upper_bound_dx.push_back(ROL::ROL_INF<double>());

// #ifndef CREATE_RST
//         auto pressure_obj = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( target_wall_pressure_functional, design_parameterization, &(flow_constraints->dXvdXp) );
//         objective = pressure_obj;
// #endif
    }
    //for (auto& lower : constraint_lower_bound_dx) {
    //    lower = -ROL::ROL_INF<double>();
    //}
    //for (auto& upper : constraint_upper_bound_dx) {
    //    upper = ROL::ROL_INF<double>();
    //}

    double tol = 0.0;
    std::cout << "Objective value= " << objective->value(*simulation_variables, *control_variables, tol) << std::endl;

    dg->output_results_vtk(9999);

    double timing_start, timing_end;
    timing_start = MPI_Wtime();
    // Verbosity setting
    parlist.sublist("General").set("Print Verbosity", 1);

    //parlist.sublist("Status Test").set("Gradient Tolerance", 1e-9);
    parlist.sublist("Status Test").set("Gradient Tolerance", GRAD_CONVERGENCE);
    parlist.sublist("Status Test").set("Iteration Limit", max_design_cycle);

    parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
    parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",3e-1); // Might be needed for p2 BFGS
    parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
    parlist.sublist("Step").sublist("Line Search").set("Function Evaluation Limit",LINESEARCH_MAX_ITER); // 0.5^30 ~  1e-10
    parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").get("Backtracking Rate", BACKTRACKING_RATE);
    parlist.sublist("Step").sublist("Line Search").set("Accept Linesearch Minimizer",true);//false);
    parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type",line_search_method);
    parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type",line_search_curvature);


    parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
    //parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory SR1");
    //parlist.sublist("General").sublist("Secant").set("Maximum Storage",(int)n_design_variables);
    parlist.sublist("General").sublist("Secant").set("Maximum Storage", 200);

    parlist.sublist("Full Space").set("Preconditioner",preconditioner_string);

    ROL::Ptr< const ROL::AlgorithmState <double> > algo_state;


    switch (opt_type) {
        case OptimizationAlgorithm::full_space_composite_step: {
            // Full space problem
            auto dual_sim_p = simulation_variables->clone();
            //opt = ROL::OptimizationProblem<double> ( objective, des_var_p, flow_constraints, dual_sim_p );
            opt = ROL::OptimizationProblem<double> ( objective, des_var_p, flow_constraints, dual_sim_p );

            // Set parameters.

            parlist.sublist("Step").set("Type","Composite Step");
            ROL::ParameterList& steplist = parlist.sublist("Step").sublist("Composite Step");
            steplist.set("Initial Radius", 1e2);
            steplist.set("Use Constraint Hessian", true); // default is true
            steplist.set("Output Level", 1);

            steplist.sublist("Optimality System Solver").set("Nominal Relative Tolerance", 1e-8); // default 1e-8
            steplist.sublist("Optimality System Solver").set("Fix Tolerance", true);
            const int cg_iteration_limit = 200;
            steplist.sublist("Tangential Subproblem Solver").set("Iteration Limit", cg_iteration_limit);
            steplist.sublist("Tangential Subproblem Solver").set("Relative Tolerance", 1e-2);

            *outStream << "Starting optimization with " << n_design_variables << "..." << std::endl;
            ROL::OptimizationSolver<double> solver( opt, parlist );
            solver.solve( *outStream );
            algo_state = solver.getAlgorithmState();

            break;
        }
        case OptimizationAlgorithm::reduced_space_bfgs:
            parlist.sublist("General").sublist("Secant").set("Use as Hessian", true);
            [[fallthrough]];
        case OptimizationAlgorithm::reduced_space_newton: {
            if (opt_type == OptimizationAlgorithm::reduced_space_newton) {
                parlist.sublist("General").sublist("Secant").set("Use as Hessian", false);
            }
            *outStream << "Starting optimization with " << n_design_variables << "..." << std::endl;

            const bool is_reduced_space = true;
            ROL::Ptr<ROL::Vector<double>>                       design_variables               = getDesignVariables(simulation_variables, control_variables, is_reduced_space);
            ROL::Ptr<ROL::BoundConstraint<double>>              design_bounds                  = getDesignBoundConstraint(simulation_variables, control_variables, is_reduced_space);
            ROL::Ptr<ROL::Objective<double>>                    reduced_drag_objective         = getObjective(objective, flow_constraints, simulation_variables, control_variables, is_reduced_space);
            std::vector<ROL::Ptr<ROL::Constraint<double>>>      reduced_inequality_constraints = getInequalityConstraint(nonlinear_inequalities_as_objectives, flow_constraints, simulation_variables, control_variables, is_reduced_space);
            std::vector<ROL::Ptr<ROL::Vector<double>>>          dual_inequality                = getInequalityMultiplier(nonlinear_inequality_targets);
            std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> inequality_bounds              = getSlackBoundConstraint(nonlinear_inequality_targets, constraint_lower_bound_dx, constraint_upper_bound_dx);

            opt = ROL::OptimizationProblem<double> ( reduced_drag_objective, design_variables, design_bounds,
                                                     reduced_inequality_constraints, dual_inequality, inequality_bounds);
            ROL::EProblem problem_type_opt = opt.getProblemType();
            ROL::EProblem problem_type = ROL::TYPE_EB;
            if (problem_type_opt != problem_type) std::abort();

            parlist.sublist("Step").sublist("Primal Dual Active Set").set("Iteration Limit",PDAS_MAX_ITER);
            parlist.sublist("General").sublist("Krylov").set("Absolute Tolerance", LINEAR_SOLVER_ABS_TOL);
            parlist.sublist("General").sublist("Krylov").set("Relative Tolerance", LINEAR_SOLVER_REL_TOL);
            parlist.sublist("General").sublist("Krylov").set("Iteration Limit", LINEAR_SOLVER_MAX_ITS);
            parlist.sublist("General").sublist("Krylov").set("Use Initial Guess", true);

            parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
            parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",3e-1); // Might be needed for p2 BFGS
            parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
            parlist.sublist("Step").sublist("Line Search").set("Accept Linesearch Minimizer",true);//false);
            parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type",line_search_method);
            parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type",line_search_curvature);


            // This step transforms the inequality into equality + slack variables with box constraints.
            auto x      = opt.getSolutionVector();
            auto g      = x->dual().clone();
            auto l      = opt.getMultiplierVector();
            auto c      = l->dual().clone();
            auto obj    = opt.getObjective();
            auto con    = opt.getConstraint();
            auto bnd    = opt.getBoundConstraint();

            for (auto &constraint_dual : dual_inequality) {
                constraint_dual->zero();
            }

            auto pdas_step = ROL::makePtr<PHiLiP::PrimalDualActiveSetStep<double>>(parlist);
            auto status_test = ROL::makePtr<ROL::StatusTest<double>>(parlist);
            const bool printHeader = true;

            const ROL::Ptr<ROL::Algorithm<double>> algorithm = ROL::makePtr<ROL::Algorithm<double>>( pdas_step, status_test, printHeader );
            algorithm->run(*x, *g, *l, *c, *obj, *con, *bnd, true, *outStream);
            algo_state = algorithm->getState();

            break;
        } case OptimizationAlgorithm::reduced_sqp: {
            [[fallthrough]];

        //     // Reduced space problem
        //     const bool storage = true;
        //     const bool useFDHessian = false;
        //     // Create reduced-objective by combining objective with PDE constraints.
        //     ROL::Ptr<ROL::Vector<double>> drag_adjoint = ROL::makePtr<VectorAdaptor>(des_var_adj_rol);
        //     auto reduced_drag_objective = ROL::makePtr<ROL::Reduced_Objective_SimOpt_FailSafe<double>>( objective, flow_constraints, simulation_variables, control_variables, drag_adjoint, storage, useFDHessian);

        //     // Create reduced-constraint by combining lift-objective with PDE constraints.
        //     ROL::Ptr<ROL::SimController<double> > stateStore = ROL::makePtr<ROL::SimController<double>>();
        //     ROL::Ptr<ROL::Vector<double>> lift_adjoint = drag_adjoint->clone();
        //     ROL::Ptr<ROL::SingletonVector<double>> lift_constraint_residual_rol_p = ROL::makePtr<ROL::SingletonVector<double>> (0.0);
        //     //auto reduced_lift_constraint = ROL::makePtr<ROL::Reduced_Constraint_SimOpt_FailSafe<double>>(
        //     //    lift_constraint, flow_constraints, stateStore,
        //     //    simulation_variables, control_variables, lift_adjoint, lift_constraint_residual_rol_p,
        //     //    storage, useFDHessian);

        //     // Create reduced-objective by combining objective with PDE constraints.
        //     auto reduced_lift_objective = ROL::makePtr<ROL::Reduced_Objective_SimOpt_FailSafe<double>>( constraint1, flow_constraints, simulation_variables, control_variables, lift_adjoint, storage, useFDHessian);
        //     std::cout << " Converting reduced lift objective into reduced_lift_constraint " << std::endl;
        //     ROL::Ptr<ROL::Constraint<double>> reduced_lift_constraint = ROL::makePtr<ROL::ConstraintFromObjective<double>> (reduced_lift_objective, lift_target);

        //     std::cout << " Starting check_reduced_constraint " << std::endl;
        //     lift_constraint_residual_rol_p->setScalar(1.0);
        //     //(void) check_reduced_constraint<dim,nstate>( nx_ffd, reduced_lift_constraint, control_variables, lift_constraint_residual_rol_p);

        //     // Run the algorithm
        //     parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
        //     //auto reduced_sqp_step = ROL::makePtr<ROL::SequentialQuadraticProgrammingStep<double>>(parlist);
        //     auto reduced_sqp_step = ROL::makePtr<ROL::InteriorPointStep<double>>(parlist);

        //     auto status_test = ROL::makePtr<ROL::StatusTest<double>>(parlist);
        //     const bool printHeader = false;//true;
        //     ROL::Algorithm<double> algorithm(reduced_sqp_step, status_test, printHeader);
        //     algorithm.run(*control_variables, *lift_constraint_residual_rol_p, *reduced_drag_objective, *reduced_lift_constraint, false, *outStream);
        //     algo_state = algorithm.getState();
        //     break;
        } case OptimizationAlgorithm::full_space_birosghattas: {

            *outStream << "Starting optimization with " << n_design_variables << " control variables..." << std::endl;

            parlist.sublist("General").sublist("Secant").set("Use as Hessian", false);
            const bool is_reduced_space = false;
            ROL::Ptr<ROL::Vector<double>>                       design_variables               = getDesignVariables(simulation_variables, control_variables, is_reduced_space);
            ROL::Ptr<ROL::BoundConstraint<double>>              design_bounds                  = getDesignBoundConstraint(simulation_variables, control_variables, is_reduced_space);
            ROL::Ptr<ROL::Objective<double>>                    drag_objective_simopt          = getObjective(objective, flow_constraints, simulation_variables, control_variables, is_reduced_space);
            std::vector<ROL::Ptr<ROL::Constraint<double>>>      inequality_constraints         = getInequalityConstraint(nonlinear_inequalities_as_objectives, flow_constraints, simulation_variables, control_variables, is_reduced_space);
            std::vector<ROL::Ptr<ROL::Vector<double>>>          dual_inequality                = getInequalityMultiplier(nonlinear_inequality_targets);
            std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> inequality_bounds              = getSlackBoundConstraint(nonlinear_inequality_targets, constraint_lower_bound_dx, constraint_upper_bound_dx);

            ROL::Ptr<ROL::Constraint<double>>                   equality_constraints           = flow_constraints;
            ROL::Ptr<ROL::Vector<double>>                       dual_equality                  = simulation_variables->clone();
            dual_equality->zero();

            opt = ROL::OptimizationProblem<double> ( drag_objective_simopt, design_variables, design_bounds,
                                                     equality_constraints, dual_equality,
                                                     inequality_constraints, dual_inequality, inequality_bounds);
            ROL::EProblem problem_type_opt = opt.getProblemType();
            ROL::EProblem problem_type = ROL::TYPE_EB;
            if (problem_type_opt != problem_type) std::abort();

            parlist.sublist("Step").sublist("Primal Dual Active Set").set("Iteration Limit",PDAS_MAX_ITER);
            parlist.sublist("General").sublist("Secant").set("Use as Preconditioner", true);
            parlist.sublist("General").sublist("Krylov").set("Absolute Tolerance", LINEAR_SOLVER_ABS_TOL);
            parlist.sublist("General").sublist("Krylov").set("Relative Tolerance", LINEAR_SOLVER_REL_TOL);
            parlist.sublist("General").sublist("Krylov").set("Iteration Limit", LINEAR_SOLVER_MAX_ITS);
            parlist.sublist("General").sublist("Krylov").set("Use Initial Guess", true);

            parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
            parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",3e-1); // Might be needed for p2 BFGS
            parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
            parlist.sublist("Step").sublist("Line Search").set("Function Evaluation Limit",LINESEARCH_MAX_ITER); // 0.5^30 ~  1e-10
            parlist.sublist("Step").sublist("Line Search").set("Accept Linesearch Minimizer",true);//false);
            parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type",line_search_method);
            parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type",line_search_curvature);


            // This step transforms the inequality into equality + slack variables with box constraints.
            auto x      = opt.getSolutionVector();
            auto g      = x->dual().clone();
            auto l      = opt.getMultiplierVector();
            auto c      = l->dual().clone();
            auto obj    = opt.getObjective();
            auto con    = opt.getConstraint();
            auto bnd    = opt.getBoundConstraint();

            for (auto &constraint_dual : dual_inequality) {
                constraint_dual->zero();
            }

            auto pdas_step = ROL::makePtr<PHiLiP::PrimalDualActiveSetStep<double>>(parlist);
            auto status_test = ROL::makePtr<ROL::StatusTest<double>>(parlist);
            const bool printHeader = true;

            const ROL::Ptr<ROL::Algorithm<double>> algorithm = ROL::makePtr<ROL::Algorithm<double>>( pdas_step, status_test, printHeader );
            algorithm->run(*x, *g, *l, *c, *obj, *con, *bnd, true, *outStream);
            algo_state = algorithm->getState();

            break;
        }
    }
    std::cout << " Current lift = " << lift_functional.evaluate_functional()
              << " Current OASPL = " << acoustic_functional.evaluate_functional(true,true,false)
              << ". Current drag = " << drag_functional.evaluate_functional()
            //   << ". Current pressure drag = " << pressure_drag_functional.evaluate_functional()
              << ". Drag with quadratic lift penalty = " << objective->value(*simulation_variables, *control_variables, tol);
    static int resulting_optimization = 5000;
    std::cout << "Outputting final grid resulting_optimization: " << resulting_optimization << std::endl;
    dg->output_results_vtk(resulting_optimization++);

    outfile_pressure_drag.close();
    outfile_total_drag.close();
    outfile_acoustic.close();


    timing_end = MPI_Wtime();
    *outStream << "The process took " << timing_end - timing_start << " seconds to run." << std::endl;

    *outStream << "Total n_vmult for algorithm " << n_vmult << std::endl;

    test_error += algo_state->statusFlag;

    filebuffer.close();

    if (opt_type != OptimizationAlgorithm::full_space_birosghattas) break;
    }
    }

    return test_error;
}

#if PHILIP_DIM==2
    template class AeroAcousticOptimization2D <PHILIP_DIM,PHILIP_DIM+2>;
    template class AeroAcousticOptimization2D <PHILIP_DIM,PHILIP_DIM+3>;
#endif

} // Tests namespace
} // PHiLiP namespace



