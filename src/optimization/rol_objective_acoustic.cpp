#include "rol_to_dealii_vector.hpp"
#include "rol_objective_acoustic.hpp"

#include <deal.II/optimization/rol/vector_adaptor.h>

#include "global_counter.hpp"
#include "functional/extraction_functional.hpp"
// #include "functional/amiet_model.hpp"
// #include "functional/acoustic_adjoint.hpp"

namespace PHiLiP {

using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;

template <int dim, int nstate>
ROLAcousticObjectiveSimOpt<dim,nstate>::ROLAcousticObjectiveSimOpt(
    std::shared_ptr<DGBase<dim,double,Triangulation>> dg_input,
    // std::shared_ptr<BaseParameterization<dim>> _design_parameterization,
    std::shared_ptr<FreeFormDeformationParameterization<dim>> _design_parameterization,
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> precomputed_dXvdXp)
    : design_parameterization(_design_parameterization)
    , dg(dg_input)
{
    //Initial location of extraction point
    dealii::Point<dim,double> initial_extraction_point;
    if constexpr(dim==2){
            initial_extraction_point[0] = dg->all_parameters->boundary_layer_extraction_param.extraction_point_x;
            initial_extraction_point[1] = dg->all_parameters->boundary_layer_extraction_param.extraction_point_y;
        } else if constexpr(dim==3){
            initial_extraction_point[0] = dg->all_parameters->boundary_layer_extraction_param.extraction_point_x;
            initial_extraction_point[1] = dg->all_parameters->boundary_layer_extraction_param.extraction_point_y;
            initial_extraction_point[2] = 0;
        }

    std::cout << "Initial Location " << initial_extraction_point[0] << ",,," << initial_extraction_point[1] << std::endl;

    // this->dg = dg_input;

    int number_of_sampling = this->dg->all_parameters->boundary_layer_extraction_param.number_of_sampling;
    dealii::Point<3,double> observer_coord_ref;
    observer_coord_ref[0] = dg->all_parameters->amiet_param.observer_coord_ref_x;
    observer_coord_ref[1] = dg->all_parameters->amiet_param.observer_coord_ref_y;
    observer_coord_ref[2] = dg->all_parameters->amiet_param.observer_coord_ref_z;

    ExtractionFunctional<dim,nstate,double,Triangulation> boundary_layer_extraction(dg, initial_extraction_point, number_of_sampling);
    this->functional = std::make_shared<AmietModelFunctional<dim,nstate,double,Triangulation>>(dg,boundary_layer_extraction,observer_coord_ref);

     Assert(this->dg->high_order_grid == design_parameterization->high_order_grid, 
          dealii::ExcMessage("Functional and DesignParameterization do not point to the same high order grid."));
    design_parameterization->initialize_design_variables(design_var);
    const unsigned int n_design_variables = design_parameterization->get_number_of_design_variables();
    
    if (precomputed_dXvdXp) {
        if (precomputed_dXvdXp->m() == this->dg->high_order_grid->volume_nodes.size() && precomputed_dXvdXp->n() == n_design_variables) {
            dXvdXp.copy_from(*precomputed_dXvdXp);
        }
    } else {
        design_parameterization->compute_dXv_dXp(dXvdXp);
    }

}


template <int dim, int nstate>
void ROLAcousticObjectiveSimOpt<dim,nstate>::update(
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    bool /*flag*/, int /*iter*/)
{
    std::cout << "Updating" << std::endl;
    this->functional->set_state(ROL_vector_to_dealii_vector_reference(des_var_sim));

    design_var =  ROL_vector_to_dealii_vector_reference(des_var_ctl);
    design_parameterization->update_mesh_from_design_variables(dXvdXp, design_var);

     //Update location of extraction point
    dealii::Point<dim,double> initial_extraction_point;
    if constexpr(dim==2){
            initial_extraction_point[0] = dg->all_parameters->boundary_layer_extraction_param.extraction_point_x;
            initial_extraction_point[1] = dg->all_parameters->boundary_layer_extraction_param.extraction_point_y;
        } else if constexpr(dim==3){
            initial_extraction_point[0] = dg->all_parameters->boundary_layer_extraction_param.extraction_point_x;
            initial_extraction_point[1] = dg->all_parameters->boundary_layer_extraction_param.extraction_point_y;
            initial_extraction_point[2] = 0;
        }

    this->new_extraction_point = design_parameterization->ffd_new_point_location(initial_extraction_point);

    std::cout << "NEW LOCATION " << this->new_extraction_point[0] << ",,," << this->new_extraction_point[1] << std::endl;

    // this->dg = dg_input;

    int number_of_sampling = this->dg->all_parameters->boundary_layer_extraction_param.number_of_sampling;
    dealii::Point<3,double> observer_coord_ref;
    observer_coord_ref[0] = dg->all_parameters->amiet_param.observer_coord_ref_x;
    observer_coord_ref[1] = dg->all_parameters->amiet_param.observer_coord_ref_y;
    observer_coord_ref[2] = dg->all_parameters->amiet_param.observer_coord_ref_z;

    ExtractionFunctional<dim,nstate,double,Triangulation> boundary_layer_extraction(dg, new_extraction_point, number_of_sampling);
    this->functional->boundary_layer_extraction = std::make_shared<ExtractionFunctional<dim,nstate,double,Triangulation>> (boundary_layer_extraction);
    // this->functional->evaluate_functional(true,true,false);

    // function = acoustic_functional;
}


template <int dim, int nstate>
double ROLAcousticObjectiveSimOpt<dim,nstate>::value(
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &tol )
{
    // Tolerance tends to not be used except in the case of a reduced objective function.
    // In that scenario, tol is the constraint norm.
    // If the flow has not converged (>1e-5 or is nan), simply return a high functional.
    // This is likely happening in the linesearch while optimizing in the reduced-space.
    std::cout << "Computing value" << std::endl;
    if (tol > 1e-5 || std::isnan(tol)) return 1e200;
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = true;
    const bool compute_dIdX = true;//false;
    const bool compute_d2I = false;
    return this->functional->evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );


}

template <int dim, int nstate>
void ROLAcousticObjectiveSimOpt<dim,nstate>::gradient_1(
    ROL::Vector<double> &gradient_sim,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
{
    std::cout << "Computing gradient 1" << std::endl;
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = true;
    const bool compute_dIdX = true;//false;
    const bool compute_d2I = false;
    this->functional->evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
    auto &dIdW = ROL_vector_to_dealii_vector_reference(gradient_sim);
    dIdW = this->functional->dIdw;
}

template <int dim, int nstate>
void ROLAcousticObjectiveSimOpt<dim,nstate>::gradient_2(
    ROL::Vector<double> &gradient_ctl,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
    
{
    std::cout << "Computing gradient 2" << std::endl;
    update(des_var_sim, des_var_ctl);

    // pointer to functional
    // std::shared_ptr< Functional<dim,nstate,double> > functional_ptr =  std::make_shared<Functional<dim,nstate,double>>(this->functional);

    this->functional->evaluate_functional(true,true,false);
    // Create acoustic adjoint object using Amiet functional
    this->acoustic_adjoint = std::make_shared<AcousticAdjoint <dim,nstate,double,Triangulation>>(this->dg,this->functional);
    // std::shared_ptr<AmietModelFunctional<dim,nstate,double,Triangulation>> amiet_test = std::make_shared<AmietModelFunctional<dim,nstate,double,Triangulation>>(dg,boundary_layer_extraction,observer_coord_ref);

    this->acoustic_adjoint->compute_dIdXd(this->dg->high_order_grid);
    auto &dIdXp = ROL_vector_to_dealii_vector_reference(gradient_ctl);
    dIdXp = this->acoustic_adjoint->dIdXd;

    // const bool compute_dIdW = false, compute_dIdX = true, compute_d2I = false;
    // functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

    // const auto &dIdXv = functional.dIdX;

    // auto &dealii_output = ROL_vector_to_dealii_vector_reference(gradient_ctl);
    // dXvdXp.Tvmult(dealii_output, dIdXv);

    // This part below was already commented (Zineb, april 10 2025)

    //n_vmult += 1;

    // auto dIdXvs = dIdXv;
    // {
    //     dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid->surface_nodes);
    //     MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //         meshmover(*(functional.dg->high_order_grid->triangulation),
    //           functional.dg->high_order_grid->initial_mapping_fe_field,
    //           functional.dg->high_order_grid->dof_handler_grid,
    //           functional.dg->high_order_grid->surface_to_volume_indices,
    //           dummy_vector);
    //     meshmover.apply_dXvdXvs_transpose(dIdXv, dIdXvs);
    // }

    // auto &dIdXp = ROL_vector_to_dealii_vector_reference(gradient_ctl);
    // {
    //     dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    //     ffd.get_dXvsdXp (*(functional.dg->high_order_grid), ffd_design_variables_indices_dim, dXvsdXp);
    //     dXvsdXp.Tvmult(dIdXp, dIdXvs);
    // }

}

template <int dim, int nstate>
void ROLAcousticObjectiveSimOpt<dim,nstate>::hessVec_11(
    ROL::Vector<double> &output_vector,
    const ROL::Vector<double> &input_vector,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = true;
    const bool compute_dIdX = false;
    const bool compute_d2I = true;
    this->functional->evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

    const auto &dealii_input = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &hv = ROL_vector_to_dealii_vector_reference(output_vector);

    this->functional->d2IdWdW->vmult(hv, dealii_input);

    //n_vmult += 1;
}

template <int dim, int nstate>
void ROLAcousticObjectiveSimOpt<dim,nstate>::hessVec_12(
    ROL::Vector<double> &output_vector,
    const ROL::Vector<double> &input_vector,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const auto &dealii_input = ROL_vector_to_dealii_vector_reference(input_vector);

    // auto dXvsdXp_input = functional.dg->high_order_grid->volume_nodes;
    // {
    //     dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    //     ffd.get_dXvsdXp (functional.dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
    //     dXvsdXp.vmult(dXvsdXp_input, dealii_input);
    // }

    // auto dXvdXp_input = dXvsdXp_input;
    // {
    //     dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid->surface_nodes);
    //     MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //         meshmover(*(functional.dg->high_order_grid->triangulation),
    //           functional.dg->high_order_grid->initial_mapping_fe_field,
    //           functional.dg->high_order_grid->dof_handler_grid,
    //           functional.dg->high_order_grid->surface_to_volume_indices,
    //           dummy_vector);
    //     meshmover.apply_dXvdXvs(dXvsdXp_input, dXvdXp_input);
    // }

    // auto &d2IdWdXp_input = ROL_vector_to_dealii_vector_reference(output_vector);
    // {
    //     const bool compute_dIdW = false, compute_dIdX = false, compute_d2I = true;
    //     functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
    //     functional.d2IdWdX.vmult(d2IdWdXp_input, dXvdXp_input);
    // }

    auto dXvdXp_input = this->dg->high_order_grid->volume_nodes;
    dXvdXp.vmult(dXvdXp_input, dealii_input);

    auto &dealii_output = ROL_vector_to_dealii_vector_reference(output_vector);
    {
        const bool compute_dIdW = true, compute_dIdX = false, compute_d2I = true;
        this->functional->evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
        this->functional->d2IdWdX->vmult(dealii_output, dXvdXp_input);
    }

    //n_vmult += 2;

}

template <int dim, int nstate>
void ROLAcousticObjectiveSimOpt<dim,nstate>::hessVec_21(
    ROL::Vector<double> &output_vector,
    const ROL::Vector<double> &input_vector,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = true;
    const bool compute_dIdX = false;
    const bool compute_d2I = true;
    this->functional->evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

    const auto &dealii_input = ROL_vector_to_dealii_vector_reference(input_vector);

    auto d2IdXdW_input = this->dg->high_order_grid->volume_nodes;
    this->functional->d2IdWdX->Tvmult(d2IdXdW_input, dealii_input);

    // auto d2IdXvsdW_input = functional.dg->high_order_grid->volume_nodes;
    // {
    //     dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid->surface_nodes);
    //     MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //         meshmover(*(functional.dg->high_order_grid->triangulation),
    //           functional.dg->high_order_grid->initial_mapping_fe_field,
    //           functional.dg->high_order_grid->dof_handler_grid,
    //           functional.dg->high_order_grid->surface_to_volume_indices,
    //           dummy_vector);
    //     meshmover.apply_dXvdXvs_transpose(d2IdXdW_input, d2IdXvsdW_input);
    // }

    // auto &d2IdXpdW_input = ROL_vector_to_dealii_vector_reference(output_vector);
    // {
    //     dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    //     ffd.get_dXvsdXp (*(functional.dg->high_order_grid), ffd_design_variables_indices_dim, dXvsdXp);
    //     dXvsdXp.Tvmult(d2IdXpdW_input, d2IdXvsdW_input);
    // }

    auto &dealii_output = ROL_vector_to_dealii_vector_reference(output_vector);
    dXvdXp.Tvmult(dealii_output, d2IdXdW_input);

    //n_vmult += 2;
}

template <int dim, int nstate>
void ROLAcousticObjectiveSimOpt<dim,nstate>::hessVec_22(
    ROL::Vector<double> &output_vector,
    const ROL::Vector<double> &input_vector,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);


    const auto &dealii_input = ROL_vector_to_dealii_vector_reference(input_vector);

    // dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    // ffd.get_dXvsdXp (*(functional.dg->high_order_grid), ffd_design_variables_indices_dim, dXvsdXp);

    // auto dXvsdXp_input = functional.dg->high_order_grid->volume_nodes;
    // {
    //     dXvsdXp.vmult(dXvsdXp_input, dealii_input);
    // }

    // auto dXvdXp_input = dXvsdXp_input;
    // {
    //     dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid->surface_nodes);
    //     MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //         meshmover(*(functional.dg->high_order_grid->triangulation),
    //           functional.dg->high_order_grid->initial_mapping_fe_field,
    //           functional.dg->high_order_grid->dof_handler_grid,
    //           functional.dg->high_order_grid->surface_to_volume_indices,
    //           dummy_vector);
    //     meshmover.apply_dXvdXvs(dXvsdXp_input, dXvdXp_input);
    // }

    auto dXvdXp_input = this->dg->high_order_grid->volume_nodes;
    dXvdXp.vmult(dXvdXp_input, dealii_input);

    auto d2IdXdXp_input = this->dg->high_order_grid->volume_nodes;
    {
        const bool compute_dIdW = true, compute_dIdX = false, compute_d2I = true;
        this->functional->evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
        this->functional->d2IdXdX->vmult(d2IdXdXp_input, dXvdXp_input);
    }

    //auto d2IdXvsdXp_input = functional.dg->high_order_grid->volume_nodes;
    //{
    //    dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid->surface_nodes);
    //    MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //        meshmover(*(functional.dg->high_order_grid->triangulation),
    //          functional.dg->high_order_grid->initial_mapping_fe_field,
    //          functional.dg->high_order_grid->dof_handler_grid,
    //          functional.dg->high_order_grid->surface_to_volume_indices,
    //          dummy_vector);
    //    meshmover.apply_dXvdXvs_transpose(d2IdXdXp_input, d2IdXvsdXp_input);
    //}

    //auto &d2IdXpdXp_input = ROL_vector_to_dealii_vector_reference(output_vector);
    //{
    //    dXvsdXp.Tvmult(d2IdXpdXp_input, d2IdXvsdXp_input);
    //}

    auto &dealii_output = ROL_vector_to_dealii_vector_reference(output_vector);
    dXvdXp.Tvmult(dealii_output, d2IdXdXp_input);

    //n_vmult += 3;
}

// template class ROLAcousticObjectiveSimOpt <PHILIP_DIM,1>;
// template class ROLAcousticObjectiveSimOpt <PHILIP_DIM,2>;
// template class ROLAcousticObjectiveSimOpt <PHILIP_DIM,3>;
template class ROLAcousticObjectiveSimOpt <PHILIP_DIM,4>;
template class ROLAcousticObjectiveSimOpt <PHILIP_DIM,5>;

} // PHiLiP namespace
