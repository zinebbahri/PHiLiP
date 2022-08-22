#ifndef __MODEL__
#define __MODEL__

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include "parameters/parameters_manufactured_solution.h"
#include "physics/manufactured_solution.h"

namespace PHiLiP {
namespace Physics {

/// Physics model additional terms and equations to the baseline physics. 
template <int dim, int nstate, typename real>
class ModelBase
{
public:
	/// Constructor
	ModelBase(
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function_input = nullptr);

    /// Virtual destructor required for abstract classes.
    virtual ~ModelBase() = 0;

    /// Manufactured solution function
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function;

    /// Convective flux terms additional to the baseline physics
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> 
    convective_flux (
        const std::array<real,nstate> &conservative_soln) const = 0;

    /// Dissipative flux terms additional to the baseline physics
	virtual std::array<dealii::Tensor<1,dim,real>,nstate> 
	dissipative_flux (
    	const std::array<real,nstate> &conservative_soln,
    	const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const = 0;

    /// Convective Numerical Split Flux for split form additional to the baseline physics
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> 
    convective_numerical_split_flux (
        const std::array<real,nstate> &soln_const, 
        const std::array<real,nstate> &soln_loop) const = 0;

    /// Spectral radius of convective term Jacobian.
    /** Used for scalar dissipation
     */
    virtual std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*solution*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const = 0;

    /// Maximum convective eigenvalue used in Lax-Friedrichs
    virtual real max_convective_eigenvalue (const std::array<real,nstate> &soln) const = 0;

    //adding physical source 
    /// Physical source terms 
    virtual std::array<real,nstate> physical_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const = 0;

    /// Source terms additional to the baseline physics
    virtual std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution,
        const dealii::types::global_dof_index cell_index) const = 0;

    /// Evaluates boundary values and gradients on the other side of the face.
    virtual void boundary_face_values (
        const int /*boundary_type*/,
        const dealii::Point<dim, real> &/*pos*/,
        const dealii::Tensor<1,dim,real> &/*normal*/,
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;

    // Quantities needed to be updated by DG for the model -- accomplished by DGBase update_model_variables()
    dealii::LinearAlgebra::distributed::Vector<int> cellwise_poly_degree; ///< Cellwise polynomial degree
    dealii::LinearAlgebra::distributed::Vector<double> cellwise_volume; ////< Cellwise element volume
};

} // Physics namespace
} // PHiLiP namespace

#endif
