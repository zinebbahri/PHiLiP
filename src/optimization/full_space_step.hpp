#ifndef __FULLSPACE_STEP_H__
#define __FULLSPACE_STEP_H__

#include "ROL_Types.hpp"
#include "ROL_Step.hpp"
#include "ROL_LineSearch.hpp"

// Unconstrained Methods
#include "ROL_GradientStep.hpp"
#include "ROL_NonlinearCGStep.hpp"
#include "ROL_SecantStep.hpp"
#include "ROL_NewtonStep.hpp"
#include "ROL_NewtonKrylovStep.hpp"

#include <sstream>
#include <iomanip>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/vector_memory.templates.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>

#include "optimization/rol_to_dealii_vector.hpp"
#include "optimization/flow_constraints.hpp"
#include "optimization/rol_objective.hpp"

template<typename Real = double>
class dealiiSolverVectorWrappingROL
{
private:
    ROL::Ptr<ROL::Vector<Real>> rol_vector_ptr;
public:
    using value_type = Real;

    dealiiSolverVectorWrappingROL() {};

    dealiiSolverVectorWrappingROL(ROL::Ptr<ROL::Vector<Real>> input_vector)
    : rol_vector_ptr(input_vector)
    {};

    ROL::Ptr<ROL::Vector<Real>> getVector()
    {
        return rol_vector_ptr;
    }
    ROL::Ptr<const ROL::Vector<double>> getVector() const
    {
        return rol_vector_ptr;
    }
    
    // Resize the current object to have the same size and layout as
    // the model_vector argument provided. The second argument
    // indicates whether to clear the current object after resizing.
    // The second argument must have a default value equal to false.
    void reinit (const dealiiSolverVectorWrappingROL &model_vector,
                 const bool leave_elements_uninitialized = false)
    {
        (void) leave_elements_uninitialized;
        rol_vector_ptr = model_vector.getVector()->clone();
    }
    // Inner product between the current object and the argument.
    double operator * (const dealiiSolverVectorWrappingROL &v) const
    {
        return rol_vector_ptr->dot( *(v.getVector()) );
    }
    // Assignment of a scalar
    dealiiSolverVectorWrappingROL & operator=(const double a)
    {
        rol_vector_ptr->setScalar(a);
        return *this;
    }
    // Multiply the elements of the current object by a fixed value.
    dealiiSolverVectorWrappingROL & operator*=(const double a)
    {
        rol_vector_ptr->scale(a);
        return *this;
    }
    // Addition of vectors
    void add (const dealiiSolverVectorWrappingROL &x)
    {
        rol_vector_ptr->plus( *(x.getVector()) );
    }
    // Scaled addition of vectors
    void add (const double  a,
              const dealiiSolverVectorWrappingROL &x)
    {
        rol_vector_ptr->axpy(a, *(x.getVector()) );
    }
    // Scaled addition of vectors
    void sadd (const double  a,
               const double  b,
               const dealiiSolverVectorWrappingROL &x)
    {
        rol_vector_ptr->scale(a);
        rol_vector_ptr->axpy(b, *(x.getVector()) );
    }
    // Scaled assignment of a vector
    void equ (const double  a,
              const dealiiSolverVectorWrappingROL &x)
    {
        rol_vector_ptr->set( *(x.getVector()) );
        rol_vector_ptr->scale(a);
    }
    // Combined scaled addition of vector x into the current object and
    // subsequent inner product of the current object with v.
    double add_and_dot (const double  a,
                        const dealiiSolverVectorWrappingROL &x,
                        const dealiiSolverVectorWrappingROL &v)
    {
        this->add(a, x);
        return (*this) * v;
    }
    // Return the l2 norm of the vector.
    double l2_norm () const
    {
        return std::sqrt( (*this) * (*this) );
    }
};


template class dealii::VectorMemory<dealiiSolverVectorWrappingROL<double>>;
template class dealii::GrowingVectorMemory<dealiiSolverVectorWrappingROL<double>>;


namespace ROL {

template <class Real>
class FullSpace_BirosGhattas : public Step<Real> {
private:

    // Vectors used for cloning.
    ROL::Ptr<Vector<Real> > xvec_;
    ROL::Ptr<Vector<Real> > gvec_;
    ROL::Ptr<Vector<Real> > lvec_;
    ROL::Ptr<Vector<Real> > cvec_;

    ROL::Ptr<Objective<Real>> merit_function_;
    ROL::Ptr<Vector<Real>> lagrange_mult_search_direction_;

    ROL::Ptr<Vector<Real>> previous_reduced_gradient_;

    ROL::Ptr<Step<Real> >        desc_;       ///< Unglobalized step object
    ROL::Ptr<Secant<Real> >      secant_;     ///< Secant object (used for quasi-Newton)
    ROL::Ptr<LineSearch<Real> >  lineSearch_; ///< Line-search object
  
    ESecant             esec_;
    ELineSearch         els_;   ///< enum determines type of line search
    ECurvatureCondition econd_; ///< enum determines type of curvature condition
  
    Real penalty_value_;
    bool acceptLastAlpha_;  ///< For backwards compatibility. When max function evaluations are reached take last step
  
    bool usePreviousAlpha_; ///< If true, use the previously accepted step length (if any) as the new initial step length
  
    int verbosity_;
    bool computeObj_;
    Real fval_;
  
    ROL::ParameterList parlist_;
  
    std::string lineSearchName_;  
    std::string secantName_;
  
public:
  
    using Step<Real>::initialize;
    using Step<Real>::compute;
    using Step<Real>::update;
  
    /** \brief Constructor.
  
        Standard constructor to build a FullSpace_BirosGhattas object.  Algorithmic 
        specifications are passed in through a ROL::ParameterList.  The
        line-search type, secant type, Krylov type, or nonlinear CG type can
        be set using user-defined objects.
  
        @param[in]     parlist    is a parameter list containing algorithmic specifications
        @param[in]     lineSearch is a user-defined line search object
        @param[in]     secant     is a user-defined secant object
        @param[in]     krylov     is a user-defined Krylov object
        @param[in]     nlcg       is a user-defined Nonlinear CG object
    */
    FullSpace_BirosGhattas(
        ROL::ParameterList &parlist,
        const ROL::Ptr<LineSearch<Real> > &lineSearch = ROL::nullPtr,
        const ROL::Ptr<Secant<Real> > &secant = ROL::nullPtr)
        : Step<Real>()
        , desc_(ROL::nullPtr)
        , secant_(secant)
        , lineSearch_(lineSearch)
        , els_(LINESEARCH_USERDEFINED)
        , econd_(CURVATURECONDITION_WOLFE)
        , verbosity_(0)
        , computeObj_(true)
        , fval_(0)
        , parlist_(parlist)
    {
        // Parse parameter list
        ROL::ParameterList& Llist = parlist.sublist("Step").sublist("Line Search");
        ROL::ParameterList& Glist = parlist.sublist("General");
        econd_ = StringToECurvatureCondition(Llist.sublist("Curvature Condition").get("Type","Strong Wolfe Conditions") );
        acceptLastAlpha_ = Llist.get("Accept Last Alpha", false); 
        verbosity_ = Glist.get("Print Verbosity",0);
        computeObj_ = Glist.get("Recompute Objective Function",false);
        // Initialize Line Search
        if (lineSearch_ == ROL::nullPtr) {
          lineSearchName_ = Llist.sublist("Line-Search Method").get("Type","Cubic Interpolation"); 
          els_ = StringToELineSearch(lineSearchName_);
          lineSearch_ = LineSearchFactory<Real>(parlist);
        } 
        else { // User-defined linesearch provided
          lineSearchName_ = Llist.sublist("Line-Search Method").get("User Defined Line-Search Name",
                                                                    "Unspecified User Defined Line-Search");
        }

        secantName_ = Glist.sublist("Secant").get("Type","Limited-Memory BFGS");
        esec_ = StringToESecant(secantName_);
        secant_ = SecantFactory<Real>(parlist);
  
    }

    void computeLagrangianGradient(
        Vector<Real> &lagrangian_gradient,
        const Vector<Real> &design_variables,
        const Vector<Real> &lagrange_mult,
        const Vector<Real> &objective_gradient,
        Constraint<Real> &equal_constraints) const
    {
        /* Apply adjoint of constraint Jacobian to current multiplier. */
        Real tol = std::sqrt(ROL_EPSILON<Real>());
        equal_constraints.applyAdjointJacobian(lagrangian_gradient, lagrange_mult, design_variables, tol);
        lagrangian_gradient.plus(objective_gradient);
    }

    void computeInitialLagrangeMultiplier(
        Vector<Real> &lagrange_mult,
        const Vector<Real> &design_variables,
        const Vector<Real> &objective_gradient,
        Constraint<Real> &equal_constraints) const
    {
 
        Real one(1);
 
        /* Form right-hand side of the augmented system. */
        ROL::Ptr<Vector<Real> > rhs1 = gvec_->clone();
        ROL::Ptr<Vector<Real> > rhs2 = cvec_->clone();

        // rhs1 is the negative gradient of the Lagrangian
        // rhs2 is zero
        computeLagrangianGradient(*rhs1, design_variables, lagrange_mult, objective_gradient, equal_constraints);
        rhs1->scale(-one);
        rhs2->zero();
 
        /* Declare left-hand side of augmented system. */
        ROL::Ptr<Vector<Real> > lhs1 = xvec_->clone();
        ROL::Ptr<Vector<Real> > lhs2 = lvec_->clone();
 
        /* Compute linear solver tolerance. */
        Real b1norm  = rhs1->norm();
        Real tol = std::sqrt(ROL_EPSILON<Real>());
        //Real tol = 1e-12;//setTolOSS(lmhtol_*b1norm);
 
        /* Solve augmented system. */
        const std::vector<Real> augiters = equal_constraints.solveAugmentedSystem(*lhs1, *lhs2, *rhs1, *rhs2, design_variables, tol);
 
        /* Return updated Lagrange multiplier. */
        // lhs2 is the multiplier update
        lagrange_mult.plus(*lhs2);
 
    }  // computeInitialLagrangeMultiplier

  
    virtual void initialize(
        Vector<Real> &design_variables,
        const Vector<Real> &gradient,
        Vector<Real> &lagrange_mult,
        const Vector<Real> &equal_constraints_values,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        AlgorithmState<Real> &algo_state ) override
    {
        BoundConstraint<Real> bound_constraints;
        bound_constraints.deactivate();
        initialize(
            design_variables,
            gradient,
            lagrange_mult,
            equal_constraints_values,
            objective,
            equal_constraints,
            bound_constraints, // new argument
            algo_state);
    }

    void initialize(
        Vector<Real> &design_variables,
        const Vector<Real> &gradient,
        Vector<Real> &lagrange_mult,
        const Vector<Real> &equal_constraints_values,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        BoundConstraint<Real> &bound_constraints,
        AlgorithmState<Real> &algo_state ) override
    {
        std::cout << __PRETTY_FUNCTION__ << std::endl;
        //Real tol = std::sqrt(ROL_EPSILON<Real>())
        Real tol = ROL_EPSILON<Real>();
        Real zero(0);

        // Initialize the algorithm state
        algo_state.iter  = 0;
        algo_state.nfval = 0;
        algo_state.ncval = 0;
        algo_state.ngrad = 0;

        ROL::Ptr<StepState<Real> > step_state = Step<Real>::getState();
        xvec_ = design_variables.clone();
        gvec_ = gradient.clone();
        lvec_ = lagrange_mult.clone();
        cvec_ = equal_constraints_values.clone();

        lagrange_mult_search_direction_ = lagrange_mult.clone();

        // Initialize state descent direction and gradient storage
        step_state->descentVec  = design_variables.clone();
        step_state->gradientVec = gradient.clone();
        step_state->constraintVec = equal_constraints_values.clone();
        step_state->searchSize  = zero;

        // Project design_variables onto bound_constraints set
        if ( bound_constraints.isActivated() ) {
          bound_constraints.project(design_variables);
        }
        // Update objective function, get value, and get gradient
        const bool changed_design_variables = true;
        objective.update(design_variables, changed_design_variables, algo_state.iter);
        algo_state.value = objective.value(design_variables, tol);
        algo_state.nfval++;
        objective.gradient(*(step_state->gradientVec), design_variables, tol);
        algo_state.ngrad++;

        // Update equal_constraints.
        equal_constraints.update(design_variables,true,algo_state.iter);
        equal_constraints.value(*(step_state->constraintVec), design_variables, zero);
        algo_state.cnorm = cvec_->norm();
        algo_state.ncval++;

        // Compute gradient of Lagrangian at new multiplier guess.
        ROL::Ptr<Vector<Real> > lagrangian_gradient = step_state->gradientVec->clone();
        computeLagrangianGradient(*lagrangian_gradient, design_variables, lagrange_mult, *(step_state->gradientVec), equal_constraints);
        const auto &lagrangian_gradient_simopt = dynamic_cast<const Vector_SimOpt<Real>&>(*lagrangian_gradient);
        previous_reduced_gradient_ = lagrangian_gradient_simopt.get_2()->clone();
        algo_state.ngrad++;

        // // Not sure why this is done in ROL_Step.hpp
        // if ( bound_constraints.isActivated() ) {
        //     ROL::Ptr<Vector<Real> > xnew = design_variables.clone();
        //     xnew->set(design_variables);
        //     xnew->axpy(-one,(step_state->gradientVec)->dual());
        //     bound_constraints.project(*xnew);
        //     xnew->axpy(-one,design_variables);
        //     algo_state.gnorm = xnew->norm();
        // }
        // else {
        //     algo_state.gnorm = (step_state->gradientVec)->norm();
        // }
        algo_state.gnorm = (step_state->gradientVec)->norm();

        // I don't have to initialize with merit function since it does nothing
        // with it. But might as well be consistent.
        penalty_value_ = 1.0;
        merit_function_ = ROL::makePtr<ROL::AugmentedLagrangian<Real>> (
                makePtrFromRef<Objective<Real>>(objective),
                makePtrFromRef<Constraint<Real>>(equal_constraints),
                lagrange_mult,
                penalty_value_,
                design_variables,
                equal_constraints_values,
                parlist_);

        // Dummy search direction vector used to initialize the linesearch.
        ROL::Ptr<Vector<Real> > search_direction_dummy = design_variables.clone();
        lineSearch_->initialize(design_variables, *search_direction_dummy, gradient, *merit_function_, bound_constraints);
    }
    Real computeAugmentedLagrangianPenalty(
        const Vector<Real> &search_direction,
        const Vector<Real> &lagrange_mult_search_direction,
        const Vector<Real> &design_variables,
        const Vector<Real> &objective_gradient,
        const Vector<Real> &equal_constraints_values,
        const Vector<Real> &adjoint_jacobian_lagrange,
        Constraint<Real> &equal_constraints,
        const Real offset)
    {
        std::cout << __PRETTY_FUNCTION__ << std::endl;
        // Biros and Ghattas 2005, Part II
        // Equation (2.10)
        Real penalty = objective_gradient.dot(search_direction);
        std::cout << "penalty1 " << penalty <<std::endl;
        penalty += adjoint_jacobian_lagrange.dot(search_direction);
        std::cout << "penalty2 " << penalty <<std::endl;
        penalty += equal_constraints_values.dot(lagrange_mult_search_direction);
        std::cout << "penalty3 " << penalty <<std::endl;

        const ROL::Ptr<Vector<Real>> jacobian_search_direction = equal_constraints_values.clone();
        Real tol = std::sqrt(ROL_EPSILON<Real>());
        equal_constraints.applyJacobian(*jacobian_search_direction, search_direction, design_variables, tol);

        Real denom = jacobian_search_direction->dot(equal_constraints_values);
        std::cout << "denom " << denom <<std::endl;

        penalty /= denom;
        std::cout << "penalty4 " << penalty <<std::endl;

        // Note that the offset is not on the fraction as in the paper.
        // The penalty term should always be positive and towards infinity.
        // It is a mistake from the paper since the numerator and denominator can be
        // small and negative. Therefore, the positive offset on a small negative
        // numerator with a small negative denominator might result in a large negative
        // penalty value.
        if (penalty > 0.0) {
            penalty += offset;
        } else {
            penalty = 1.0;
        }
        std::cout << "penalty5 " << penalty <<std::endl;

        return penalty;
    }

    class KKT_P2_Preconditioner
    {
    protected:

        const Ptr<Objective<Real>> objective_;
        const Ptr<Constraint_SimOpt<Real>> equal_constraints_;

        const Ptr<const Vector_SimOpt<Real>> design_variables_;
        const Ptr<const Vector<Real>> lagrange_mult_;
        const Ptr<const Vector<Real>> simulation_variables_;
        const Ptr<const Vector<Real>> control_variables_;

        const ROL::Ptr<Secant<Real> > secant_;

    private:

        const Ptr<Vector<Real>> temp_design_variables_size_vector_;

    public:
        KKT_P2_Preconditioner(
            const Ptr<Objective<Real>> objective,
            const Ptr<Constraint<Real>> equal_constraints,
            const Ptr<const Vector<Real>> design_variables,
            const Ptr<const Vector<Real>> lagrange_mult,
            const ROL::Ptr<Secant<Real> > secant)
            : objective_(objective)
            , equal_constraints_
                (ROL::makePtrFromRef<Constraint_SimOpt<Real>>(dynamic_cast<Constraint_SimOpt<Real>&>(*equal_constraints)))
            , design_variables_
                (ROL::makePtrFromRef<const Vector_SimOpt<Real>>(dynamic_cast<const Vector_SimOpt<Real>&>(*design_variables)))
            , lagrange_mult_(lagrange_mult)
            , simulation_variables_(design_variables_->get_1())
            , control_variables_(design_variables_->get_2())
            , secant_(secant)
            , temp_design_variables_size_vector_(design_variables->clone())
        { };
        // Application of matrix to vector src. Write result into dst.
        void vmult (dealiiSolverVectorWrappingROL<double>       &dst,
                    const dealiiSolverVectorWrappingROL<double> &src) const
        {
            static int number_of_times = 0;
            number_of_times++;
            std::cout << "Number of P2_KKT vmult = " << number_of_times << std::endl;
            Real tol = 1e-15;
            //const Real one = 1.0;

            ROL::Ptr<ROL::Vector<Real>> dst_rol = dst.getVector();
            auto &dst_split = dynamic_cast<Vector_SimOpt<Real>&>(*dst_rol);
            Ptr<Vector<Real>> dst_design = dst_split.get_1();
            auto &dst_design_split = dynamic_cast<Vector_SimOpt<Real>&>(*dst_design);

            Ptr<Vector<Real>> dst_1 = dst_design_split.get_1();
            Ptr<Vector<Real>> dst_2 = dst_design_split.get_2();
            Ptr<Vector<Real>> dst_3 = dst_split.get_2();

            const ROL::Ptr<const ROL::Vector<Real>> src_rol = src.getVector();
            const auto &src_split = dynamic_cast<const Vector_SimOpt<Real>&>(*src_rol);
            const Ptr<const Vector<Real>> src_design = src_split.get_1();
            const auto &src_design_split = dynamic_cast<const Vector_SimOpt<Real>&>(*src_design);

            const Ptr<const Vector<Real>> src_1 = src_design_split.get_1();
            const Ptr<const Vector<Real>> src_2 = src_design_split.get_2();
            const Ptr<const Vector<Real>> src_3 = src_split.get_2();

            Ptr<Vector<Real>> temp_1 = dst_1->clone();
            Ptr<Vector<Real>> temp_2 = dst_2->clone();
            Ptr<Vector<Real>> temp_3 = dst_3->clone();
            temp_1->set(*src_3);
            temp_3->set(*src_1);
            Ptr<Vector<Real>> AsTinv_temp_3 = temp_3->clone();
            equal_constraints_->applyInverseAdjointJacobian_1(*AsTinv_temp_3, *temp_3, *simulation_variables_, *control_variables_, tol);
            equal_constraints_->applyAdjointJacobian_2(*temp_2, *AsTinv_temp_3, *simulation_variables_, *control_variables_, tol);
            temp_2->scale(-1.0);
            temp_2->plus(*src_2);

            dst_3->set(*AsTinv_temp_3);
            // Need to apply Hessian inverse on dst_2
            secant_->applyH( *dst_2, *temp_2);
            //dst_2->set(*temp_2);

            Ptr<Vector<Real>> dst_1_rhs = dst_1->clone();
            equal_constraints_->applyJacobian_2(*dst_1_rhs, *dst_2, *simulation_variables_, *control_variables_, tol);
            dst_1_rhs->scale(-1.0);

            equal_constraints_->applyInverseJacobian_1(*dst_1, *dst_1_rhs, *simulation_variables_, *control_variables_, tol);

            dealii::deallog.depth_console(99);

        }
        // Application of transpose to a vector. This function is,
        // however, only used by some iterative methods.
        void Tvmult (dealiiSolverVectorWrappingROL<double>       &dst,
                     const dealiiSolverVectorWrappingROL<double> &src) const
        {
            vmult(dst, src);
        }
    };

  
    class KKT_Operator
    {
    protected:

        const Ptr<Objective<Real>> objective_;
        const Ptr<Constraint<Real>> equal_constraints_;

        const Ptr<const Vector<Real>> design_variables_;
        const Ptr<const Vector<Real>> lagrange_mult_;


    private:

        const Ptr<Vector<Real>> temp_design_variables_size_vector_;

    public:
        KKT_Operator(
            const Ptr<Objective<Real>> objective,
            const Ptr<Constraint<Real>> equal_constraints,
            const Ptr<const Vector<Real>> design_variables,
            const Ptr<const Vector<Real>> lagrange_mult)
            : objective_(objective)
            , equal_constraints_(equal_constraints)
            , design_variables_(design_variables)
            , lagrange_mult_(lagrange_mult)
            , temp_design_variables_size_vector_(design_variables->clone())
        { };
        // Application of matrix to vector src. Write result into dst.
        void vmult (dealiiSolverVectorWrappingROL<double>       &dst,
                    const dealiiSolverVectorWrappingROL<double> &src) const
        {
            static int number_of_times = 0;
            number_of_times++;
            std::cout << "Number of KKT vmult = " << number_of_times << std::endl;
            Real tol = 1e-15;
            const Real one = 1.0;

            ROL::Ptr<ROL::Vector<Real>> dst_rol = dst.getVector();
            const ROL::Ptr<const ROL::Vector<Real>> src_rol = src.getVector();

            auto &dst_split = dynamic_cast<Vector_SimOpt<Real>&>(*dst_rol);
            const auto &src_split = dynamic_cast<const Vector_SimOpt<Real>&>(*src_rol);

            Ptr<Vector<Real>> dst_design = dst_split.get_1();
            Ptr<Vector<Real>> dst_constraints = dst_split.get_2();

            const Ptr<const Vector<Real>> src_design = src_split.get_1();
            const Ptr<const Vector<Real>> src_constraints = src_split.get_2();

            // Top left block times top vector
            {
                objective_->hessVec(*dst_design, *src_design, *design_variables_, tol);
                equal_constraints_->applyAdjointHessian(*temp_design_variables_size_vector_, *lagrange_mult_, *src_design, *design_variables_, tol);
                dst_design->axpy(one, *temp_design_variables_size_vector_);
            }
            {
                //dst_design->set(*src_design);
            }

            // Top right block times bottom vector
            equal_constraints_->applyAdjointJacobian(*temp_design_variables_size_vector_, *src_constraints, *design_variables_, tol);
            dst_design->axpy(one, *temp_design_variables_size_vector_);

            // Bottom left left block times top vector
            equal_constraints_->applyJacobian(*dst_constraints, *src_design, *design_variables_, tol);

            // Bottom right block times bottom vector
            // 0 block in KKT
            dealii::deallog.depth_console(99);

        }
        // // Application of transpose to a vector. This function is,
        // // however, only used by some iterative methods.
        // void Tvmult (dealiiSolverVectorWrappingROL<double>       &dst,
        //              const dealiiSolverVectorWrappingROL<double> &src) const
        // {
        //     vmult(dst, src);
        // }
    };

    // std::vector<Real> solve_KKT_system(
    //     const Vector<Real> &search_direction,
    //     const Vector<Real> &lag_search_direction,
    //     const Vector<Real> &design_variables,
    //     const Vector<Real> &lagrange_mult,
    //     Objective<Real> &objective,
    //     Constraint<Real> &equal_constraints,
    // {
    //     auto &my_objective = dynamic_cast<PHiLiP::ROLObjectiveSimOpt<PHILIP_DIM>>(objective);
    //     auto &flow_constraints = dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>>(equal_constraints);

    //     dealii::TrilinosWrappers::BlockSparseMatrix kkt_hessian;
    //     kkt_hessian.reinit(3,3);
    //     kkt_hessian.block(0, 0).copy_from( functional.d2IdWdW);
    //     kkt_hessian.block(0, 1).copy_from( functional.d2IdWdX);
    //     kkt_hessian.block(0, 2).copy_from( dRdW_transpose);

    //     kkt_hessian.block(1, 0).copy_from( d2IdXdW);
    //     kkt_hessian.block(1, 1).copy_from( functional.d2IdXdX);
    //     kkt_hessian.block(1, 2).copy_from( dRdX_transpose);

    //     kkt_hessian.block(2, 0).copy_from( dg->system_matrix);
    //     kkt_hessian.block(2, 1).copy_from( dg->dRdXv);
    //     dealii::TrilinosWrappers::SparsityPattern zero_sparsity_pattern(dg->locally_owned_dofs, MPI_COMM_WORLD, 0);
    //     zero_sparsity_pattern.compress();
    //     kkt_hessian.block(2, 2).reinit(zero_sparsity_pattern);

    //     kkt_hessian.collect_sizes();

    //     dealii::LinearAlgebra::distributed::BlockVector<double> block_vector(3);
    //     block_vector.block(0) = dg->solution;
    //     block_vector.block(1) = dg->high_order_grid.volume_nodes;
    //     block_vector.block(2) = dummy_dual;
    //     dealii::LinearAlgebra::distributed::BlockVector<double> Hv(3);
    //     dealii::LinearAlgebra::distributed::BlockVector<double> Htv(3);
    // }
    template<typename MatrixType, typename VectorType, typename PreconditionerType>
    std::pair<unsigned int, double>
    solve_linear (
        MatrixType &matrix_A,
        VectorType &right_hand_side,
        VectorType &solution,
        PreconditionerType &preconditioner)
        //const PHiLiP::Parameters::LinearSolverParam & param = )
    {
        dealii::deallog.depth_console(999);
        // dealii::PreconditionIdentity precond();
        // precond.initialize (matrix_A);
        dealii::SolverControl solver_control(100000, 1.e-15, true, true);
        // {
        //     const unsigned int 	max_n_tmp_vectors = 2000;
        //     const bool 	right_preconditioning = false;
        //     const bool 	use_default_residual = true;
        //     const bool 	force_re_orthogonalization = false;
        //     typedef typename dealii::SolverGMRES<VectorType>::AdditionalData AddiData;
        //     AddiData add_data( max_n_tmp_vectors, right_preconditioning, use_default_residual, force_re_orthogonalization);
        //     dealii::SolverGMRES<VectorType> solver_gmres(solver_control, add_data);
        //     solver_gmres.solve(matrix_A, solution, right_hand_side
        //     //, precond);
        //     //, dealii::PreconditionIdentity());
        //     , preconditioner);
        // }
        {
            const unsigned int 	max_n_tmp_vectors = 2000;
            typedef typename dealii::SolverFGMRES<VectorType>::AdditionalData AddiData;
            AddiData add_data( max_n_tmp_vectors );
            dealii::SolverFGMRES<VectorType> solver_fgmres(solver_control, add_data);
            solver_fgmres.solve(matrix_A, solution, right_hand_side
            //, precond);
            //, dealii::PreconditionIdentity());
            , preconditioner);
        }



        return {-1.0, -1.0};
    }

    std::vector<Real> solve_KKT_system(
        Vector<Real> &search_direction,
        Vector<Real> &lag_search_direction,
        const Vector<Real> &design_variables,
        const Vector<Real> &lagrange_mult,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints)
    {
        Real tol = std::sqrt(ROL_EPSILON<Real>());
        const Real one = 1.0;

        /* Form gradient of the Lagrangian. */
        ROL::Ptr<Vector<Real> > objective_gradient = gvec_->clone();
        objective.gradient(*objective_gradient, design_variables, tol);
        // Apply adjoint of equal_constraints Jacobian to current Lagrange multiplier.
        ROL::Ptr<Vector<Real> > adjoint_jacobian_lagrange = gvec_->clone();
        equal_constraints.applyAdjointJacobian(*adjoint_jacobian_lagrange, lagrange_mult, design_variables, tol);
  
        /* Form right-hand side of the augmented system. */
        ROL::Ptr<Vector<Real> > rhs1 = gvec_->clone();
        ROL::Ptr<Vector<Real> > rhs2 = cvec_->clone();
        // rhs1 is the negative gradient of the Lagrangian
        computeLagrangianGradient(*rhs1, design_variables, lagrange_mult, *objective_gradient, equal_constraints);
        rhs1->scale(-one);
        // rhs2 is the contraint value
        equal_constraints.value(*rhs2, design_variables, tol);
        rhs2->scale(-one);
  
        /* Declare left-hand side of augmented system. */
        ROL::Ptr<Vector<Real> > lhs1 = xvec_->clone();
        ROL::Ptr<Vector<Real> > lhs2 = lvec_->clone();

        ROL::Vector_SimOpt lhs_rol(lhs1, lhs2);
        ROL::Vector_SimOpt rhs_rol(rhs1, rhs2);

        KKT_Operator kkt_operator(
            makePtrFromRef<Objective<Real>>(objective),
            makePtrFromRef<Constraint<Real>>(equal_constraints),
            makePtrFromRef<const Vector<Real>>(design_variables),
            makePtrFromRef<const Vector<Real>>(lagrange_mult));

        KKT_P2_Preconditioner kkt_p2_precond(
            makePtrFromRef<Objective<Real>>(objective),
            makePtrFromRef<Constraint<Real>>(equal_constraints),
            makePtrFromRef<const Vector<Real>>(design_variables),
            makePtrFromRef<const Vector<Real>>(lagrange_mult),
            secant_);

        dealiiSolverVectorWrappingROL<double> lhs(makePtrFromRef(lhs_rol));
        dealiiSolverVectorWrappingROL<double> rhs(makePtrFromRef(lhs_rol));

        (void) solve_linear (kkt_operator, rhs, lhs, kkt_p2_precond);

        search_direction.set(*lhs1);
        lag_search_direction.set(*lhs2);

        std::vector<double> test(10);
        return test;
        
    }
    void compute(
        Vector<Real> &search_direction,
        const Vector<Real> &design_variables,
        const Vector<Real> &lagrange_mult,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        AlgorithmState<Real> &algo_state ) override
    {
        BoundConstraint<Real> bound_constraints;
        bound_constraints.deactivate();
        compute( search_direction, design_variables, lagrange_mult, objective, equal_constraints, bound_constraints, algo_state );
    }
    void compute(
        Vector<Real> &search_direction,
        const Vector<Real> &design_variables,
        const Vector<Real> &lagrange_mult,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        BoundConstraint<Real> &bound_constraints,
        AlgorithmState<Real> &algo_state ) override
    {
        std::cout << __PRETTY_FUNCTION__ << std::endl;
        ROL::Ptr<StepState<Real> > step_state = Step<Real>::getState();
  
        Real tol = std::sqrt(ROL_EPSILON<Real>());
        const Real one = 1.0;

        /* Form gradient of the Lagrangian. */
        ROL::Ptr<Vector<Real> > objective_gradient = gvec_->clone();
        objective.gradient(*objective_gradient, design_variables, tol);
        // Apply adjoint of equal_constraints Jacobian to current Lagrange multiplier.
        ROL::Ptr<Vector<Real> > adjoint_jacobian_lagrange = gvec_->clone();
        equal_constraints.applyAdjointJacobian(*adjoint_jacobian_lagrange, lagrange_mult, design_variables, tol);
  
        /* Form right-hand side of the augmented system. */
        ROL::Ptr<Vector<Real> > rhs1 = gvec_->clone();
        ROL::Ptr<Vector<Real> > rhs2 = cvec_->clone();
        // rhs1 is the negative gradient of the Lagrangian
        computeLagrangianGradient(*rhs1, design_variables, lagrange_mult, *objective_gradient, equal_constraints);
        rhs1->scale(-one);
        // rhs2 is the contraint value
        equal_constraints.value(*rhs2, design_variables, tol);
        rhs2->scale(-one);
  
        /* Declare left-hand side of augmented system. */
        ROL::Ptr<Vector<Real> > lhs1 = xvec_->clone();
        ROL::Ptr<Vector<Real> > lhs2 = lvec_->clone();
  
        // /* Compute linear solver tolerance. */
        // Real b1norm  = rhs1->norm();
        // Real tol = setTolOSS(lmhtol_*b1norm);
  
        /* Solve augmented system. */
        //const std::vector<Real> augiters = equal_constraints.solveAugmentedSystem(*lhs1, *lhs2, *rhs1, *rhs2, design_variables, tol);
        std::cout 
            << "Startingto solve augmented system..."
            << std::endl;
        //const std::vector<Real> kkt_iters = equal_constraints.solveAugmentedSystem(*lhs1, *lhs2, *rhs1, *rhs2, design_variables, tol);
        const std::vector<Real> kkt_iters = solve_KKT_system(*lhs1, *lhs2, design_variables, lagrange_mult, objective, equal_constraints);

        step_state->SPiter = kkt_iters.size();
        std::cout 
            << "Finished solving augmented system..."
            << std::endl;

        search_direction.set(*lhs1);
        lagrange_mult_search_direction_->set(*lhs2);

        //#pen_ = parlist.sublist("Step").sublist("Augmented Lagrangian").get("Initial Penalty Parameter",ten);
        /* Create merit function based on augmented Lagrangian */
        const Real penalty_offset = 1e-4;
        penalty_value_ = computeAugmentedLagrangianPenalty(
            search_direction,
            *lagrange_mult_search_direction_,
            design_variables,
            *objective_gradient,
            *(step_state->constraintVec),
            *adjoint_jacobian_lagrange,
            equal_constraints,
            penalty_offset);
        std::cout 
            << "Finished computeAugmentedLagrangianPenalty..."
            << std::endl;
        AugmentedLagrangian<Real> &augLag = dynamic_cast<AugmentedLagrangian<Real>&>(*merit_function_);
        augLag.reset(lagrange_mult, penalty_value_);

        const bool changed_design_variables = true;
        merit_function_->update(design_variables, changed_design_variables, algo_state.iter);
        ROL::Ptr<Vector<Real> > merit_function_gradient = gvec_->clone();
        merit_function_->gradient( *merit_function_gradient, design_variables, tol );
        Real directional_derivative_step = merit_function_gradient->dot(search_direction);
        directional_derivative_step += step_state->constraintVec->dot(*lagrange_mult_search_direction_);
        std::cout 
            << "Directional_derivative_step (Should be negative for descent direction)"
            << directional_derivative_step
            << std::endl;

        /* Perform line-search */
        fval_ = merit_function_->value(design_variables, tol );
        step_state->nfval = 0;
        step_state->ngrad = 0;
        std::cout 
            << "Performing line search..."
            << " Initial merit function value = " << fval_
            << std::endl;
        lineSearch_->setData(algo_state.gnorm,*merit_function_gradient);
        Real nfval_before = step_state->nfval;
        lineSearch_->run(step_state->searchSize,
                         fval_,
                         step_state->nfval,
                         step_state->ngrad,
                         directional_derivative_step,
                         search_direction,
                         design_variables,
                         *merit_function_,
                         bound_constraints);
        Real nfval_after = step_state->nfval;
        std::cout 
            << "End of line search... searchSize is..."
            << step_state->searchSize
            << " and number of function evaluations: "
            << nfval_after - nfval_before
            << " Final merit function value = " << fval_
            << std::endl;

        {
            auto &search_simopt = dynamic_cast<Vector_SimOpt<Real>&>(*search_direction);
            search_sim =
            simulation_variables_(design_variables_->get_1())
            control_variables_(design_variables_->get_2())
            // Reduced step
            secant_->applyH
        }

        // // Make correction if maximum function evaluations reached
        // if(!acceptLastAlpha_) {
        //     lineSearch_->setMaxitUpdate(step_state->searchSize,fval_,algo_state.value);
        // }
        // Compute scaled descent direction
        lagrange_mult_search_direction_->scale(step_state->searchSize);
        search_direction.scale(step_state->searchSize);
        if ( bound_constraints.isActivated() ) {
            search_direction.plus(design_variables);
            bound_constraints.project(search_direction);
            search_direction.axpy(static_cast<Real>(-1),design_variables);
        }
        std::cout
            << "End of compute..."
            << std::endl;

    }
  
    /** \brief Update step, if successful.
  
        Given a trial step, \f$s_k\f$, this function updates \f$x_{k+1}=x_k+s_k\f$. 
        This function also updates the secant approximation.
  
        @param[in,out]   design_variables        is the updated iterate
        @param[in]       search_direction        is the computed trial step
        @param[in]       objective               is the objective function
        @param[in]       equal_constraints       are the bound equal_constraints
        @param[in]       algo_state              contains the current state of the algorithm
    */
    void update(
        Vector<Real> &design_variables,
        Vector<Real> &lagrange_mult,
        const Vector<Real> &search_direction,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        AlgorithmState<Real> &algo_state ) override
    {
        std::cout << __PRETTY_FUNCTION__ << std::endl;
        BoundConstraint<Real> bound_constraints;
        bound_constraints.deactivate();
        update( design_variables, lagrange_mult, search_direction, objective, equal_constraints, bound_constraints, algo_state );
    }
    void update (
        Vector<Real> &design_variables,
        Vector<Real> &lagrange_mult,
        const Vector<Real> &search_direction,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        BoundConstraint< Real > &bound_constraints,
        AlgorithmState<Real> &algo_state ) override
    {
        std::cout << __PRETTY_FUNCTION__ << std::endl;
        Real tol = std::sqrt(ROL_EPSILON<Real>());
        (void) bound_constraints;
        design_variables.plus(search_direction);
        lagrange_mult.plus(*lagrange_mult_search_direction_);

        // Update StepState
        ROL::Ptr<StepState<Real> > step_state = Step<Real>::getState();
        step_state->descentVec  = design_variables.clone();
        objective.gradient(*(step_state->gradientVec), design_variables, tol);
        equal_constraints.value(*(step_state->constraintVec), design_variables, tol);

        ROL::Ptr<Vector<Real> > lagrangian_gradient = step_state->gradientVec->clone();
        computeLagrangianGradient(*lagrangian_gradient, design_variables, lagrange_mult, *(step_state->gradientVec), equal_constraints);

        algo_state.nfval += step_state->nfval;
        algo_state.ngrad += step_state->ngrad;


        algo_state.value = objective.value(design_variables, tol);
        algo_state.gnorm = lagrangian_gradient->norm();
        algo_state.cnorm = step_state->constraintVec->norm();
        algo_state.snorm = search_direction.norm();
        algo_state.snorm += lagrange_mult_search_direction_->norm();


        const auto current_reduced_gradient = (dynamic_cast<Vector_SimOpt<Real>&>(*lagrangian_gradient)).get_2();
        const auto control_search_direction = (dynamic_cast<const Vector_SimOpt<Real>&>(search_direction)).get_2();
        secant_->updateStorage(design_variables, *current_reduced_gradient, *previous_reduced_gradient_, *control_search_direction, algo_state.snorm, algo_state.iter+1);
        previous_reduced_gradient_ = current_reduced_gradient;

        std::cout
        << " algo_state.value: "  <<   algo_state.value
        << " algo_state.gnorm: "  <<   algo_state.gnorm
        << " algo_state.cnorm: "  <<   algo_state.cnorm
        << " algo_state.snorm: "  <<   algo_state.snorm
        << " algo_state.snorm: "  <<   algo_state.snorm
        << " penalty_value_: "<< penalty_value_;

        algo_state.iterateVec->set(design_variables);
        algo_state.lagmultVec->set(lagrange_mult);
        algo_state.iter++;
    }
  
    /** \brief Print iterate header.
  
        This function produces a string containing header information.
    */
    std::string printHeader( void ) const override
    {
      //std::string head = desc_->printHeader();
      //head.erase(std::remove(head.end()-3,head.end(),'\n'), head.end());
      std::stringstream hist;
      // hist.write(head.c_str(),head.length());
      // hist << std::setw(10) << std::left << "ls_#fval";
      // hist << std::setw(10) << std::left << "ls_#grad";
      hist << "\n";
      return hist.str();
    }
    
    /** \brief Print step name.
  
        This function produces a string containing the algorithmic step information.
    */
    std::string printName( void ) const override
    {
      //std::string name = desc_->printName();
      std::stringstream hist;
      //hist << name;
      hist << "Line Search: " << lineSearchName_;
      hist << " satisfying " << ECurvatureConditionToString(econd_) << "\n";
      return hist.str();
    }
  
    /** \brief Print iterate status.
  
        This function prints the iteration status.
  
        @param[in]     algo_state    is the current state of the algorithm
        @param[in]     printHeader   if set to true will print the header at each iteration
    */
    std::string print( AlgorithmState<Real> & algo_state, bool print_header = false ) const override
    {
      const ROL::Ptr<const StepState<Real> > step_state = Step<Real>::getStepState();
      // std::string desc = desc_->print(algo_state,false);
      // desc.erase(std::remove(desc.end()-3,desc.end(),'\n'), desc.end());
      // std::string name = desc_->printName();
      // size_t pos = desc.find(name);
      // if ( pos != std::string::npos ) {
      //   desc.erase(pos, name.length());
      // }
  
      std::stringstream hist;
      if ( algo_state.iter == 0 ) {
        hist << printName();
      }
      if ( print_header ) {
        hist << printHeader();
      }
      //hist << desc;
      if ( algo_state.iter == 0 ) {
        hist << "\n";
      }
      else {
        hist << std::setw(10) << std::left << step_state->nfval;              
        hist << std::setw(10) << std::left << step_state->ngrad;
        hist << "\n";
      }
      return hist.str();
    }
}; // class FullSpace_BirosGhattas

} // namespace ROL
#endif