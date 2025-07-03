#include "parameters/parameters_boundary_layer_extraction.h"

namespace PHiLiP {
namespace Parameters {

// boundary layer extraction inputs
BoundaryLayerExtractionParam::BoundaryLayerExtractionParam () {}

void BoundaryLayerExtractionParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("boundary_layer_extraction");
    {
        prm.declare_entry("number_of_sampling", "100",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "Number of sampling points on the extraction line. Default value is 100.");
        prm.declare_entry("extraction_point_x", "1.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "The x coordinate of extraction start point. Default value is 1.0. ");
        prm.declare_entry("extraction_point_y", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "The y coordinate of extraction start point. Default value is 0.0. ");
        prm.declare_entry("extraction_point_z", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "The z coordinate of extraction start point. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_1", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 2. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_2", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 2. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_3", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 3. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_4", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 4. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_5", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 5. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_6", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 6. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_7", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 7. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_8", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 8. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_9", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 9. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_10", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 10. Default value is 0.0. ");
       prm.declare_entry("dy_FFD_25", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 25. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_26", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 26. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_27", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 27. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_28", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 28. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_29", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 29. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_30", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 30. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_31", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 31. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_32", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 32. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_33", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 33. Default value is 0.0. ");
        prm.declare_entry("dy_FFD_34", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "Magnitude of perturbation FFD point 34. Default value is 0.0. ");

    }
    prm.leave_subsection();
}

void BoundaryLayerExtractionParam ::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("boundary_layer_extraction");
    {
        number_of_sampling = prm.get_integer("number_of_sampling");
        extraction_point_x = prm.get_double("extraction_point_x");
        extraction_point_y = prm.get_double("extraction_point_y");
        extraction_point_z = prm.get_double("extraction_point_z");
        dy_FFD_1 = prm.get_double("dy_FFD_1");
        dy_FFD_2 = prm.get_double("dy_FFD_2");
        dy_FFD_3 = prm.get_double("dy_FFD_3");
        dy_FFD_4 = prm.get_double("dy_FFD_4");
        dy_FFD_5 = prm.get_double("dy_FFD_5");
        dy_FFD_6 = prm.get_double("dy_FFD_6");
        dy_FFD_7 = prm.get_double("dy_FFD_7");
        dy_FFD_8 = prm.get_double("dy_FFD_8");
        dy_FFD_9 = prm.get_double("dy_FFD_9");
        dy_FFD_10 = prm.get_double("dy_FFD_10");
        dy_FFD_25 = prm.get_double("dy_FFD_25");
        dy_FFD_26 = prm.get_double("dy_FFD_26");
        dy_FFD_27 = prm.get_double("dy_FFD_27");
        dy_FFD_28 = prm.get_double("dy_FFD_28");
        dy_FFD_29 = prm.get_double("dy_FFD_28");
        dy_FFD_30 = prm.get_double("dy_FFD_30");
        dy_FFD_31 = prm.get_double("dy_FFD_31");
        dy_FFD_32 = prm.get_double("dy_FFD_32");
        dy_FFD_33 = prm.get_double("dy_FFD_33");
        dy_FFD_34 = prm.get_double("dy_FFD_34");

    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace