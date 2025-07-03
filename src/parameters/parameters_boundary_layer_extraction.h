#ifndef __PARAMETERS_BOUNDARY_LAYER_EXTRACTION_H__
#define __PARAMETERS_BOUNDARY_LAYER_EXTRACTION_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to boundary layer extraction
class BoundaryLayerExtractionParam
{
public:
    ///< The number of sampling points on the extraction line.
    int number_of_sampling;

    ///< The x coordinate of extraction start point.
    double extraction_point_x;

    ///< The y coordinate of extraction start point.
    double extraction_point_y;

    ///< The z coordinate of extraction start point.
    double extraction_point_z;

      ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_1;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_2;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_3;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_4;
    ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_5;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_6;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_7;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_8;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_9;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_10;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_25;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_26;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_27;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_28;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_29;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_30;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_31;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_32;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_33;
        ///< Magnitude of perturbation FFD point 5.
    double dy_FFD_34;


    BoundaryLayerExtractionParam (); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif