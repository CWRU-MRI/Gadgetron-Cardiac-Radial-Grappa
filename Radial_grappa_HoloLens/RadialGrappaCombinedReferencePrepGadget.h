/** \file   RadialGrappaCombinedReferencePrepGadget.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian reconstruction to prepare the reference data, working on the IsmrmrdReconData.
    \author Hui Xue


Modified 06/2016 by DNF from RadialGrappaCombinedReferencePrep
*/

#pragma once

#include "GenericReconBase.h"

#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"

namespace Gadgetron {

    class EXPORTGADGETSMRICORE RadialGrappaCombinedReferencePrepGadget : public GenericReconDataBase
    {
    public:
        GADGET_DECLARE(RadialGrappaCombinedReferencePrepGadget);

        typedef GenericReconDataBase BaseClass;

        RadialGrappaCombinedReferencePrepGadget();
        ~RadialGrappaCombinedReferencePrepGadget();

        /// ------------------------------------------------------------------------------------
        /// parameters to control the reconstruction
        /// ------------------------------------------------------------------------------------

        /// ref preparation
        /// whether to average all N for ref generation
        /// for the interleaved mode, the sampling times will be counted and used for averaging
        /// it is recommended to set N as the interleaved dimension
        GADGET_PROPERTY(average_all_ref_N, bool, "Whether to average all N for ref generation", true);
        /// whether to average all S for ref generation
        GADGET_PROPERTY(average_all_ref_S, bool, "Whether to average all S for ref generation", false);
        /// whether to update ref for every incoming IsmrmrdReconData; for some applications, we may want to only compute ref data once
        /// if false, the ref will only be prepared for the first incoming IsmrmrdReconData
        GADGET_PROPERTY(prepare_ref_always, bool, "Whether to prepare ref for every incoming IsmrmrdReconData", true);

    protected:

        // --------------------------------------------------
        // variables for protocol
        // --------------------------------------------------

        /// indicate whether ref has been prepared for an encoding space
        std::vector<bool> ref_prepared_;

        // for every encoding space
        // calibration mode
        std::vector<Gadgetron::ismrmrdCALIBMODE> calib_mode_;

        // --------------------------------------------------
        // variable for recon
        // --------------------------------------------------

        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);

    };
}
