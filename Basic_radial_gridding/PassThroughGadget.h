/** \file   PassThroughGadget.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian grappa and grappaone reconstruction, working on the IsmrmrdReconData.
    \author Hui Xue


Modified 06/2016 by DNF from GenericReconCartesianGrappaGadget.h 
*/

#pragma once

#include "GenericReconGadget.h"
#include "hoNDArray.h"

namespace Gadgetron {

    /// define the recon status
    template <typename T>
    class EXPORTGADGETSMRICORE PassThroughObj
    {
    public:

        PassThroughObj() {}
        virtual ~PassThroughObj() {}

        // ------------------------------------
        /// recon outputs
        // ------------------------------------
        /// reconstructed images, headers and meta attributes
        IsmrmrdImageArray recon_res_;
    };
}

namespace Gadgetron {

    class EXPORTGADGETSMRICORE PassThroughGadget : public GenericReconGadget
    {
    public:
        GADGET_DECLARE(PassThroughGadget);

        typedef GenericReconGadget BaseClass;
        typedef Gadgetron::PassThroughObj< std::complex<float> > ReconObjType;

        PassThroughGadget();
        ~PassThroughGadget();
	GADGET_PROPERTY(debug_folder, std::string, "If set, the debug output will be written out","");
	GADGET_PROPERTY(dataset_3D, bool, "Was it a 3D dataset?", false);
    protected:

        // --------------------------------------------------
        // variable for recon
        // --------------------------------------------------
        // record the recon kernel, coil maps etc. for every encoding space
        std::vector< ReconObjType > recon_obj_;
	bool dataset3D;

        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);
        virtual void transfer_data(IsmrmrdReconBit& recon_bit, ReconObjType& recon_obj, size_t encoding);

    };
}
