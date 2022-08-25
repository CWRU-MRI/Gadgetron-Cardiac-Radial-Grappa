/** \file   ComplexToFloatDFGadget.h
    \brief  This Gadget converts complex float values to float format.
    \author Hui Xue
*/

#ifndef ComplexToFloatDFGadget_H_
#define ComplexToFloatDFGadget_H_

#include "Gadget.h"
#include "hoNDArray.h"
#include "ismrmrd/meta.h"
#include "gadgetron_mricore_export.h"

#include <ismrmrd/ismrmrd.h>
#include "GadgetronTimer.h"

namespace Gadgetron
{
    class EXPORTGADGETSMRICORE ComplexToFloatDFGadget:public Gadget2<ISMRMRD::ImageHeader, hoNDArray< std::complex<float> > >
    {
    public:

        GADGET_DECLARE(ComplexToFloatDFGadget);

        typedef std::complex<float> ValueType;

        ComplexToFloatDFGadget();
        virtual ~ComplexToFloatDFGadget();
        GADGET_PROPERTY(perform_timing, bool, "Whether to perform timing", false);

    protected:
        virtual int process(GadgetContainerMessage<ISMRMRD::ImageHeader>* m1, GadgetContainerMessage< hoNDArray< ValueType > >* m2);
        GadgetronTimer gt_timer_;
    };
}

#endif // ComplexToFloatDFGadget
