#ifndef FloatToFixPointDFGadget_H_
#define FloatToFixPointDFGadget_H_

#include "Gadget.h"
#include "hoNDArray.h"
#include "ismrmrd/meta.h"
#include "gadgetron_mricore_export.h"

#include <ismrmrd/ismrmrd.h>
#include "GadgetronTimer.h"

namespace Gadgetron
{

    /**
    * This Gadget converts float values to fix point integer format.
    *
    * How the conversion is done will depend on the image type:
    * Magnitude images: Values above 4095 will be clamped.
    * Real or Imag: Values below -2048 and above 2047 will be clamped. Zero will be 2048.
    * Phase: -pi will be 0, +pi will be 4095.
    *
    */

    template <typename T> 
    class EXPORTGADGETSMRICORE FloatToFixPointDFGadget:public Gadget2<ISMRMRD::ImageHeader, hoNDArray< float > >
    {
    public:

        GADGET_DECLARE(FloatToFixPointDFGadget);

        FloatToFixPointDFGadget();
        virtual ~FloatToFixPointDFGadget();
        GadgetronTimer gt_timer_;

    protected:
        GADGET_PROPERTY(max_intensity, T, "Maximum intensity value", std::numeric_limits<T>::max() );
        GADGET_PROPERTY(min_intensity, T, "Minimal intensity value", std::numeric_limits<T>::min());
        GADGET_PROPERTY(intensity_offset, T, "Intensity offset", 0);
        GADGET_PROPERTY(perform_timing, bool, "Whether to perform timing", false);

        T max_intensity_value_;
        T min_intensity_value_;
        T intensity_offset_value_;

        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(GadgetContainerMessage<ISMRMRD::ImageHeader>* m1, GadgetContainerMessage< hoNDArray< float > >* m2);
    };

    class EXPORTGADGETSMRICORE FloatToShortDFGadget :public FloatToFixPointDFGadget < short > 
    {
    public:
        GADGET_DECLARE(FloatToShortDFGadget);

        FloatToShortDFGadget();
        virtual ~FloatToShortDFGadget();
    };

    class EXPORTGADGETSMRICORE FloatToUShortDFGadget :public FloatToFixPointDFGadget < unsigned short >
    {
    public:
        GADGET_DECLARE(FloatToUShortDFGadget);

        FloatToUShortDFGadget();
        virtual ~FloatToUShortDFGadget();
    };

    class EXPORTGADGETSMRICORE FloatToIntDFGadget :public FloatToFixPointDFGadget < int >
    {
    public:
        GADGET_DECLARE(FloatToIntDFGadget);

        FloatToIntDFGadget();
        virtual ~FloatToIntDFGadget();
    };

    class EXPORTGADGETSMRICORE FloatToUIntDFGadget :public FloatToFixPointDFGadget < unsigned int >
    {
    public:
        GADGET_DECLARE(FloatToUIntDFGadget);

        FloatToUIntDFGadget();
        virtual ~FloatToUIntDFGadget();
    };
}

#endif /* FloatToFixPointDFGadget_H_ */
