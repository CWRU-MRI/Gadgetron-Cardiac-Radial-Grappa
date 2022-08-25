#ifndef ImageArraySplitDFGadget_H
#define ImageArraySplitDFGadget_H

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_mricore_export.h"
#include "GadgetronTimer.h"

#include "mri_core_data.h"

namespace Gadgetron{

  class EXPORTGADGETSMRICORE ImageArraySplitDFGadget : 
  public Gadget1<IsmrmrdImageArray>
    {
    public:
      GADGET_DECLARE(ImageArraySplitDFGadget)
      ImageArraySplitDFGadget();
      GADGET_PROPERTY(perform_timing, bool, "Whether to perform timing", false);
	
    protected:
      virtual int process(GadgetContainerMessage<IsmrmrdImageArray>* m1);
      Gadgetron::GadgetronTimer gt_timer_;
      int process_called_times = 0;
    };
}
#endif //ImageArraySplitDFGadget_H
