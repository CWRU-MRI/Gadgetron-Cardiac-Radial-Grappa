#pragma once

#include "Gadget.h"
#include "hoNDArray.h"
#include "cuNDArray.h"
#include "basicradial_lib_export.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>
#include "Gadget.h"
#include "GadgetronTimer.h"

#include "ismrmrd/xml.h"
#include "ismrmrd/meta.h"

#include "mri_core_def.h"
#include "mri_core_data.h"
#include "mri_core_utility.h"

#include "ImageIOAnalyze.h"
#include "GenericReconBase.h"
#include "GenericReconGadget.h"


#include "NFFT2DGadget_basic.h"
#include "cuNFFT.h"
#include "vector_td_utilities.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_fileio.h"
#include "cuNDArray_utils.h"
#include "GadgetIsmrmrdReadWrite.h"
#include "ismrmrd/xml.h"
#include "radial_utilities.h"
#include <cmath>

#include "hoNDArray_elemwise.h"

#include "mri_core_utility.h"
#include "mri_core_kspace_filter.h"
#include "mri_core_def.h"


namespace Gadgetron {

    class EXPORTGADGETSMRICORE NFFT2DGadget_basic : public GenericReconImageBase
    {
    public:
        GADGET_DECLARE(NFFT2DGadget_basic);

        typedef GenericReconImageBase BaseClass;


        NFFT2DGadget_basic();
        ~NFFT2DGadget_basic();

GADGET_PROPERTY(perform_timing, bool, "Whether to perform timing", true);
GADGET_PROPERTY(debug_folder, std::string, "If set, the debug output will be written out","");


  protected:

        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1);

  protected:
        
    template<class T> GadgetContainerMessage< hoNDArray<T> >* 
      duplicate_array( GadgetContainerMessage< hoNDArray<T> > *array );        
    
    boost::shared_ptr< hoNDArray<float_complext> > 
      extract_samples_from_queue ( ACE_Message_Queue<ACE_MT_SYNCH> *queue);
    
    boost::shared_ptr< hoNDArray<float> > 
      extract_trajectory_from_queue ( ACE_Message_Queue<ACE_MT_SYNCH> *queue );
    
    void extract_trajectory_and_dcw_from_queue
      ( ACE_Message_Queue<ACE_MT_SYNCH> *queue, cuNDArray<floatd2> *traj, cuNDArray<float> *dcw );

  protected:
    boost::shared_ptr< ACE_Message_Queue<ACE_MT_SYNCH> > frame_readout_queue_;
    boost::shared_ptr< ACE_Message_Queue<ACE_MT_SYNCH> > frame_traj_queue_;
    std::vector<size_t> dimensions_;
    std::vector<float> field_of_view_;
    size_t num_slices_;
    size_t num_seg_;
    size_t num_repetitions_;
    size_t num_coils_;
    size_t num_partitions_;
    size_t num_profiles_;
    size_t samples_per_readout_;
    size_t num_trajectory_dims_; // 2 for trajectories only, 3 for both trajectories + dcw
	std::vector<size_t> img_dims;
	GadgetronTimer gt_timer_;

  };
}
