/** \file   RadialGrappaCombinedReconstructionGadget.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian grappa and grappaone reconstruction, working on the IsmrmrdReconData.
    \author Hui Xue


Modified 06/2016 by DNF from GenericReconCartesianGrappaGadget.h 
- Instead of calibration for each encoding space, assume there is a 1D-IFFT gadget upstream that performs a 1D-fft in
the partition direction such that we are now reconstructing several slices. 
- The recon object will be sized by slices, not by encoding direction.  This assumes there is no acceleration in the partition direction

*/

#pragma once

#include "GenericReconGadget.h"
#include "hoNDArray.h"
#include <thrust/device_vector.h>
#include "cuNDArray.h"

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

    /// define the recon status
    template <typename T>
    class EXPORTGADGETSMRICORE RadialGrappaCombinedObj
    {
    public:

        RadialGrappaCombinedObj() {}
        virtual ~RadialGrappaCombinedObj() {}

        // ------------------------------------
        /// recon outputs
        // ------------------------------------
        /// reconstructed images, headers and meta attributes
        IsmrmrdImageArray recon_res_;

        // ------------------------------------
        /// buffers used in the recon 
        // ------------------------------------
        /// [RO E1 E2 srcCHA Nor1 Sor1 SLC]
        hoNDArray<T> ref_calib_;
        /// [RO E1 E2 dstCHA Nor1 Sor1 SLC]
        hoNDArray<T> ref_calib_dst_;

        /// reference data ready for coil map estimation
        /// [RO E1 E2 dstCHA Nor1 Sor1 SLC]
        hoNDArray<T> ref_coil_map_;

        /// convolution kernel, [RO E1 E2 srcCHA - uncombinedCHA dstCHA - uncombinedCHA Nor1 Sor1 SLC]
        hoNDArray<T> weight_set_;

        /// coil sensitivity map, [RO E1 E2 dstCHA - uncombinedCHA Nor1 Sor1 SLC]
        hoNDArray<T> coil_map_;

    };
}

namespace Gadgetron {

    class EXPORTGADGETSMRICORE RadialGrappaCombinedReconstructionGadget : public GenericReconGadget
    {
    public:
        GADGET_DECLARE(RadialGrappaCombinedReconstructionGadget);

        typedef GenericReconGadget BaseClass;
        typedef Gadgetron::RadialGrappaCombinedObj< std::complex<float> > ReconObjType;

        RadialGrappaCombinedReconstructionGadget();
        ~RadialGrappaCombinedReconstructionGadget();

        /// ------------------------------------------------------------------------------------
        /// parameters to control the reconstruction
        /// ------------------------------------------------------------------------------------

        /// ------------------------------------------------------------------------------------
        /// image sending
        GADGET_PROPERTY(send_out_gfactor, bool, "Whether to send out gfactor map", false);
        GADGET_PROPERTY(send_out_snr_map, bool, "Whether to send out SNR map", false);

        /// ------------------------------------------------------------------------------------
        /// Grappa parameters
        GADGET_PROPERTY(grappa_kSize_RO, int, "Grappa kernel size RO", 3);
        GADGET_PROPERTY(grappa_kSize_E1, int, "Grappa kernel size E1", 2);
        GADGET_PROPERTY(grappa_kSize_E2, int, "Grappa kernel size E2", 1);
	GADGET_PROPERTY(grappa_segSize_RO, int, "Grappa segment size RO", 5);
	GADGET_PROPERTY(grappa_segSize_E1, int, "Grappa segment size E1", 3);
        //GADGET_PROPERTY(grappa_reg_lamda, double, "Grappa regularization threshold", 0.0005);
	GADGET_PROPERTY(grappa_reg_lamda, double, "Grappa regularization threshold", 0);
        GADGET_PROPERTY(grappa_calib_over_determine_ratio, double, "Grappa calibration overdermination ratio", 45);
	GADGET_PROPERTY(use_gpu_,bool,"If true, recon will try to use GPU resources (when available)", true);

        /// ------------------------------------------------------------------------------------
        /// down stream coil compression
        /// if downstream_coil_compression==true, down stream coil compression is used
        /// if downstream_coil_compression_num_modesKept > 0, this number of channels will be used as the dst channels
        /// if downstream_coil_compression_num_modesKept==0 and downstream_coil_compression_thres>0, the number of dst channels will be determined  by this threshold
        GADGET_PROPERTY(downstream_coil_compression, bool, "Whether to perform downstream coil compression", true);
        GADGET_PROPERTY(downstream_coil_compression_thres, double, "Threadhold for downstream coil compression", 0.002);
        GADGET_PROPERTY(downstream_coil_compression_num_modesKept, size_t, "Number of modes to keep for downstream coil compression", 0);

    protected:

        // --------------------------------------------------
        // variable for recon
        // --------------------------------------------------
        // record the recon kernel, coil maps etc. for every encoding space
        std::vector< ReconObjType > recon_obj_;

        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);


        // ------------------------------------
        /// DNF added
        // ------------------------------------
	hoNDArray<std::complex<float>>& imported_weight_set_;
	int accel_factor_E1, kNRO, kNE1;
 	void perform_nufft(IsmrmrdImageArray* recon_res_, int process_called_times_, int current_slice_);
	bool dataset3D;
	bool useGPU;
	 cuNDArray<float_complext>* dev_weights_ptr;	 
	cuNDArray<float_complext> device_weights_;
	std::vector<size_t> recon_dimensions_;
    	std::vector<float> field_of_view_;
    	boost::shared_ptr< cuNDArray<floatd2> > traj;
    	boost::shared_ptr<cuNDArray<float> > dcw;
	int total_slices_;
	int fs_E2;
 
	//Added for NUFFT

	boost::shared_ptr< ACE_Message_Queue<ACE_MT_SYNCH> > frame_readout_queue_;
	boost::shared_ptr< ACE_Message_Queue<ACE_MT_SYNCH> > frame_traj_queue_;
	size_t num_trajectory_dims_; // 2 for trajectories only, 3 for both trajectories + dcw
	//hoNDArray<std::complex<float>> full_im_results;
	//boost::shared_ptr< cuNDArray<floatd2> > traj;
	//boost::shared_ptr<cuNDArray<float> > dcw;
	std::vector<size_t> img_dims;
	//cuNFFT_plan<float,2> plan;
	std::vector<size_t> dimensions_;
	size_t num_slices_, num_seg_, num_repetitions_, num_coils_, num_partitions_, num_profiles_, samples_per_readout_;

    };

}
