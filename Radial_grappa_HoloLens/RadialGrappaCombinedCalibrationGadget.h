/** \file   RadialGrappaCombinedCalibrationGadget.h
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

        /// for combined image channel
        /// convolution kernel, [RO E1 E2 srcCHA - uncombinedCHA dstCHA - uncombinedCHA Nor1 Sor1 SLC]
        hoNDArray<T> kernel_;

        /// coil sensitivity map, [RO E1 E2 dstCHA - uncombinedCHA Nor1 Sor1 SLC]
        hoNDArray<T> coil_map_;

    };
}

namespace Gadgetron {

    class EXPORTGADGETSMRICORE RadialGrappaCombinedCalibrationGadget : public GenericReconGadget
    {
    public:
        GADGET_DECLARE(RadialGrappaCombinedCalibrationGadget);

        typedef GenericReconGadget BaseClass;
        typedef Gadgetron::RadialGrappaCombinedObj< std::complex<float> > ReconObjType;

        RadialGrappaCombinedCalibrationGadget();
        ~RadialGrappaCombinedCalibrationGadget();

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

        // --------------------------------------------------
        // recon step functions
        // --------------------------------------------------

        // if downstream coil compression is used, determine number of channels used and prepare the ref_calib_dst_
        virtual void prepare_down_stream_coil_compression_ref_data(const hoNDArray< std::complex<float> >& ref_src, hoNDArray< std::complex<float> >& ref_coil_map, hoNDArray< std::complex<float> >& ref_dst, size_t encoding);

        // calibration, if only one dst channel is prescribed, the GrappaOne is used
        virtual void perform_calib(IsmrmrdReconBit& recon_bit, ReconObjType& recon_obj, size_t encoding);

	// get multiplication kernel
	void grappa2d_calib_multiplication_kernel(const hoNDArray<std::complex<float>>& acsSrc, const hoNDArray<std::complex<float>>& acsDst, size_t accelFactor, double thres, size_t kRO, size_t kNE1,  size_t start_RO,size_t  end_RO, size_t start_E1, size_t end_E1, size_t rseg, size_t pseg, size_t current_seg_, size_t current_slice_, hoNDArray<std::complex<float>>& convKer);

        // ------------------------------------
        /// DNF added
        // ------------------------------------
	//char filename[256] ="/home/computer/_weights_.txt";
	ofstream myfile;
	int accel_factor_E1;
	bool dataset3D;
	hoNDArray<std::complex<float>> total_weight_set_;
    template <typename T> void RadialGrappaCombined2d_calib(const hoNDArray<T>& acsSrc, const hoNDArray<T>& acsDst, double thres, size_t kRO, const std::vector<int>& kE1, const std::vector<int>& oE1,  size_t start_RO,size_t  end_RO, size_t start_E1, size_t end_E1, size_t rseg, size_t pseg, size_t current_seg_, size_t current_slice_, hoNDArray<T>& ker);
    };
}
