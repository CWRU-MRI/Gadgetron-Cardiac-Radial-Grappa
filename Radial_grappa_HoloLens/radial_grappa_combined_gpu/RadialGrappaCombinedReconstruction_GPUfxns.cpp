#include "RadialGrappaCombinedReconstruction_GPUfxns.h"
#include "RadialGrappaCombinedReconstructionGadget.h"
#include "mri_core_data.h"
#include "hoNDArray.h"

#ifdef USE_OMP
#include <omp.h>
#endif

namespace Gadgetron
{

/*
template<class T> int perform_ksp_recon_gpu(IsmrmrdReconBit& recon_bit, T& recon_obj, cuNDArray<complext<float> >* dev_weights_ptr,size_t e, int accel_factor_E1)
    {
	int ret=0;
        try
        {	  
  
            hoNDArray<float_complext>* host_data = reinterpret_cast< hoNDArray<float_complext>* >(&(recon_bit.data_.data_));
	    cuNDArray<complext<float> > device_recon_data_(host_data);
	    cuNDArray<complext<float> >* dev_recon_data_ptr_ = &device_recon_data_;
            size_t RO = device_recon_data_.get_size(0);
            size_t us_E1 = device_recon_data_.get_size(1);
            size_t us_E2 = device_recon_data_.get_size(2);
            size_t dstCHA = device_recon_data_.get_size(3);
            size_t N = device_recon_data_.get_size(4);
            size_t S = device_recon_data_.get_size(5);
            size_t LL = device_recon_data_.get_size(6);
	    size_t fs_E2=us_E2;

//	GDEBUG("Dimensions of device_recon_data_: [%d, %d, %d, %d, %d, %d, %d]\n",RO, us_E1, us_E2, dstCHA, N, S, LL);

           unsigned int wpp = dev_weights_ptr->get_size(0); //weights per point
            unsigned int af_E1_m1 = dev_weights_ptr->get_size(1); //acceleration factor_E1 minus 1

//	GDEBUG("Dimensions of weights on device: [%d, %d, %d, %d, %d, %d]\n", wpp, af_E1_m1, dstCHA, dev_weights_ptr->get_size(3), dev_weights_ptr->get_size(4), dev_weights_ptr->get_size(5));

	    size_t fs_E1 = us_E1*accel_factor_E1;

	  recon_obj.recon_res_.data_.create(RO, fs_E1, fs_E2, dstCHA, N, S, LL);
	Gadgetron::clear(recon_obj.recon_res_.data_);
//	GDEBUG("Dimensions of recon_obj.recon_res_: [%d, %d, %d, %d, %d, %d, %d]\n",RO, fs_E1, fs_E2, dstCHA, N, S, LL);

	cuNDArray<complext<float> > device_recon_res;
	device_recon_res.create(RO, fs_E1, fs_E2, dstCHA, N, S, LL);
	cuNDArray<complext<float> >* dev_recon_res_ptr = (&device_recon_res);

	size_t pts_to_recon_ = RO*fs_E1*fs_E2*dstCHA*N*S*LL;
	
	dim3 blockDim(512,1,1);
        dim3 gridDim((unsigned int) std::ceil((1.0f*pts_to_recon_)/blockDim.x), 1, 1 );

       perform_dot_product<<< gridDim, blockDim>>> (dev_recon_data_ptr_->get_data_ptr(), dev_recon_res_ptr->get_data_ptr(),dev_weights_ptr->get_data_ptr(), pts_to_recon_, RO, us_E1, fs_E1, us_E2, fs_E2, dstCHA, dstCHA, wpp, af_E1_m1);

        cudaError_t err = cudaGetLastError();
        if( err != cudaSuccess ){
          std::cerr << "RadialGrappaCombinedReconstruction_GPUfxns.cu: Unable to perform dot product: " << cudaGetErrorString(err) << std::endl;
          return GADGET_FAIL;
        }

	hoNDArray<complext<float> > tempRes;
	tempRes.create(RO, fs_E1, fs_E2, dstCHA, N, S, LL);
	dev_recon_res_ptr->to_host(&(tempRes));
	hoNDArray<std::complex<float> >* host_recon = reinterpret_cast< hoNDArray<std::complex<float> >* >(&(tempRes));

	recon_obj.recon_res_.data_.copyFrom(*host_recon);

        } //end try

        catch (...)
        {
	    ret=-1;
            GADGET_THROW("Errors happened in GenericReconNonCartesianGrappaGadget::perform_ksp_recon(...) ... ");
        }
	return ret;
    }//end perform_ksp_recon_gpu

template EXPORTGPUCORE int perform_ksp_recon_gpu(IsmrmrdReconBit& recon_bit, Gadgetron::RadialGrappaCombinedObj< std::complex<float> >& recon_obj, cuNDArray<complext<float> >* dev_weights_ptr, size_t e, int accel_factor_E1);

*/

}//end namespace Gadgetron
