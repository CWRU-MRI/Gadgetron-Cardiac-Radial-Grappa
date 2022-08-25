#include "RadialGrappaCombinedReconstruction_GPUfxns.h"
#include "cuNDArray_elemwise.h"
#include "cuNDFFT.h"
#include "GadgetronTimer.h"
#include "GPUTimer.h"
#include "cuNDArray_elemwise.h"
#include "CUBLASContextProvider.h"
#include "hoNDArray_fileio.h"
#include "hoNDArray_utils.h"

#include <cublas_v2.h>
#include <iostream>

namespace Gadgetron {

template <class T>__global__ void perform_dot_product(const T* __restrict__ recon_data_, T* __restrict__ recon_res_, const T* __restrict__ weights_, size_t pts_to_recon_, size_t RO,size_t us_E1,size_t fs_E1,size_t us_E2,size_t fs_E2,size_t srcCHA,size_t dstCHA, size_t wpp, size_t af_E1_m1, size_t current_slice_){


	long idx_in = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx_in < pts_to_recon_) {

	unsigned long ch = idx_in/(fs_E1*RO);
	unsigned long fs_e1 = (idx_in - ch*fs_E1*RO)/RO;
	unsigned long ro = idx_in - ch*fs_E1*RO - fs_e1*RO;

	unsigned long accel_factor_e1 = fs_E1/us_E1;
	unsigned long us_e1 = fs_e1/accel_factor_e1; //integer division, so this is okay to id the most recently collected projection
	unsigned long res_offset = ch*fs_E1*RO + fs_e1*RO+ro;

	if(fs_e1%accel_factor_e1 == 0){
		unsigned long data_offset = ch*us_E1*RO + us_e1*RO + ro;
		recon_res_[res_offset] = recon_data_[data_offset];
	}//this projection was collected

	else{			
		recon_res_[res_offset] = 0;
		complext<float> pt1=0;
		complext<float> pt2=0;
		complext<float> pt3=0;
		complext<float> pt4=0;
		complext<float> pt5=0;
		complext<float> pt6=0;

		unsigned long e1_offset = fs_e1%accel_factor_e1; //offset from preceeding collected projection

		for(unsigned long cc = 0; cc<srcCHA; cc++){
			//note hard-coded assumption of kernel size
			unsigned long slice_offset_ = current_slice_*RO*us_E1*dstCHA*af_E1_m1*wpp;
			unsigned long w_offset = slice_offset_ + ro*us_E1*dstCHA*af_E1_m1*wpp + us_e1*dstCHA*af_E1_m1*wpp + ch*af_E1_m1*wpp + (e1_offset-1)*wpp + cc*6; 
			unsigned long inner_def_offset = cc*us_E1*RO;			
			if(us_e1==(us_E1-1)){

				pt1= recon_data_[inner_def_offset + us_e1*RO + ro-1] * weights_[w_offset + 0];
				pt2= recon_data_[inner_def_offset + us_e1*RO + ro] * weights_[w_offset + 2];
				pt3= recon_data_[inner_def_offset + us_e1*RO + ro + 1] * weights_[w_offset + 4];
				pt4= recon_data_[inner_def_offset + (0) + (RO-1)-(ro-1)] * weights_[w_offset + 1];
				pt5= recon_data_[inner_def_offset + (0) + (RO-1)-ro] * weights_[w_offset + 3];
				pt6= recon_data_[inner_def_offset + (0) + (RO-1)-(ro+1)] * weights_[w_offset + 5];

			} //edge projections

			else{
				pt1= recon_data_[inner_def_offset + us_e1*RO + ro-1] * weights_[w_offset + 0];
				pt2= recon_data_[inner_def_offset + us_e1*RO + ro] * weights_[w_offset + 2];
				pt3= recon_data_[inner_def_offset + us_e1*RO + ro + 1] * weights_[w_offset + 4];
				pt4= recon_data_[inner_def_offset + (us_e1*RO + RO) + ro-1] * weights_[w_offset + 1];
				pt5= recon_data_[inner_def_offset + (us_e1*RO + RO) + ro] * weights_[w_offset + 3];
				pt6= recon_data_[inner_def_offset + (us_e1*RO + RO) + ro+1] * weights_[w_offset + 5];

			} //not edge projections

		recon_res_[res_offset] += (pt1+pt2+pt3+pt4+pt5+pt6);	

		}//end for over ch	

	}//end else

  	}//end if (data pt in kspace)
}//end fxn


int perform_ksp_recon_gpu(hoNDArray<float_complext>* host_data, hoNDArray<std::complex<float> >* host_recon, cuNDArray<complext<float> >* dev_weights_ptr,size_t e, int accel_factor_E1, size_t current_slice_)
    {
	int ret=0;
        try
        {	  

	size_t RO = host_data->get_size(0);
	size_t us_E1 = host_data->get_size(1);
	size_t us_E2 = host_data->get_size(2);
	size_t dstCHA = host_data->get_size(3);
	size_t N = host_data->get_size(4);
	size_t S = host_data->get_size(5);
	size_t LL = host_data->get_size(6);
	size_t fs_E2=us_E2;
	unsigned int wpp = dev_weights_ptr->get_size(0); //weights per point
	unsigned int af_E1_m1 = dev_weights_ptr->get_size(1); //acceleration factor_E1 minus 1
	size_t fs_E1 = us_E1*accel_factor_E1;

	std::vector<size_t> perm_dims;
	perm_dims.push_back(0);
	perm_dims.push_back(1);
	perm_dims.push_back(3);
	perm_dims.push_back(2);
	perm_dims.push_back(4); perm_dims.push_back(5); perm_dims.push_back(6);
	boost::shared_ptr<hoNDArray<float_complext> > host_data_perm = permute(host_data,&perm_dims);

	hoNDArray<complext<float> > tempRes_all_partitions;
	tempRes_all_partitions.create(RO, fs_E1, dstCHA, fs_E2, N, S, LL);

	for (size_t part = 0; part<fs_E2; part++){

		hoNDArray<float_complext> host_data_partition;
		host_data_partition.create(RO, us_E1,dstCHA, 1, N, S, LL);

		std::vector<size_t> start;
		start.push_back(0); start.push_back(0); start.push_back(0); start.push_back(part); 
		start.push_back(0); start.push_back(0); start.push_back(0);
	
		std::vector<size_t> size;
		for (int dims=0; dims<7; dims++){
			size.push_back(host_data_perm->get_size(dims));
		}
		size[3]=1;
		host_data_perm->get_sub_array(start,size,host_data_partition);

		cuNDArray<complext<float> > device_recon_data_(&host_data_partition);
		cuNDArray<complext<float> >* dev_recon_data_ptr_ = &device_recon_data_;	

		cuNDArray<complext<float> > device_recon_res;
		device_recon_res.create(RO, fs_E1, dstCHA, 1, N, S, LL);
		cuNDArray<complext<float> >* dev_recon_res_ptr = (&device_recon_res);

		size_t pts_to_recon_ = RO*fs_E1*dstCHA*1*N*S*LL;
	
		dim3 blockDim(512,1,1);
		dim3 gridDim((unsigned int) std::ceil((1.0f*pts_to_recon_)/blockDim.x), 1, 1 );

	       perform_dot_product<<< gridDim, blockDim >>> (dev_recon_data_ptr_->get_data_ptr(), dev_recon_res_ptr->get_data_ptr(),dev_weights_ptr->get_data_ptr(), pts_to_recon_, RO, us_E1, fs_E1, 1, 1, dstCHA, dstCHA, wpp, af_E1_m1,current_slice_);

		cudaError_t err = cudaGetLastError();
		if( err != cudaSuccess ){
		  std::cerr << "RadialGrappaCombinedReconstruction_GPUfxns.cu: Unable to perform dot product: " << cudaGetErrorString(err) << std::endl;
		  ret=-1;
		}

		cudaThreadSynchronize();

		err = cudaGetLastError();
		if( err != cudaSuccess ){
		  std::cerr << "\n\nRadialGrappaCombinedReconstruction_GPUfxns.cu: Unable to sync threads: " << cudaGetErrorString(err) << std::endl;
		  ret=-1;
		}

		hoNDArray<complext<float> > tempRes = hoNDArray<complext<float> >(dev_recon_res_ptr->get_dimensions(), &tempRes_all_partitions(0,0,0, part,0,0,0));
		dev_recon_res_ptr->to_host(&(tempRes));

	}//end for over partitions


	hoNDArray<std::complex<float> >* host_recon_temp = reinterpret_cast< hoNDArray<std::complex<float> >* >(&(tempRes_all_partitions));	
	host_recon->copyFrom(*(permute(host_recon_temp,&perm_dims)));

        } //end try

        catch (...)
        {
	    ret=-1;
            GADGET_THROW("Errors happened in GenericReconNonCartesianGrappaGadget::perform_ksp_recon(...) ... ");
        }
	return ret;
    }//end perform_ksp_recon_gpu

}//end namespace Gadgetron
