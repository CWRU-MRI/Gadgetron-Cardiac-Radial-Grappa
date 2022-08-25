
#include "RadialGrappaCombinedReconstructionGadget.h"
#include "RadialGrappaCombinedReconstruction_GPUfxns.h"
#include "mri_core_grappa.h"
#include "hoNDArray_reductions.h"
#include "hoNDArray_fileio.h"

#include "mri_core_utility.h"
#include "hoMatrix.h"
#include "hoNDArray_linalg.h"
#include "hoNDFFT.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDArray_reductions.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#ifdef USE_OMP
    #include "omp.h"
#endif // USE_OMP


/*
Modified 06/2016 by DNF from GenericReconCartesianGrappaGadget.cpp
modified grappa_2dcalib from mri_core_grappa
*/

namespace Gadgetron {

	static hoNDArray<std::complex<float> > imported_weight_cache;

    RadialGrappaCombinedReconstructionGadget::RadialGrappaCombinedReconstructionGadget() : BaseClass(), imported_weight_set_(imported_weight_cache)
    {
    }

    RadialGrappaCombinedReconstructionGadget::~RadialGrappaCombinedReconstructionGadget()
    {
    }

    int RadialGrappaCombinedReconstructionGadget::process_config(ACE_Message_Block* mb)
    {
        GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

        // -------------------------------------------------

        ISMRMRD::IsmrmrdHeader h;
        try
        {
            deserialize(mb->rd_ptr(), h);
        }
        catch (...)
        {
            GDEBUG("Error parsing ISMRMRD Header");
        }

        size_t NE = h.encoding.size();
        num_encoding_spaces_ = NE;
        GDEBUG_CONDITION_STREAM(verbose.value(), "Number of encoding spaces: " << NE);

        recon_obj_.resize(NE);

	if (h.encoding.size() != 1) {
		GDEBUG("This Gadget only supports one encoding space\n");
		return GADGET_FAIL;
	}

	ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;
	ISMRMRD::EncodingSpace r_space = h.encoding[0].reconSpace;
	ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;
	ISMRMRD::ParallelImaging p_imaging = *h.encoding[0].parallelImaging;

	if(e_space.matrixSize.z > 1){ dataset3D=true; GDEBUG("3D dataset\n");}
	else{ dataset3D = false; GDEBUG("2D dataset\n");}

	recon_dimensions_.push_back(r_space.matrixSize.x);
	recon_dimensions_.push_back(r_space.matrixSize.y);

	field_of_view_.push_back(e_space.fieldOfView_mm.x);
	field_of_view_.push_back(e_space.fieldOfView_mm.y);
	dimensions_.push_back(r_space.matrixSize.x);
	dimensions_.push_back(r_space.matrixSize.y);
	total_slices_ = e_limits.slice->maximum+1;
	fs_E2 = e_space.matrixSize.z;
	
	accel_factor_E1 = p_imaging.accelerationFactor.kspace_encoding_step_1;
	kNRO=grappa_kSize_RO.value();
	kNE1=grappa_kSize_E1.value();
	useGPU=use_gpu_.value();

        return GADGET_OK;
    }

    int RadialGrappaCombinedReconstructionGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
    {
        if (perform_timing.value()) { gt_timer_local_.start("RadialGrappaCombinedReconstructionGadget::process"); }

        process_called_times_++;

        IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();
	size_t us_RO, us_E1, us_E2, us_CH, us_N, us_S, us_LL, fs_E1;

	if(process_called_times_ == 1){
        	if (perform_timing.value()) { gt_timer_.start("RadialGrappaCombinedReconstructionGadget::loading weights"); }
		std::vector<size_t> dim = *(recon_bit_->rbit_[0].data_.data_.get_dimensions());
		us_RO=dim[0];
		us_E1=dim[1];
		us_E2=dim[2];
		us_CH=dim[3];
		us_N=dim[4];
		us_S=dim[5];
		us_LL=dim[6];
		fs_E1 = us_E1*accel_factor_E1;
		//GDEBUG("dimensions of undersampled data: [%d,%d,%d,%d,%d,%d,%d]\n",us_RO,us_E1,us_E2, us_CH, us_N,us_S,us_LL);

		std::vector<size_t> weightDim(6);
		weightDim[0] = kNRO*kNE1*us_CH;
		weightDim[1] = accel_factor_E1-1;
		weightDim[2] = us_CH;
		weightDim[3] = us_E1;
		weightDim[4] = us_RO;
		weightDim[5] = total_slices_;
		
		// cache weights assumes that the weight set is the same between runs, and if the weight set values change, the size of the weight set will change (this is currently the only check that the weights have changed)
		if( !imported_weight_set_.dimensions_equal(&weightDim)){ 
			imported_weight_set_.create(weightDim);
			clear(imported_weight_set_);
			int wssize = kNRO*kNE1*us_CH*(accel_factor_E1-1)*us_CH*us_E1*us_RO*total_slices_;

			FILE *f = fopen("/tmp/gadgetron/debug_out_df/total_weight_set_df.bin", "r");
			if ( f == NULL ) {
				std::cerr << "Cannot open file /tmp/gadgetron/debug_out_df/total_weight_set_df.bin to read!" << std::endl;
				return -1;
			}
			fread(imported_weight_set_.get_data_ptr(), sizeof(std::complex<float>), wssize, f);
			fclose(f);
		}
		
		if (perform_timing.value()) { gt_timer_.stop(); } //loading weights

	}//end first process call

//============= Use GPU======================//
if(useGPU){
//GDEBUG("\n\nUsing GPU for GRAPPA reconstruction, process called times: %d\n", process_called_times_);
	if(process_called_times_ == 1){
		if (perform_timing.value()) { gt_timer_.start("RadialGrappaCombinedReconstructionGadget::move weights"); }
	 	hoNDArray<float_complext>* imp_weights_ = reinterpret_cast< hoNDArray<float_complext>* >(&imported_weight_set_);
	 	device_weights_ = (*imp_weights_);
		dev_weights_ptr = (&device_weights_);
		if (perform_timing.value()) { gt_timer_.stop(); }
	}	

        if (recon_bit_->rbit_.size() > num_encoding_spaces_)
        {
            GWARN_STREAM("Incoming recon_bit has more encoding spaces than the protocol : " << recon_bit_->rbit_.size() << " instead of " << num_encoding_spaces_);
        }

        // for every encoding space
        for (size_t e = 0; e < recon_bit_->rbit_.size(); e++)
        {
            std::stringstream os;
            os << "_encoding_" << e;

            GDEBUG_CONDITION_STREAM(verbose.value(), "Calling " << process_called_times_ << " , encoding space : " << e);
            GDEBUG_CONDITION_STREAM(verbose.value(), "======================================================================");

	//Reconstruction
            if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
            {
                // ---------------------------------------------------------------

                //if (perform_timing.value()) { gt_timer_.start("RadialGrappaCombinedReconstructionGadget::compute_image_header"); }
                this->compute_image_header(recon_bit_->rbit_[e], recon_obj_[e].recon_res_, e);
               // if (perform_timing.value()) { gt_timer_.stop(); }

                // ---------------------------------------------------------------
                
		if (perform_timing.value()) { gt_timer_.start("RadialGrappaCombinedReconstructionGadget::perform_ksp_recon"); }		

		hoNDArray<float_complext>* host_data = reinterpret_cast< hoNDArray<float_complext>* >(&(recon_bit_->rbit_[e].data_.data_));
		hoNDArray<std::complex<float> > host_recon;	
		host_recon.create(us_RO, us_E1*accel_factor_E1, us_E2, us_CH, us_N, us_S, us_LL);
		size_t current_slice_ = recon_obj_[e].recon_res_.headers_[0].slice;
		perform_ksp_recon_gpu(host_data, &host_recon, dev_weights_ptr, e, accel_factor_E1,current_slice_); //recon on gpu
	  	recon_obj_[e].recon_res_.data_.create(us_RO, fs_E1, us_E2, us_CH, us_N, us_S, us_LL);
		Gadgetron::clear(recon_obj_[e].recon_res_.data_);
		recon_obj_[e].recon_res_.data_.copyFrom(host_recon);

		//recon_obj_[e].recon_res_.data_.copyFrom(recon_bit_->rbit_[e].data_.data_); //don't perform recon, just pass undersampled data through

/*
		if (!debug_folder_full_path_.empty())
		{
			std::stringstream os;
			os << process_called_times_;
			std::string suffix = os.str();
			gt_exporter_.export_array_complex(recon_obj_[e].recon_res_.data_, debug_folder_full_path_ + "recon_res_" + suffix);
		}
*/

		if(dataset3D){
			//Zero filling to handle partial Fourier reconstruction
			if(us_E2 != fs_E2){
				size_t pad_size_E2 = us_E2+2*(fs_E2-us_E2);
				hoNDArray<std::complex<float> > padded_recon;
				Gadgetron::pad(us_RO,fs_E1,pad_size_E2,&(recon_obj_[e].recon_res_.data_),&padded_recon);	
				vector_td<size_t,3> crop_offset;
				crop_offset[0] = 0; crop_offset[1] = 0; crop_offset[2] = fs_E2 - us_E2;
				vector_td<size_t,3> crop_size;
				crop_size[0] = us_RO; crop_size[1] = fs_E1; crop_size[2] = fs_E2;
				hoNDArray<std::complex<float> > kspace_buf;
				Gadgetron::crop(crop_offset,crop_size,&padded_recon,&kspace_buf);
				recon_obj_[e].recon_res_.data_.copyFrom(kspace_buf);				
			}
		
			hoNDFFT<float>::instance()->ifft(&(recon_obj_[e].recon_res_.data_),2);
		}//end if(dataset3D)
			
                if (perform_timing.value()) { gt_timer_.stop(); } //peform ksp recon

                // ---------------------------------------------------------------

                //if (perform_timing.value()) { gt_timer_.start("RadialGrappaCombinedReconstructionGadget::compute_image_header"); }
                this->compute_image_header(recon_bit_->rbit_[e], recon_obj_[e].recon_res_, e);
                //if (perform_timing.value()) { gt_timer_.stop(); }

                // ---------------------------------------------------------------

		if (perform_timing.value()) { gt_timer_.start("RadialGrappaCombinedReconstructionGadget::perform_nufft"); }
			perform_nufft(&(recon_obj_[e].recon_res_), process_called_times_, current_slice_);
		if (perform_timing.value()) { gt_timer_.stop(); }
		
		        // ---------------------------------------------------------------				
				
				if(recon_bit_->rbit_[0].data_.headers_(us_E1-1).isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_MEASUREMENT)){ 
					recon_obj_[e].recon_res_.headers_[0].setFlag(57); //DNF: User flag one, set it here to show this is the last scan in the measurement
				}

                // ---------------------------------------------------------------

                //if (perform_timing.value()) { gt_timer_.start("RadialGrappaCombinedReconstructionGadget::send_out_image_array"); }
                //this->send_out_image_array(recon_bit_->rbit_[e], recon_obj_[e].recon_res_, e, image_series.value() + ((int)e + 1), GADGETRON_IMAGE_REGULAR);
                this->send_out_image_array(recon_bit_->rbit_[e], recon_obj_[e].recon_res_, e, current_slice_, GADGETRON_IMAGE_REGULAR);
               //if (perform_timing.value()) { gt_timer_.stop(); }

                // ---------------------------------------------------------------

            }

            recon_obj_[e].recon_res_.data_.clear();
            recon_obj_[e].recon_res_.headers_.clear();
            recon_obj_[e].recon_res_.meta_.clear();

        } //end for (each encoding space)
}//end if(useGPU)

	//=============Don't use GPU======================//

	else{
		GDEBUG("This Gadget is only prepared for using the GPU!!!\n");
	}//end else

        m1->release();

        if (perform_timing.value()) { gt_timer_local_.stop(); } //recon + nufft

        return GADGET_OK;
} //end process

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 void RadialGrappaCombinedReconstructionGadget::perform_nufft(IsmrmrdImageArray* recon_res_, int process_called_times_, int current_slice_)
    {
        try
        {
	if(process_called_times_==1){
		num_slices_=recon_res_->data_.get_size(6);
		num_seg_=recon_res_->data_.get_size(5);
		num_repetitions_=recon_res_->data_.get_size(4);
		num_coils_=recon_res_->data_.get_size(3);
		num_partitions_=recon_res_->data_.get_size(2);
		num_profiles_=recon_res_->data_.get_size(1);
		samples_per_readout_ = recon_res_->data_.get_size(0);

		dimensions_.push_back(num_coils_);
		img_dims.push_back(dimensions_[0]);
		img_dims.push_back(dimensions_[1]);
		img_dims.push_back(num_partitions_);
		img_dims.push_back(num_coils_);
		
		// ========= Get trajectory/dcw for frame ========= //
		//traj = compute_radial_trajectory_fixed_angle_2d<float>(samples_per_readout_,num_profiles_,1,M_PI);
		traj = compute_radial_trajectory_fixed_angle_2d<float>(samples_per_readout_,num_profiles_,1,0);
		dcw = compute_radial_dcw_fixed_angle_2d<float>(samples_per_readout_,num_profiles_,1.0,1.0f/samples_per_readout_/num_profiles_);
	} //end first pass initialization

	// ========= Initialize plan ========= //
		const float kernel_width = 5.5f;
		//const float kernel_width = 3.0f;
		cuNFFT_plan<float,2> plan;
		plan.setup( from_std_vector<size_t,2>(img_dims), from_std_vector<size_t,2>(img_dims), kernel_width );
      		plan.preprocess( traj.get(), cuNFFT_plan<float,2>::NFFT_PREP_NC2C );

		hoNDArray<std::complex<float>> full_im_results;
		full_im_results.create(dimensions_[0],dimensions_[1],num_partitions_,1,num_repetitions_,num_seg_,num_slices_);
		Gadgetron::clear(full_im_results);		

	// ===== Reconstruction ====== //

	for (int ll=0; ll<num_slices_; ll++){
	for (int ss=0; ss<num_seg_; ss++){
	for (int rr=0; rr<num_repetitions_; rr++){

		cuNDArray<float_complext> image(&img_dims);

		// Get samples for frame 
		std::vector<size_t> dims;
		dims.push_back(samples_per_readout_);
		dims.push_back(num_profiles_);
		dims.push_back(num_partitions_);
		dims.push_back(num_coils_);

		hoNDArray<std::complex<float> > tempFrame = hoNDArray<std::complex<float> >(dims, &recon_res_->data_(0,0,0,0,rr,ss,ll));

		boost::shared_ptr< hoNDArray<float_complext> > host_samples(new hoNDArray<float_complext>(dims));
		host_samples->copyFrom(tempFrame);

		cuNDArray<float_complext> samples(host_samples.get());
		plan.compute( &samples, &image, (dcw->get_number_of_elements()>0) ? dcw.get() : 0x0, 	cuNFFT_plan<float, 2>::NFFT_BACKWARDS_NC2C );

/*
	for(unsigned int c=0;c<num_coils_;c++){

		// ========= Get samples for coil 'c' ========= //
		cuNDArray<float_complext> coil_samples;
		std::vector<size_t> coil_sample_dims(1);
		coil_sample_dims[0] = samples_per_readout_*num_profiles_*num_partitions_;
		coil_samples.create(coil_sample_dims);
		float_complext *samples_ptr = samples.get_data_ptr();

		samples_ptr += c*samples_per_readout_*num_profiles_*num_partitions_;
		float_complext *coil_samples_ptr = coil_samples.get_data_ptr();

		if(cudaMemcpy(coil_samples_ptr, samples_ptr, sizeof(float_complext)*samples_per_readout_*num_profiles_*num_partitions_, cudaMemcpyDeviceToDevice) != cudaSuccess){
			throw cuda_error("error copying coil samples\n");
		}

		// ========= Create coil output array ========= //
	      std::vector<size_t> coil_img_dims(3);
	      coil_img_dims[0] = dimensions_[0];
	      coil_img_dims[1] = dimensions_[1];
		coil_img_dims[2] = num_partitions_;
	      cuNDArray<float_complext> coil_image(&coil_img_dims);    

		// ========= Gridder ========= //
	      plan.compute( &coil_samples, & coil_image, (dcw->get_number_of_elements()>0) ? dcw.get() : 0x0, 	cuNFFT_plan<float, 2>::NFFT_BACKWARDS_NC2C );

		// Copy coil image to aggregate images
		float_complext *coil_im_ptr=coil_image.get_data_ptr();
		float_complext *agg_im_ptr=image.get_data_ptr();
		agg_im_ptr += dimensions_[0]*dimensions_[1]*num_partitions_*c;
		if(cudaMemcpy(agg_im_ptr, coil_im_ptr, sizeof(float_complext)*dimensions_[0]*dimensions_[1]*num_partitions_,cudaMemcpyDeviceToDevice) != cudaSuccess){
			throw cuda_error("error copying coil images to aggregate images\n");
		}
	  
	}//end for loop over coils
*/	
/*	
	
{ //adaptive coil combination
	// ========= Download to host ========= //

		hoNDArray<float_complext> ho_image;
		ho_image.create(img_dims[0], img_dims[1], img_dims[2],img_dims[3]);
		Gadgetron::clear(ho_image);

		image.to_host(&ho_image);	
		hoNDArray<std::complex<float> > host_image = *(reinterpret_cast< hoNDArray<std::complex<float> >* >(&ho_image));
		std::complex<float> scale=100;
		host_image*=scale;

	// ========= adaptive coil combination ========= //

	    std::vector<size_t> comb_img_dims(3);
	    comb_img_dims[0] = img_dims[0];
	    comb_img_dims[1] = img_dims[1];
	    comb_img_dims[2] = img_dims[2];
	    
	    //Prepare the output
	    hoNDArray<std::complex<float> > output = hoNDArray<std::complex<float> >(comb_img_dims, &full_im_results(0,0,0,0,rr,ss,ll));	    
	    clear(output);

	     //Calculate the coil maps
		hoNDArray<std::complex<float> > coil_map;
		coil_map.create(img_dims[0],img_dims[1],img_dims[2],img_dims[3]);
		clear(coil_map);
		coil_map_Inati(host_image, coil_map);		

		//Calculate the coil map factors
	 	hoNDArray<std::complex<float> > coil_map_temp;
		coil_map_temp.create(img_dims[0], img_dims[1], img_dims[2], img_dims[3]);
		clear(coil_map_temp);		
		Gadgetron::abs(coil_map, coil_map_temp);
		hoNDArray<std::complex<float> > coil_map_factor;
		coil_map_factor.create(img_dims[0], img_dims[1], img_dims[2]);
		clear(coil_map_factor);
		Gadgetron::sum_over_dimension(coil_map_temp, coil_map_factor,3);
		Gadgetron::multiply(coil_map_factor, coil_map_factor, coil_map_factor);		

		//Perform coil combination
		hoNDArray<std::complex<float> > combined_ims = hoNDArray<std::complex<float> >(img_dims[0], img_dims[1], img_dims[2]);
		Gadgetron::coil_combine(host_image, coil_map, 3, combined_ims);

		//Normalize
		Gadgetron::multiply(combined_ims, coil_map_factor, output);
		
}//end adaptive coil combination	

*/


{ //SOS coil combination
	// ========= Download to host ========= //

		hoNDArray<float_complext> ho_image;
		ho_image.create(img_dims[0], img_dims[1], img_dims[2],img_dims[3]);
		Gadgetron::clear(ho_image);

		image.to_host(&ho_image);	
		hoNDArray<std::complex<float> >* host_image = reinterpret_cast< hoNDArray<std::complex<float> >* >(&ho_image);

	// ========= SOS coil combination ========= //

	    //Square root of the sum of squares
	    std::vector<size_t> comb_img_dims(3);
	    comb_img_dims[0] = img_dims[0];
	    comb_img_dims[1] = img_dims[1];
	    comb_img_dims[2] = img_dims[2];

	    hoNDArray<std::complex<float> > output = hoNDArray<std::complex<float> >(comb_img_dims, &full_im_results(0,0,0,0,rr,ss,ll));
	    
	     //Zero the output
	    clear(output);

	    //Compute d* d in place
	    multiplyConj(host_image,host_image,host_image);  
		 
	    //Add up
	    for (unsigned int c = 0; c < num_coils_; c++) {
		output += hoNDArray<std::complex<float> >(comb_img_dims, &(*host_image)(0,0,0,c));
	    }     
	    
		     
	    //Take the square root in place
	    sqrt_inplace(&output); 		
}//end SOS coil combination	
		
	    recon_res_->headers_[ll*num_seg_*num_repetitions_ + ss*num_repetitions_ + rr].channels=1;
			
	//Modify Dicom names and groups
	recon_res_->meta_[ll*num_seg_*num_repetitions_ + ss*num_repetitions_ + rr].set(GADGETRON_SEQUENCEDESCRIPTION,"RadialGRAPPA");
	recon_res_->headers_[ll*num_seg_*num_repetitions_ + ss*num_repetitions_ + rr].image_series_index = current_slice_;

	}//end for rr
	}//end for ss
	}//end for ll

	recon_res_->data_.copyFrom(full_im_results);

        } //end try

        catch (...)
        {
            GADGET_THROW("Errors happened in GenericReconNonCartesianGrappaGadget::perform_nufft(...) ... ");
        }
}//end perform_nufft

    GADGET_FACTORY_DECLARE(RadialGrappaCombinedReconstructionGadget)
}
