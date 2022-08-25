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

    NFFT2DGadget_basic::NFFT2DGadget_basic() : BaseClass()
    {
    gt_timer_.set_timing_in_destruction(false);
    }

    NFFT2DGadget_basic::~NFFT2DGadget_basic()
    {
    }


  int NFFT2DGadget_basic::process_config(ACE_Message_Block* mb)
  {

        ISMRMRD::IsmrmrdHeader h;
        try
        {
            deserialize(mb->rd_ptr(), h);
        }
        catch (...)
        {
            GDEBUG("Error parsing ISMRMRD Header"); 
        }


	    if (h.encoding.size() != 1) {
	      GDEBUG("This Gadget only supports one encoding space\n");
	      return GADGET_FAIL;
	    }

	    ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;
	    ISMRMRD::EncodingSpace r_space = h.encoding[0].reconSpace;
	    ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;

	    //GDEBUG("Encoding matrix size: %d, %d\nRecon matrix size: %d, %d\n", e_space.matrixSize.x, e_space.matrixSize.y, r_space.matrixSize.x, r_space.matrixSize.y);

	    dimensions_.push_back(r_space.matrixSize.x);
	    dimensions_.push_back(r_space.matrixSize.y);

	    field_of_view_.push_back(e_space.fieldOfView_mm.x);
	    field_of_view_.push_back(e_space.fieldOfView_mm.y);
	    //GDEBUG("FOV: %f, %f\n", r_space.fieldOfView_mm.x, r_space.fieldOfView_mm.y);

    return GADGET_OK;
  }//end process_config

  int NFFT2DGadget_basic::process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1)
  {    
        if (perform_timing.value()) { gt_timer_.start("NFFT2DGadget::process");}
	IsmrmrdImageArray* recon_res_ = m1->getObjectPtr();
	process_called_times_++;

// ===== First pass initialization ====== //

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
	} //end first pass initialization
		
	// ========= Get trajectory/dcw for frame ========= //
	// Assume constant trajectory over time
		boost::shared_ptr< cuNDArray<floatd2> > traj = compute_radial_trajectory_fixed_angle_2d<float>(samples_per_readout_,num_profiles_,1,0);
       		boost::shared_ptr<cuNDArray<float> > dcw = compute_radial_dcw_fixed_angle_2d<float>(samples_per_readout_,num_profiles_,1.0,1.0f/samples_per_readout_/num_profiles_);
	
	// ========= Initialize plan ========= //
		const float kernel_width =5.5f;
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
      
      plan.compute( &coil_samples, & coil_image, (dcw->get_number_of_elements()>0) ? dcw.get() : 0x0, 	cuNFFT_plan<float,2>::NFFT_BACKWARDS_NC2C );

	// Copy coil image to aggregate images
	float_complext *coil_im_ptr=coil_image.get_data_ptr();
	float_complext *agg_im_ptr=image.get_data_ptr();
	agg_im_ptr += dimensions_[0]*dimensions_[1]*num_partitions_*c;
	if(cudaMemcpy(agg_im_ptr, coil_im_ptr, sizeof(float_complext)*dimensions_[0]*dimensions_[1]*num_partitions_,cudaMemcpyDeviceToDevice) != cudaSuccess){
		throw cuda_error("error copying coil images to aggregate images\n");
	}
  
}//end for loop over coils

// ========= Download to host ========= //

	hoNDArray<float_complext> ho_image;
	ho_image.create(img_dims[0], img_dims[1], img_dims[2],img_dims[3]);
	Gadgetron::clear(ho_image);

	image.to_host(&ho_image);	

	hoNDArray<std::complex<float> >* host_image = reinterpret_cast< hoNDArray<std::complex<float> >* >(&ho_image);

//if (perform_timing.value()) { gt_timer_.stop(); } //Gridding

// ========= SOS coil combination ========= //

//if (perform_timing.value()) { gt_timer_.start("NFFTGadget::SOS coil combination"); }
    //Square root of the sum of squares
    std::vector<size_t> comb_img_dims(3);
    comb_img_dims[0] = img_dims[0];
    comb_img_dims[1] = img_dims[1];
    comb_img_dims[2] = img_dims[2];
	
    hoNDArray<std::complex<float> > output = hoNDArray<std::complex<float> >(comb_img_dims, &full_im_results(0,0,0,0,rr,ss,ll));
    
     //Zero out the output
    clear(output);

    //Compute d* d in place
    multiplyConj(host_image,host_image,host_image);  
         
    //Add up
    for (unsigned int c = 0; c < num_coils_; c++) {
        output += hoNDArray<std::complex<float> >(comb_img_dims, &(*host_image)(0,0,0,c));
    }       
             
    //Take the square root in place
    sqrt_inplace(&output); 

//if (perform_timing.value()) { gt_timer_.stop(); } //SOS

	recon_res_->headers_[ll*num_seg_*num_repetitions_ + ss*num_repetitions_ + rr].channels=1;

}//end for rr
}//end for ss
}//end for ll

	recon_res_->data_.copyFrom(full_im_results);

	//send out results
	if (this->next()->putq(m1) == -1){
		GERROR("Failed: NFFT2DGadget, passing on data to next gadget");
		return GADGET_FAIL;
	}

   if (perform_timing.value()) { gt_timer_.stop(); } //process

    return GADGET_OK;
  } //end process


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  template<class T> GadgetContainerMessage< hoNDArray<T> >* NFFT2DGadget_basic::duplicate_array( GadgetContainerMessage< hoNDArray<T> > *array )
  {
    GadgetContainerMessage< hoNDArray<T> > *copy = new GadgetContainerMessage< hoNDArray<T> >();   
    *(copy->getObjectPtr()) = *(array->getObjectPtr());
    return copy;
  }
  
  boost::shared_ptr< hoNDArray<float_complext> > NFFT2DGadget_basic::extract_samples_from_queue ( ACE_Message_Queue<ACE_MT_SYNCH> *queue)                                             
  {    
    if(!queue) {
      GDEBUG("Illegal queue pointer, cannot extract samples\n");
      throw std::runtime_error("NFFT2DGadget_basic::extract_samples_from_queue: illegal queue pointer");	
    }

    unsigned int readouts_buffered = queue->message_count();
    
    std::vector<size_t> dims;
    dims.push_back(samples_per_readout_*readouts_buffered);
    dims.push_back(num_coils_);
    
    boost::shared_ptr< hoNDArray<float_complext> > host_samples(new hoNDArray<float_complext>(dims));
    
    for (unsigned int p=0; p<readouts_buffered; p++) {
      
      ACE_Message_Block* mbq;
      if (queue->dequeue_head(mbq) < 0) {
        GDEBUG("Message dequeue failed\n");
        throw std::runtime_error("NFFT2DGadget_basic::extract_samples_from_queue: dequeing failed");	
      }
      
      GadgetContainerMessage< hoNDArray< std::complex<float> > > *daq = AsContainerMessage<hoNDArray< std::complex<float> > >(mbq);
	
      if (!daq) {
        GDEBUG("Unable to interpret data on message queue\n");
        throw std::runtime_error("NFFT2DGadget_basic::extract_samples_from_queue: failed to interpret data");	
      }
	
      for (unsigned int c = 0; c < num_coils_; c++) {
	
        float_complext *data_ptr = host_samples->get_data_ptr();
        data_ptr += c*samples_per_readout_*readouts_buffered+p*samples_per_readout_;
	    
        std::complex<float> *r_ptr = daq->getObjectPtr()->get_data_ptr();
        r_ptr += c*daq->getObjectPtr()->get_size(0);
	  
        memcpy(data_ptr, r_ptr, samples_per_readout_*sizeof(float_complext));
      }

      mbq->release();

    }
    
    return host_samples;
  }

  boost::shared_ptr< hoNDArray<float> > NFFT2DGadget_basic::extract_trajectory_from_queue ( ACE_Message_Queue<ACE_MT_SYNCH> *queue )
  {    
    if(!queue) {
      GDEBUG("Illegal queue pointer, cannot extract trajectory\n");
      throw std::runtime_error("NFFT2DGadget_basic::extract_trajectory_from_queue: illegal queue pointer");	
    }

    unsigned int readouts_buffered = queue->message_count();
    
    std::vector<size_t> dims;
    dims.push_back(num_trajectory_dims_); // 2 for trajectories only, 3 for both trajectories + dcw
    dims.push_back(samples_per_readout_);
    dims.push_back(readouts_buffered);
    
    boost::shared_ptr< hoNDArray<float> > host_traj(new hoNDArray<float>(&dims));
    
    for (unsigned int p=0; p<readouts_buffered; p++) {      
      ACE_Message_Block* mbq;
      if (queue->dequeue_head(mbq) < 0) {
        GDEBUG("Message dequeue failed\n");
        throw std::runtime_error("NFFT2DGadget_basic::extract_trajectory_from_queue: dequeing failed");	
      }
      
      GadgetContainerMessage< hoNDArray<float> > *daq = AsContainerMessage<hoNDArray<float> >(mbq);
	
      if (!daq) {
        GDEBUG("Unable to interpret data on message queue\n");
        throw std::runtime_error("NFFT2DGadget_basic::extract_trajectory_from_queue: failed to interpret data");	
      }

      float *data_ptr = host_traj->get_data_ptr();
      data_ptr += num_trajectory_dims_*samples_per_readout_*p;
      
      float *r_ptr = daq->getObjectPtr()->get_data_ptr();
      
      memcpy(data_ptr, r_ptr, num_trajectory_dims_*samples_per_readout_*sizeof(float));
      
      mbq->release();
    }
    
    return host_traj;
  }

  void NFFT2DGadget_basic::extract_trajectory_and_dcw_from_queue
  ( ACE_Message_Queue<ACE_MT_SYNCH> *queue, cuNDArray<floatd2> *traj, cuNDArray<float> *dcw )
  {
    // Extract trajectory and (if present) density compensation weights.
    // They are stored as a float array of dimensions: {2,3} x #samples_per_readout x #readouts.
    // We need
    // - a floatd2 trajectory array 
    // - a float dcw array 
    //
        
    if( num_trajectory_dims_ == 2 ){
      boost::shared_ptr< hoNDArray<float> > host_traj = extract_trajectory_from_queue( queue );
      std::vector<size_t> dims_1d; dims_1d.push_back(host_traj->get_size(1)*host_traj->get_size(2));
      hoNDArray<floatd2> host_traj2(&dims_1d,(floatd2*)host_traj->get_data_ptr());
      *traj = cuNDArray<floatd2>(host_traj2);

    }
    else{

      boost::shared_ptr< hoNDArray<float> > host_traj_dcw = extract_trajectory_from_queue( queue );

      std::vector<size_t> order;
      order.push_back(1); order.push_back(2); order.push_back(0);
      
      boost::shared_ptr< hoNDArray<float> > host_traj_dcw_shifted = permute( host_traj_dcw.get(), &order );
      
      std::vector<size_t> dims_1d;
      dims_1d.push_back(host_traj_dcw_shifted->get_size(0)*host_traj_dcw_shifted->get_size(1));
      
      hoNDArray<float> tmp(&dims_1d, host_traj_dcw_shifted->get_data_ptr()+2*dims_1d[0]);
      *dcw = tmp;
      
      std::vector<size_t> dims_2d = dims_1d; dims_2d.push_back(2);
      order.clear(); order.push_back(1); order.push_back(0);
      
      tmp.create(&dims_2d, host_traj_dcw_shifted->get_data_ptr());
      auto _traj = permute( &tmp, &order );
      hoNDArray<floatd2> tmp2(&dims_1d,(floatd2*)_traj->get_data_ptr());
      
      *traj = cuNDArray<floatd2>(tmp2);
    }

    std::vector<size_t >dims_2d;
    dims_2d.push_back(traj->get_number_of_elements());
    dims_2d.push_back(1); // Number of frames

    traj->reshape(&dims_2d);
    if( num_trajectory_dims_ == 3 ) dcw->reshape(&dims_2d);
  }

  GADGET_FACTORY_DECLARE(NFFT2DGadget_basic)
}
