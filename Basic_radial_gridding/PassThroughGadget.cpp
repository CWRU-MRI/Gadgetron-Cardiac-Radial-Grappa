
#include "PassThroughGadget.h"
#include "mri_core_grappa.h"
#include "hoNDArray_reductions.h"
#include "hoNDArray_fileio.h"

/*
Modified 06/2016 by DNF from GenericReconCartesianGrappaGadget.cpp

    The input is IsmrmrdReconData and output is single 2D or 3D ISMRMRD images

    If required, the gfactor map can be sent out

    If the  number of required destination channel is 1, the GrappaONE recon will be performed

    The image number computation logic is implemented in compute_image_number function, which can be overloaded
*/

namespace Gadgetron {

    PassThroughGadget::PassThroughGadget() : BaseClass()
    {
    }

    PassThroughGadget::~PassThroughGadget()
    {
    }

    int PassThroughGadget::process_config(ACE_Message_Block* mb)
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
	dataset3D=dataset_3D.value();

        return GADGET_OK;
    }//end process_config

    int PassThroughGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
    {

        process_called_times_++;

        IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();
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
                // ---------------------------------------------------------------

                this->transfer_data(recon_bit_->rbit_[e], recon_obj_[e], e);
		if(dataset3D){
		hoNDFFT<float>::instance()->ifft(&(recon_obj_[e].recon_res_.data_),2);
		}

                // ---------------------------------------------------------------

                this->compute_image_header(recon_bit_->rbit_[e], recon_obj_[e].recon_res_, e);

                // ---------------------------------------------------------------

                this->send_out_image_array(recon_bit_->rbit_[e], recon_obj_[e].recon_res_, e, image_series.value() + ((int)e + 1), GADGETRON_IMAGE_REGULAR);

            recon_obj_[e].recon_res_.data_.clear();
            recon_obj_[e].recon_res_.headers_.clear();
            recon_obj_[e].recon_res_.meta_.clear();      

            } //end for (every encoding space)  
        m1->release();

        return GADGET_OK;
    }//end process

 void PassThroughGadget::transfer_data(IsmrmrdReconBit& recon_bit, ReconObjType& recon_obj, size_t e)
    {
        try
        {
            typedef std::complex<float> T;

            size_t RO = recon_bit.data_.data_.get_size(0);
            size_t E1 = recon_bit.data_.data_.get_size(1);
            size_t E2 = recon_bit.data_.data_.get_size(2);
            size_t dstCHA = recon_bit.data_.data_.get_size(3);
            size_t N = recon_bit.data_.data_.get_size(4);
            size_t S = recon_bit.data_.data_.get_size(5);
            size_t SLC = recon_bit.data_.data_.get_size(6);

	//GDEBUG("Dimensions of recon_bit.data_.data_: [%d, %d, %d, %d, %d, %d, %d]\n",RO, E1, E2, dstCHA, N, S, SLC);
            
	    recon_obj.recon_res_.data_.create(RO, E1, E2, dstCHA, N, S, SLC);
	//GDEBUG("Dimensions of recon_obj.recon_res_: [%d, %d, %d, %d, %d, %d, %d]\n",RO, E1, E2, dstCHA, N, S, SLC);

	    recon_obj.recon_res_.data_.copyFrom(recon_bit.data_.data_);


        } //end try

        catch (...)
        {
            GADGET_THROW("Errors happened in PassThroughGadget::perform_ksp_recon(...) ... ");
        }
}//end transfer_data
  
    GADGET_FACTORY_DECLARE(PassThroughGadget)
}
