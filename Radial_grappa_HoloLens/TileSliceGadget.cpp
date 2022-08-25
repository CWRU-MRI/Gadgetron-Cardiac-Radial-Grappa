
#include "TileSliceGadget.h"
#include <iomanip>

#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDFFT.h"

#include "mri_core_utility.h"
#include "mri_core_kspace_filter.h"
#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

    TileSliceGadget::TileSliceGadget() : BaseClass()
    {
    }

    TileSliceGadget::~TileSliceGadget()
    {
    }

    int TileSliceGadget::process_config(ACE_Message_Block* mb)
    {
        GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

        ISMRMRD::IsmrmrdHeader h;
        try
        {
            deserialize(mb->rd_ptr(), h);
        }
        catch (...)
        {
            GDEBUG("Error parsing ISMRMRD Header");
        }

        if (!h.acquisitionSystemInformation)
        {
            GDEBUG("acquisitionSystemInformation not found in header. Bailing out");
            return GADGET_FAIL;
        }

        // -------------------------------------------------

        size_t NE = h.encoding.size();

        num_encoding_spaces_ = NE;

        GDEBUG_CONDITION_STREAM(verbose.value(), "Number of encoding spaces: " << NE);

        // get the encoding FOV and recon FOV

        encoding_FOV_.resize(NE);
        recon_FOV_.resize(NE);
        recon_size_.resize(NE);
        
        ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;
	if(e_space.matrixSize.z > 1){ dataset3D=true; GDEBUG("3D dataset\n");}
	else{ dataset3D = false; GDEBUG("2D dataset\n");}
	
	numSlices = h.encoding[0].encodingLimits.slice->maximum + 1;

        size_t e;
        for (e = 0; e < NE; e++)
        {
            encoding_FOV_[e].resize(3, 0);
            encoding_FOV_[e][0] = h.encoding[e].encodedSpace.fieldOfView_mm.x;
            encoding_FOV_[e][1] = h.encoding[e].encodedSpace.fieldOfView_mm.y;
            encoding_FOV_[e][2] = h.encoding[e].encodedSpace.fieldOfView_mm.z;

            recon_FOV_[e].resize(3, 0);
            recon_FOV_[e][0] = h.encoding[e].reconSpace.fieldOfView_mm.x;
            recon_FOV_[e][1] = h.encoding[e].reconSpace.fieldOfView_mm.y;
            recon_FOV_[e][2] = h.encoding[e].reconSpace.fieldOfView_mm.z;

            recon_size_[e].resize(3, 0);
            recon_size_[e][0] = h.encoding[e].reconSpace.matrixSize.x;
            recon_size_[e][1] = h.encoding[e].reconSpace.matrixSize.y;
            recon_size_[e][2] = h.encoding[e].reconSpace.matrixSize.z;

            GDEBUG_CONDITION_STREAM(verbose.value(), "Encoding space : " << e << " - encoding FOV : [" << encoding_FOV_[e][0] << " " << encoding_FOV_[e][1] << " " << encoding_FOV_[e][2] << " ]");
            GDEBUG_CONDITION_STREAM(verbose.value(), "Encoding space : " << e << " - recon    FOV : [" << recon_FOV_[e][0]    << " " << recon_FOV_[e][1]    << " " << recon_FOV_[e][2] << " ]");
            GDEBUG_CONDITION_STREAM(verbose.value(), "Encoding space : " << e << " - recon    size : [" << recon_size_[e][0] << " " << recon_size_[e][1] << " " << recon_size_[e][2] << " ]");
            
            performTile=perform_tile.value();
            if(dataset3D && performTile){ //NOTE: Check that the automated calculation of FOV works
		    // hard code header values for display at scanner, if data is 3D
		    h.encoding[e].reconSpace.fieldOfView_mm.x = h.encoding[e].reconSpace.fieldOfView_mm.x * 2;
		    h.encoding[e].reconSpace.fieldOfView_mm.y = h.encoding[e].reconSpace.fieldOfView_mm.y*(recon_size_[0][2]/2);
		    h.encoding[e].reconSpace.fieldOfView_mm.z = h.encoding[e].reconSpace.fieldOfView_mm.z;
		    h.encoding[e].reconSpace.matrixSize.x = recon_size_[0][0]*2;
		    h.encoding[e].reconSpace.matrixSize.y = recon_size_[0][1]*recon_size_[0][2]/2;
		    h.encoding[e].reconSpace.matrixSize.z = 1;
            }

		//Set image header userParameter float to exportForVisualization value
		exportToVis = 0;
		cropROOS = 0;
		sendInit = 0;
		ISMRMRD::UserParameters testParam = h.userParameters.get();
		std::vector<ISMRMRD::UserParameterDouble> testDouble = testParam.userParameterDouble;
		for(int m=0; m<testDouble.size(); m++){
			if(testDouble[m].name.compare("exportToVisualization") == 0){
				exportToVis = testDouble[m].value;
			}
			if(testDouble[m].name.compare("sendInit") == 0){
				sendInit = testDouble[m].value;
			}
			if(testDouble[m].name.compare("cropROOS") == 0){
				cropROOS = testDouble[m].value;
			}
		}

        }       

        return GADGET_OK;
    }

    int TileSliceGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1)
    {


        if (perform_timing.value()) { gt_timer_.start("TileSliceGadget::process"); }

        GDEBUG_CONDITION_STREAM(verbose.value(), "TileSliceGadget::process(...) starts ... ");

        process_called_times_++;

        IsmrmrdImageArray* recon_res_ = m1->getObjectPtr();

        // print out recon info
        if (verbose.value())
        {
            GDEBUG_STREAM("----> TileSliceGadget::process(...) has been called " << process_called_times_ << " times ...");
            std::stringstream os;
            recon_res_->data_.print(os);
            GDEBUG_STREAM(os.str());
        }

if(dataset3D){
		// ----------------------------------------------------------
		// FOV adjustment
		// ----------------------------------------------------------
		GADGET_CHECK_RETURN(this->adjust_FOV(*recon_res_) == GADGET_OK, GADGET_FAIL); //remove oversampling in partition direction

		//------------------
		// Tile partitions
		//------------------
		if(performTile){
			GADGET_CHECK_RETURN(this->tile_partitions(*recon_res_) == GADGET_OK, GADGET_FAIL);
		}
	} //end check if dataset3D

	else{
		//GADGET_CHECK_RETURN(this->flipudlr2d(*recon_res_) == GADGET_OK, GADGET_FAIL);
	}//end else (dataset is 2D, not 3D)

	if(cropROOS){
		GADGET_CHECK_RETURN(this->crop_artifact_ring(*recon_res_) == GADGET_OK, GADGET_FAIL);
	}

        // make sure the image header is consistent with data
        size_t N = recon_res_->headers_.get_number_of_elements();
        for (size_t n = 0; n < N; n++)
        {
            recon_res_->headers_(n).matrix_size[0] = recon_res_->data_.get_size(0);
            recon_res_->headers_(n).matrix_size[1] = recon_res_->data_.get_size(1);
            recon_res_->headers_(n).matrix_size[2] = recon_res_->data_.get_size(2);
	    recon_res_->headers_(n).field_of_view[0] = recon_res_->meta_[0].as_double("recon_FOV", 0);
	    recon_res_->headers_(n).field_of_view[1] = recon_res_->meta_[0].as_double("recon_FOV", 1);
	    recon_res_->headers_(n).field_of_view[2] = recon_res_->meta_[0].as_double("recon_FOV", 2);
	    recon_res_->headers_(n).user_float[0] = exportToVis;
	    recon_res_->headers_(n).user_float[1] = sendInit;
	    recon_res_->headers_(n).user_int[0] = numSlices;
        }

        GDEBUG_CONDITION_STREAM(verbose.value(), "TileSliceGadget::process(...) ends ... ");

        // ----------------------------------------------------------
        // send out results
        // ----------------------------------------------------------
        if (this->next()->putq(m1) == -1)
        {
            GERROR("TileSliceGadget::process, passing data on to next gadget");
            return GADGET_FAIL;
        }

        if (perform_timing.value()) { gt_timer_.stop(); }

        return GADGET_OK;
    } //end process


    int TileSliceGadget::adjust_FOV(IsmrmrdImageArray& recon_res)
    {
        try
        {

            size_t RO = recon_res.data_.get_size(0);
            size_t E1 = recon_res.data_.get_size(1);
            size_t E2 = recon_res.data_.get_size(2);
           
            double encodingFOV_RO = recon_res.meta_[0].as_double("encoding_FOV", 0);
            double encodingFOV_E1 = recon_res.meta_[0].as_double("encoding_FOV", 1);
            double encodingFOV_E2 = recon_res.meta_[0].as_double("encoding_FOV", 2);

            double reconFOV_RO = recon_res.meta_[0].as_double("recon_FOV", 0);
            double reconFOV_E1 = recon_res.meta_[0].as_double("recon_FOV", 1);
            double reconFOV_E2 = recon_res.meta_[0].as_double("recon_FOV", 2);

            long encoding = recon_res.meta_[0].as_long("encoding", 0);

            size_t reconSizeRO = recon_size_[encoding][0];
            size_t reconSizeE1 = recon_size_[encoding][1];
            size_t reconSizeE2 = recon_size_[encoding][2];
	
	if(reconSizeE2 != E2){
		size_t diff = E2 - reconSizeE2;
		vector_td<size_t,3> crop_offset;
		crop_offset[0] = 0; crop_offset[1] = 0; crop_offset[2] = diff;
		//crop_offset[0] = 0; crop_offset[1] = 0; crop_offset[2] = diff-1;
		vector_td<size_t,3> crop_size;
		crop_size[0] = RO; crop_size[1] = E1; crop_size[2] = reconSizeE2;
		Gadgetron::crop(crop_offset,crop_size,&recon_res.data_,&kspace_buf_);
		recon_res.data_.copyFrom(kspace_buf_);
	}
	
        }
        catch (...)
        {
            GERROR_STREAM("Errors in TileSliceGadget::adjust_FOV(IsmrmrdImageArray& data) ... ");
            return GADGET_FAIL;
        }

        return GADGET_OK;
    } //end adjustFOV


    int TileSliceGadget::tile_partitions(IsmrmrdImageArray& recon_res)
    {

	try{
            size_t RO = recon_res.data_.get_size(0);
            size_t E1 = recon_res.data_.get_size(1);
            size_t E2 = recon_res.data_.get_size(2);
	std::vector<size_t> reshape_dims;
	reshape_dims.push_back(RO);reshape_dims.push_back(E1*E2);
	recon_res.data_.reshape(reshape_dims);

	hoNDArray< std::complex<float> > temp;
	temp.create(RO*2,E1*E2/2);
	Gadgetron::clear(temp);

	size_t numBytes = sizeof(std::complex<float>);

	size_t source_ind;
	size_t dest_ind;
	for (size_t e2 = 0; e2<E2; e2++){
		for (size_t e1 = 0; e1<E1; e1++){
			source_ind = e2*E1*RO + e1*RO;
			if(e2<std::ceil(E2/2)){
				dest_ind = e2*E1*(RO*2) + e1*RO*2;
			}
			else {
				dest_ind = (e2-std::ceil(E2/2))*E1*RO*2 + e1*RO*2 + RO;
			}
			memcpy(&temp[dest_ind],&(recon_res.data_[source_ind]),numBytes*RO);
		}
	}

	//Flip ud and lr
	hoNDArray< std::complex<float> > temp2;
	temp2.create(RO*2,E1*E2/2);
	Gadgetron::clear(temp2);

	for (size_t e1 = 0; e1<E1*E2/2; e1++){
		for (size_t ro = 0; ro<RO*2; ro++){
			source_ind = e1*RO*2 + ro;
			dest_ind = e1*RO*2 + (RO*2-ro-1);
			temp2(dest_ind) = temp(source_ind);
			//temp2(dest_ind) = std::pow(temp(source_ind),0.8);;
		}	
	}

	hoNDArray< std::complex<float> > temp3;
	temp3.create(RO*2,E1*E2/2);
	Gadgetron::clear(temp3);

	for (size_t e1 = 0; e1<E1*E2/2; e1++){
		source_ind = e1*RO*2;
		dest_ind = (E1*E2/2-e1)*RO*2 - RO*2;
		memcpy(&temp3[dest_ind],&temp2[source_ind],numBytes*RO*2);
	}

		
	recon_res.data_.copyFrom(temp3);

	//change meta data to matching tiling dimensions for display at scanner	
	double sizeRO = RO*2;
	double sizeE1 = E1*E2/2;
	double sizeE2 = 1;
	recon_res.meta_[0].set("recon_matrix",sizeRO);
	recon_res.meta_[0].append("recon_matrix",sizeE1);
	recon_res.meta_[0].append("recon_matrix",sizeE2);
	double fovRO = recon_res.meta_[0].as_double("recon_FOV", 0);
	double fovE1 = fovRO*2;
	double fovE2 = recon_res.meta_[0].as_double("recon_FOV", 2)/E2;
	recon_res.meta_[0].set("recon_FOV",fovRO);
	recon_res.meta_[0].append("recon_FOV",fovE1);
	recon_res.meta_[0].append("recon_FOV",fovE2);			

	}//end try

        catch (...)
        {
            GERROR_STREAM("Errors in TileSliceGadget::tile_partitions(IsmrmrdImageArray& data) ... ");
            return GADGET_FAIL;
        }

        return GADGET_OK;
    } //end tile_partitions


    int TileSliceGadget::flipudlr2d(IsmrmrdImageArray& recon_res)
    {

	try{
            size_t RO = recon_res.data_.get_size(0);
            size_t E1 = recon_res.data_.get_size(1);

	hoNDArray< std::complex<float> > temp;
	temp.create(RO,E1);
	Gadgetron::clear(temp);

	size_t numBytes = sizeof(std::complex<float>);
	size_t source_ind;
	size_t dest_ind;

	//flipud
	for (size_t e1 = 0; e1<E1; e1++){
		for (size_t ro = 0; ro<RO; ro++){
			source_ind = e1*RO + ro;
			dest_ind = e1*RO + (RO-ro-1);
			temp(dest_ind) = recon_res.data_[source_ind];
			//temp(dest_ind) = std::pow(recon_res.data_[source_ind],0.8);;
		}	
	}

	//fliplr
	hoNDArray< std::complex<float> > temp2;
	temp2.create(RO,E1);
	Gadgetron::clear(temp2);

	for (size_t e1 = 0; e1<E1; e1++){
		source_ind = e1*RO;
		dest_ind = (E1-e1)*RO - RO;
		memcpy(&temp2[dest_ind],&temp[source_ind],numBytes*RO);
	}

		
	recon_res.data_.copyFrom(temp2);	

	}//end try

        catch (...)
        {
            GERROR_STREAM("Errors in TileSliceGadget::flipudlr2d(IsmrmrdImageArray& data) ... ");
            return GADGET_FAIL;
        }

        return GADGET_OK;
    } //end flipud2D



    int TileSliceGadget::crop_artifact_ring(IsmrmrdImageArray& recon_res_)
    {

	try{
	
	size_t matrix_size = recon_res_.data_.get_size(0);
	size_t cropped_matrix = std::floor(matrix_size/std::sqrt(2));
	vector_td<size_t,3> crop_size;
	long encoding = recon_res_.meta_[0].as_long("encoding", 0);
	size_t reconSizeE2 = recon_size_[encoding][2];
	crop_size[0] = cropped_matrix; crop_size[1] = cropped_matrix; crop_size[2] = reconSizeE2;
	Gadgetron::crop(crop_size,&recon_res_.data_,&kspace_buf2_);
	recon_res_.data_.copyFrom(kspace_buf2_);	
	
	//change meta data to match cropped dimensions for display at scanner. note E2 does not change, but needs to be defined here because of the meta_[].set method (double check this)
	double matrixRO_old = recon_res_.meta_[0].as_double("recon_matrix", 0);
	double matrixE1_old = recon_res_.meta_[0].as_double("recon_matrix", 1);
	double matrixRO_new = recon_res_.data_.get_size(0);
	double matrixE1_new = recon_res_.data_.get_size(1);
	double matrixE2 = recon_res_.data_.get_size(2);
	recon_res_.meta_[0].set("recon_matrix",matrixRO_new);
	recon_res_.meta_[0].append("recon_matrix",matrixE1_new);
	recon_res_.meta_[0].append("recon_matrix",matrixE2);
	double fovRO_old = recon_res_.meta_[0].as_double("recon_FOV", 0);
	double fovE1_old = recon_res_.meta_[0].as_double("recon_FOV", 1);
	double fovE2 = recon_res_.meta_[0].as_double("recon_FOV", 2);
	double fovRO_new = fovRO_old*(recon_res_.data_.get_size(0)/matrixRO_old);
	double fovE1_new = fovE1_old*(recon_res_.data_.get_size(1)/matrixE1_old);
	recon_res_.meta_[0].set("recon_FOV",fovRO_new);
	recon_res_.meta_[0].append("recon_FOV",fovE1_new);
	recon_res_.meta_[0].append("recon_FOV",fovE2);		

	}//end try

        catch (...)
        {
            GERROR_STREAM("Errors in TileSliceGadget::tile_partitions(IsmrmrdImageArray& data) ... ");
            return GADGET_FAIL;
        }

        return GADGET_OK;
    } //end remove_artifact_ring


    int TileSliceGadget::close(unsigned long flags)
    {
        GDEBUG_CONDITION_STREAM(true, "TileSliceGadget - close(flags) : " << flags);

        if (BaseClass::close(flags) != GADGET_OK) return GADGET_FAIL;

        if (flags != 0)
        {
        }

        return GADGET_OK;
    }

    // ----------------------------------------------------------------------------------------

    GADGET_FACTORY_DECLARE(TileSliceGadget)

}
