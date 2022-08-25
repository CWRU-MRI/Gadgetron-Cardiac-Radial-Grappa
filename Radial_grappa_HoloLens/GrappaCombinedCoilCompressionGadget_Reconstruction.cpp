/*
Modified 10/30/2016 from Modified 10/30/2016 by DF from GenericReconEigenChannelGadget
*/

#include "GrappaCombinedCoilCompressionGadget_Reconstruction.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

static hoNDArray<std::complex<float> > imported_KL_coeff_cache;

    GrappaCombinedCoilCompressionGadget_Reconstruction::GrappaCombinedCoilCompressionGadget_Reconstruction() : BaseClass(), imported_KL_coeff_(imported_KL_coeff_cache)
    {
    }

    GrappaCombinedCoilCompressionGadget_Reconstruction::~GrappaCombinedCoilCompressionGadget_Reconstruction()
    {
    }

    int GrappaCombinedCoilCompressionGadget_Reconstruction::process_config(ACE_Message_Block* mb)
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

        calib_mode_.resize(NE, ISMRMRD_noacceleration);

        KLT_.resize(NE);

        for (size_t e = 0; e < h.encoding.size(); e++)
        {
            ISMRMRD::EncodingSpace e_space = h.encoding[e].encodedSpace;
            ISMRMRD::EncodingSpace r_space = h.encoding[e].reconSpace;
            ISMRMRD::EncodingLimits e_limits = h.encoding[e].encodingLimits;

	    total_slices_ = e_limits.slice->maximum+1;

            if (!h.encoding[e].parallelImaging)
            {
                GDEBUG_STREAM("Parallel Imaging section not found in header");
                calib_mode_[e] = ISMRMRD_noacceleration;
            }
            else
            {

                ISMRMRD::ParallelImaging p_imaging = *h.encoding[0].parallelImaging;
                std::string calib = *p_imaging.calibrationMode;

                bool separate = (calib.compare("separate") == 0);
                bool embedded = (calib.compare("embedded") == 0);
                bool external = (calib.compare("external") == 0);
                bool interleaved = (calib.compare("interleaved") == 0);
                bool other = (calib.compare("other") == 0);

                calib_mode_[e] = Gadgetron::ISMRMRD_noacceleration;
                if (p_imaging.accelerationFactor.kspace_encoding_step_1 > 1 || p_imaging.accelerationFactor.kspace_encoding_step_2 > 1)
                {
                    if (interleaved)
                        calib_mode_[e] = Gadgetron::ISMRMRD_interleaved;
                    else if (embedded)
                        calib_mode_[e] = Gadgetron::ISMRMRD_embedded;
                    else if (separate)
                        calib_mode_[e] = Gadgetron::ISMRMRD_separate;
                    else if (external)
                        calib_mode_[e] = Gadgetron::ISMRMRD_external;
                    else if (other)
                        calib_mode_[e] = Gadgetron::ISMRMRD_other;
                }
            }
        }

        return GADGET_OK;
        
    }

    int GrappaCombinedCoilCompressionGadget_Reconstruction::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
    {
        if (perform_timing.value()) { gt_timer_.start("GrappaCombinedCoilCompressionGadget_Reconstruction::process"); }

        process_called_times_++;

        IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();
        if (recon_bit_->rbit_.size() > num_encoding_spaces_)
        {
            GWARN_STREAM("Incoming recon_bit has more encoding spaces than the protocol : " << recon_bit_->rbit_.size() << " instead of " << num_encoding_spaces_);
        }

        // for every encoding space, prepare the recon_bit_->rbit_[e].ref_
        size_t e, n, s, slc;
        for (e = 0; e < recon_bit_->rbit_.size(); e++)
        {
            auto & rbit = recon_bit_->rbit_[e];
            std::stringstream os;
            os << "_encoding_" << e;

            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;

            size_t RO = data.get_size(0);
            size_t E1 = data.get_size(1);
            size_t E2 = data.get_size(2);
            size_t CHA = data.get_size(3);
            size_t N = data.get_size(4);
            size_t S = data.get_size(5);
            size_t SLC = data.get_size(6);

            //GDEBUG_STREAM("GrappaCombinedCoilCompressionGadget_Reconstruction - incoming data array : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");


	//Assume we only have one encoding space

	if(process_called_times_ == 1){
		if (perform_timing.value()) { gt_timer_.start("GrappaCombinedCoilCompressionGadget_Reconstruction::load KL_coeff"); }
	
		std::vector<size_t> coeff_dim(3);
		coeff_dim[0] = CHA;
		coeff_dim[1] = upstream_coil_compression_num_modesKept.value();
		coeff_dim[2] = total_slices_;

		if( !imported_KL_coeff_.dimensions_equal(&coeff_dim)){ 

std::cout<<"\n\n\nLOADING\n"<<std::endl;
		std::vector<size_t> dim = *imported_KL_coeff_.get_dimensions();
		GDEBUG("dimensions of calibration data: ");
		for(int d = 0; d<dim.size(); d++){
			std::cout<<dim[d]<<", ";
		}
		std::cout<<"\n\n\n\n\n\n"<<std::endl;


			imported_KL_coeff_.create(coeff_dim);
			clear(imported_KL_coeff_);


			if (!debug_folder_full_path_.empty()){ 
				gt_exporter_.import_array_complex(imported_KL_coeff_,debug_folder_full_path_ + "KL_coeff");
			}
		}

		if (perform_timing.value()) { gt_timer_.stop(); }

		try
		{
		    size_t n, s, slc;

		    if(KLT_[e].size()!=total_slices_) KLT_[e].resize(total_slices_);
		    for (slc = 0; slc < total_slices_; slc++)
		    {
		        if (KLT_[e][slc].size() != S) KLT_[e][slc].resize(S);
		        for (s = 0; s < S; s++)
		        {
		            if (KLT_[e][slc][s].size() != N) KLT_[e][slc][s].resize(N);
		        }
		    }

		   for(int ss=0; ss<total_slices_; ss++){	
		      hoNDArray< std::complex<float> > M_one_slice; 
			M_one_slice.create(imported_KL_coeff_.get_size(0),imported_KL_coeff_.get_size(1));
			clear(M_one_slice);
			memcpy(&M_one_slice[0],&imported_KL_coeff_(ss*M_one_slice.get_number_of_elements()),sizeof(std::complex<float>)*M_one_slice.get_number_of_elements());
		      KLT_[e][ss][0][0].set_KL_transformation(M_one_slice);
		      KLT_[e][ss][0][0].output_length(M_one_slice.get_size(1));
		   }	
		}
		catch (...)
		{
		    GADGET_THROW("Errors setting KL coeff(...) ... ");
		}

	} //end if process called times = 1


            // apply KL coefficients
	    //int current_slice_ = (process_called_times_-1)%total_slices_; 
	    int current_slice_ = recon_bit_->rbit_[e].data_.headers_[0].idx.slice;
            apply_eigen_channel_coefficients_df(KLT_[e], rbit.data_.data_,current_slice_);
        }

        
        if (this->next()->putq(m1) < 0)
        {
            GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
            return GADGET_FAIL;
        }

	if (perform_timing.value()) { gt_timer_.stop(); }
        return GADGET_OK;
    }


   void GrappaCombinedCoilCompressionGadget_Reconstruction::apply_eigen_channel_coefficients_df(const std::vector< std::vector< std::vector< hoNDKLT<std::complex<float> > > > >& KLT, hoNDArray<std::complex<float> >& data, int current_slice_)
    {
	//apply eigen channel coefficients, code copied from mri_core_utility but modified to take handle one slice at a time
        try
        {
            size_t RO = data.get_size(0);
            size_t E1 = data.get_size(1);
            size_t E2 = data.get_size(2);
            size_t CHA = data.get_size(3);
            size_t N = data.get_size(4);
            size_t S = data.get_size(5);
            size_t SLC = data.get_size(6); //will be one here due to choice of triggering/buffering in this pipeline

            //GADGET_CHECK_THROW(KLT.size() == SLC);

            size_t dstCHA = KLT[0][0][0].output_length();

            hoNDArray<std::complex<float > >dstData;
            dstData.create(RO, E1, E2, dstCHA, N, S, SLC);

            size_t n, s, slc;
            for (slc = 0; slc < SLC; slc++)
            {
                for (s = 0; s < S; s++)
                {
                    size_t s_KLT = s;
                   // if (s_KLT >= KLT[slc].size()) s_KLT = KLT[slc].size()-1;

                    for (n = 0; n < N; n++)
                    {
                        size_t n_KLT = n;
                        //if (n_KLT >= KLT[slc][s_KLT].size()) n_KLT = KLT[slc][s_KLT].size()-1;

                        std::complex<float> * pData = &(data(0, 0, 0, 0, n, s, slc));
                        hoNDArray<std::complex<float> > data_in(RO, E1, E2, CHA, pData);

                        std::complex<float>* pDstData = &(dstData(0, 0, 0, 0, n, s, slc));
                        hoNDArray<std::complex<float> > data_out(RO, E1, E2, dstCHA, pDstData);

                        KLT[current_slice_][s_KLT][n_KLT].transform(data_in, data_out, 3);
                    }
                }
            }

            data = dstData;
        }
        catch (...)
        {
            GADGET_THROW("Errors in apply_eigen_channel_coefficients(...) ... ");
        }
    }



    GADGET_FACTORY_DECLARE(GrappaCombinedCoilCompressionGadget_Reconstruction)
}
