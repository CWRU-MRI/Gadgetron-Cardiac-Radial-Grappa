
#include "RadialGrappaCombinedReferencePrepGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"


/*
Modified 06/2016 by DNF from GenericCartesianGrappaReferencePrepGadget
*/


namespace Gadgetron {

    RadialGrappaCombinedReferencePrepGadget::RadialGrappaCombinedReferencePrepGadget() : BaseClass()
    {
    }

    RadialGrappaCombinedReferencePrepGadget::~RadialGrappaCombinedReferencePrepGadget()
    {
    }

    int RadialGrappaCombinedReferencePrepGadget::process_config(ACE_Message_Block* mb)
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
        ref_prepared_.resize(NE, false);

        for (size_t e = 0; e < h.encoding.size(); e++)
        {
            ISMRMRD::EncodingSpace e_space = h.encoding[e].encodedSpace;
            ISMRMRD::EncodingSpace r_space = h.encoding[e].reconSpace;
            ISMRMRD::EncodingLimits e_limits = h.encoding[e].encodingLimits;

            GDEBUG_CONDITION_STREAM(verbose.value(), "---> Encoding space : " << e << " <---");
            GDEBUG_CONDITION_STREAM(verbose.value(), "Encoding matrix size: " << e_space.matrixSize.x << " " << e_space.matrixSize.y << " " << e_space.matrixSize.z);
            GDEBUG_CONDITION_STREAM(verbose.value(), "Encoding field_of_view : " << e_space.fieldOfView_mm.x << " " << e_space.fieldOfView_mm.y << " " << e_space.fieldOfView_mm.z);
            GDEBUG_CONDITION_STREAM(verbose.value(), "Recon matrix size : " << r_space.matrixSize.x << " " << r_space.matrixSize.y << " " << r_space.matrixSize.z);
            GDEBUG_CONDITION_STREAM(verbose.value(), "Recon field_of_view :  " << r_space.fieldOfView_mm.x << " " << r_space.fieldOfView_mm.y << " " << r_space.fieldOfView_mm.z);

            if (!h.encoding[e].parallelImaging)
            {
                GDEBUG_STREAM("Parallel Imaging section not found in header");
                calib_mode_[e] = ISMRMRD_noacceleration;
            }
            else
            {

                ISMRMRD::ParallelImaging p_imaging = *h.encoding[0].parallelImaging;
                GDEBUG_CONDITION_STREAM(verbose.value(), "acceFactorE1 is " << p_imaging.accelerationFactor.kspace_encoding_step_1);
                GDEBUG_CONDITION_STREAM(verbose.value(), "acceFactorE2 is " << p_imaging.accelerationFactor.kspace_encoding_step_2);

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

    int RadialGrappaCombinedReferencePrepGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
    {

//DNF: removed much of this because not relevant for radial sampling
        if (perform_timing.value()) { gt_timer_.start("RadialGrappaCombinedReferencePrepGadget::process"); }

        process_called_times_++;

        IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();
        if (recon_bit_->rbit_.size() > num_encoding_spaces_)
        {
            GWARN_STREAM("Incoming recon_bit has more encoding spaces than the protocol : " << recon_bit_->rbit_.size() << " instead of " << num_encoding_spaces_);
        }

        // for every encoding space, prepare the recon_bit_->rbit_[e].ref_
        size_t e;
        for (e = 0; e < recon_bit_->rbit_.size(); e++)
        {
            auto & rbit = recon_bit_->rbit_[e];
            std::stringstream os;
            os << "_encoding_" << e;

            // -----------------------------------------
            // no acceleration mode
            // check the availability of ref data
            // -----------------------------------------
            if (prepare_ref_always.value() || !ref_prepared_[e])
            {
                // if no ref data is set, make copy the ref point from the  data
		//DNF: reference data may be set by flagging the data as calibration data?
                if (!rbit.ref_)
                {

                    rbit.ref_ = rbit.data_;
                    ref_prepared_[e] = true;
                }
            }
            else
            {
                if (ref_prepared_[e])
                {
                    if (rbit.ref_)
                    {
                        // remove the ref
                        rbit.ref_ = boost::none;
                    }
                }

                continue;
            }

            if (!rbit.ref_) continue;

            // useful variables
            hoNDArray< std::complex<float> >& ref = (*rbit.ref_).data_;           

           // ref = ref_calib;
            ref_prepared_[e] = true;

        }

        if (this->next()->putq(m1) < 0)
        {
            GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
            return GADGET_FAIL;
        }

       if (perform_timing.value()) { gt_timer_.stop(); }

        return GADGET_OK;
    }

    GADGET_FACTORY_DECLARE(RadialGrappaCombinedReferencePrepGadget)
}
