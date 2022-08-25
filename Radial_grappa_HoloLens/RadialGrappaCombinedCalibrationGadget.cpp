
#include "RadialGrappaCombinedCalibrationGadget.h"
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

//#ifdef USE_OMP
    #include "omp.h"
//#endif // USE_OMP


/*
Modified 06/2016 by DNF from GenericReconCartesianGrappaGadget.cpp
modified grappa_2dcalib from mri_core_grappa
*/

namespace Gadgetron {

    RadialGrappaCombinedCalibrationGadget::RadialGrappaCombinedCalibrationGadget() : BaseClass()
    {
    }

    RadialGrappaCombinedCalibrationGadget::~RadialGrappaCombinedCalibrationGadget()
    {
    }

    int RadialGrappaCombinedCalibrationGadget::process_config(ACE_Message_Block* mb)
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

	ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;
	ISMRMRD::ParallelImaging p_imaging = *h.encoding[0].parallelImaging;
	if(e_space.matrixSize.z > 1){ dataset3D=true; }
	else{ dataset3D = false; }
	
	accel_factor_E1 = p_imaging.accelerationFactor.kspace_encoding_step_1;

        return GADGET_OK;
    }

    int RadialGrappaCombinedCalibrationGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
    {

        if (perform_timing.value()) { gt_timer_local_.start("RadialGrappaCombinedCalibrationGadget::process"); }

        process_called_times_++;

	if(process_called_times_ == 1){
		IsmrmrdReconData* recon_bit_temp_ = m1->getObjectPtr();
		std::vector<size_t> dim = *(*recon_bit_temp_->rbit_[0].ref_).data_.get_dimensions();
		GDEBUG("dimensions of calibration data: ");
		for(int d = 0; d<dim.size(); d++){
			std::cout<<dim[d]<<", ";
		}
		std::cout<<std::endl;

		//assume same number of channels for source and destination (no coil compression in this stage)
		total_weight_set_.create(grappa_kSize_RO.value()*grappa_kSize_E1.value()*dim[3],(accel_factor_E1-1), dim[3],dim[1]/accel_factor_E1,dim[0],dim[6]);		
		Gadgetron::clear(total_weight_set_);
		GDEBUG("size of total weight set: [kernel size RO*kernel size E1*ncSrc, (af-1), ncDst, np/af,nr,nslice] = [%d,%d,%d,%d,%d,%d]\n", grappa_kSize_RO.value()*grappa_kSize_E1.value()*dim[3],(accel_factor_E1-1),dim[3],dim[1]/accel_factor_E1,dim[0],dim[6]);
	GDEBUG("Acceleration factor used for calibration: %d\n",accel_factor_E1);

	}//end first process call

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


	//Calibration
            if (recon_bit_->rbit_[e].ref_)
            {
                // ---------------------------------------------------------------
                // after this step, the recon_obj_[e].ref_calib_ is set

                if (perform_timing.value()) { gt_timer_.start("RadialGrappaCombinedCalibrationGadget::make_ref_coil_map"); }
		    hoNDArray< std::complex<float> >& ref_data = (*recon_bit_->rbit_[e].ref_).data_;
		    std::vector<size_t> dim = *ref_data.get_dimensions();
		    recon_obj_[e].ref_calib_.create(dim, ref_data.begin());
                if (perform_timing.value()) { gt_timer_.stop(); }

                // ---------------------------------------------------------------
                // after this step, the recon_obj_[e].ref_calib_dst_ is modified.  this step could be used for downstream coil compression
                if (perform_timing.value()) { gt_timer_.start("RadialGrappaCombinedCalibrationGadget::prepare_down_stream_ ref data"); }
                	this->prepare_down_stream_coil_compression_ref_data(recon_obj_[e].ref_calib_, recon_obj_[e].ref_coil_map_, recon_obj_[e].ref_calib_dst_, e);
                if (perform_timing.value()) { gt_timer_.stop(); }

                // ---------------------------------------------------------------

                // after this step, calibration is performed
                if (perform_timing.value()) { gt_timer_.start("RadialGrappaCombinedCalibrationGadget::perform_calib"); }
                	this->perform_calib(recon_bit_->rbit_[e], recon_obj_[e], e);
                if (perform_timing.value()) { gt_timer_.stop(); }
                // ---------------------------------------------------------------

                recon_bit_->rbit_[e].ref_ = boost::none;
            } //end if(reference data)

	//Reconstruction
	//Keep these reconstruction steps to output calibration images
            if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
            {
                // ---------------------------------------------------------------

                if (perform_timing.value()) { gt_timer_.start("RadialGrappaCombinedCalibrationGadget::perform_ksp_recon"); }
		recon_obj_[e].recon_res_.data_.copyFrom(recon_bit_->rbit_[e].data_.data_);
		if(dataset3D){
			hoNDFFT<float>::instance()->ifft(&(recon_obj_[e].recon_res_.data_),2);
		}
                if (perform_timing.value()) { gt_timer_.stop(); }

                // ---------------------------------------------------------------

                if (perform_timing.value()) { gt_timer_.start("RadialGrappaCombinedCalibrationGadget::compute_image_header"); }
                this->compute_image_header(recon_bit_->rbit_[e], recon_obj_[e].recon_res_, e);
                if (perform_timing.value()) { gt_timer_.stop(); }

                // ---------------------------------------------------------------

                if (perform_timing.value()) { gt_timer_.start("RadialGrappaCombinedCalibrationGadget::send_out_image_array"); }
                this->send_out_image_array(recon_bit_->rbit_[e], recon_obj_[e].recon_res_, e, image_series.value() + ((int)e + 1), GADGETRON_IMAGE_REGULAR);
                if (perform_timing.value()) { gt_timer_.stop(); }

                // ---------------------------------------------------------------

            }

            recon_obj_[e].recon_res_.data_.clear();
            recon_obj_[e].recon_res_.headers_.clear();
            recon_obj_[e].recon_res_.meta_.clear();

        } //end for (each encoding space)

        m1->release();

        if (perform_timing.value()) { gt_timer_local_.stop(); }

        return GADGET_OK;
    }

    void RadialGrappaCombinedCalibrationGadget::prepare_down_stream_coil_compression_ref_data(const hoNDArray< std::complex<float> >& ref_src, hoNDArray< std::complex<float> >& ref_coil_map, hoNDArray< std::complex<float> >& ref_dst, size_t e)
    {
        try
        {
            if(!downstream_coil_compression.value())
            {
                GDEBUG_CONDITION_STREAM(verbose.value(), "Downstream coil compression is not prescribed ... ");
                ref_dst = ref_src;
                return;
            }

            if (downstream_coil_compression_thres.value()<0 && downstream_coil_compression_num_modesKept.value()==0)
            {
                GDEBUG_CONDITION_STREAM(verbose.value(), "Downstream coil compression is prescribed to use all input channels ... ");
                ref_dst = ref_src;
                return;
            }

            // determine how many channels to use
            size_t RO = ref_src.get_size(0);
            size_t E1 = ref_src.get_size(1);
            size_t E2 = ref_src.get_size(2);
            size_t CHA = ref_src.get_size(3);
            size_t N = ref_src.get_size(4);
            size_t S = ref_src.get_size(5);
            size_t SLC = ref_src.get_size(6);

            size_t recon_RO = ref_coil_map.get_size(0);
            size_t recon_E1 = ref_coil_map.get_size(1);
            size_t recon_E2 = ref_coil_map.get_size(2);

            std::complex<float>* pRef = const_cast< std::complex<float>* >(ref_src.begin());

            size_t dstCHA = CHA;
            if(downstream_coil_compression_num_modesKept.value()>0 && downstream_coil_compression_num_modesKept.value()<=CHA)
            {
                dstCHA = downstream_coil_compression_num_modesKept.value();
            }
            else
            {
                std::vector<float> E(CHA, 0);
                long long cha;

//#pragma omp parallel default(none) private(cha) shared(RO, E1, E2, CHA, pRef, E)
                {
                    hoNDArray< std::complex<float> > dataCha;
//#pragma omp for 
                    for (cha = 0; cha < (long long)CHA; cha++)
                    {
                        dataCha.create(RO, E1, E2, pRef + cha*RO*E1*E2);
                        float v;
                        Gadgetron::norm2(dataCha, v);
                        E[cha] = v*v;
                    }
                }

                for (cha = 1; cha < (long long)CHA; cha++)
                {
                    if (std::abs(E[cha]) < downstream_coil_compression_thres.value()*std::abs(E[0]))
                    {
                        break;
                    }
                }

                dstCHA = cha;
            }

            GDEBUG_CONDITION_STREAM(verbose.value(), "Downstream coil compression is prescribed to use " << dstCHA << " out of " << CHA << " channels ...");

            if (dstCHA < CHA)
            {
                ref_dst.create(RO, E1, E2, dstCHA, N, S, SLC);
                hoNDArray< std::complex<float> > ref_coil_map_dst;
                ref_coil_map_dst.create(recon_RO, recon_E1, recon_E2, dstCHA, N, S, SLC);

                size_t slc, s, n;
                for (slc = 0; slc < SLC; slc++)
                {
                    for (s = 0; s < S; s++)
                    {
                        for (n = 0; n < N; n++)
                        {
                            std::complex<float>* pDst = &(ref_dst(0, 0, 0, 0, n, s, slc));
                            const std::complex<float>* pSrc = &(ref_src(0, 0, 0, 0, n, s, slc));
                            memcpy(pDst, pSrc, sizeof(std::complex<float>)*RO*E1*E2*dstCHA);

                            pDst = &(ref_coil_map_dst(0, 0, 0, 0, n, s, slc));
                            pSrc = &(ref_coil_map(0, 0, 0, 0, n, s, slc));
                            memcpy(pDst, pSrc, sizeof(std::complex<float>)*recon_RO*recon_E1*recon_E2*dstCHA);
                        }
                    }
                }

                ref_coil_map = ref_coil_map_dst;
            }
            else
            {
                ref_dst = ref_src;
            }
        }
        catch(...)
        {
            GADGET_THROW("Errors happened in RadialGrappaCombinedCalibrationGadget::prepare_down_stream_coil_compression_ref_data(...) ... ");
        }
    }

    void RadialGrappaCombinedCalibrationGadget::perform_calib(IsmrmrdReconBit& recon_bit, ReconObjType& recon_obj, size_t e)
    {
        try
        {
            size_t RO = recon_bit.data_.data_.get_size(0);
            size_t E1 = recon_bit.data_.data_.get_size(1);
            size_t E2 = recon_bit.data_.data_.get_size(2);

            hoNDArray< std::complex<float> >& src = recon_obj.ref_calib_;
            hoNDArray< std::complex<float> >& dst = recon_obj.ref_calib_dst_;

	//hoNDArray< std::complex<float> > src; src.copyFrom(recon_obj.ref_calib_);
	//hoNDArray< std::complex<float> > dst; dst.copyFrom(recon_obj.ref_calib_dst_);

            size_t ref_RO = src.get_size(0);
            size_t ref_E1 = src.get_size(1);
            size_t ref_E2 = src.get_size(2);
            size_t srcCHA = src.get_size(3);
            size_t ref_N = src.get_size(4);
            size_t ref_S = src.get_size(5);
            size_t ref_SLC = src.get_size(6);
            size_t dstCHA = dst.get_size(3);

		GDEBUG(" \n\nrecon_obj.ref_calib_ size: [RO,E1,E2,CHA,N,S,SLC] = [%d,%d,%d,%d,%d,%d,%d]\n\n",ref_RO,ref_E1,ref_E2,srcCHA,ref_N,ref_S,ref_SLC);

                // allocate buffer for kernels
                size_t kRO = grappa_kSize_RO.value();
                size_t kNE1 = grappa_kSize_E1.value();
                size_t kNE2 = grappa_kSize_E2.value();

                size_t convKRO(1), convKE1(1), convKE2(1), oNE1;

		/* Force 2D calibration pattern; assume no undersampling in partition direction, and partitions treated as extra reps of calibration data
                if (E2 > 1)		
                {
                    std::vector<int> kE1, oE1;
                    std::vector<int> kE2, oE2;
                    bool fitItself = true;
                    grappa3d_kerPattern(kE1, oE1, kE2, oE2, convKRO, convKE1, convKE2, accel_factor_E1, (size_t)acceFactorE2_[e], kRO, kNE1, kNE2, fitItself);
			std::cout<<"\n\n 3D kernel pattern used\n\n"<<std::endl;
                }
                else
		*/ 
                {
                    std::vector<int> kE1, oE1;
                    bool fitItself = false;
                    Gadgetron::grappa2d_kerPattern(kE1, oE1, convKRO, convKE1,accel_factor_E1, kRO, kNE1, fitItself);

		    std::cout<<"\nkE1: ["; for (int i=0; i<kE1.size(); i++){std::cout<<kE1[i]<<", ";} std::cout<<"]"<<std::endl;
		    std::cout<<"oE1: [";for (int i=0; i<oE1.size(); i++){std::cout<<oE1[i]<<", ";}std::cout<<"]"<<std::endl;
		    std::cout<<"kEI size: "<<kE1.size() << ", oEI size: "<<oE1.size()<<"\n"<<std::endl;	
	
		oNE1=oE1.size(); 
	
                }

long long num = ref_S * ref_SLC;
long long ii;

for (ii = 0; ii < num; ii++)
{
	size_t slice = ii / ref_S;
	size_t seg = ii - slice*ref_S;

	//still need to handle readout direction edges of kspace

#pragma omp parallel for collapse(2) 
	for (int r = 3; r < ref_RO-3; r++){
	for (int p = 0; p < ref_E1; p += accel_factor_E1){

		//starRO, endRO, startE1, endE1 could be replaced by r,p alone
		size_t startRO = r;
		size_t endRO = r;
		size_t startE1 = p;
		size_t endE1 = p;

		    hoNDArray< std::complex<float> > convKer(kRO*kNE1*srcCHA, oNE1, dstCHA, &(total_weight_set_(0,0,0,p/accel_factor_E1,r,slice)));

grappa2d_calib_multiplication_kernel(src,dst, accel_factor_E1, grappa_reg_lamda.value(), kRO, kNE1, startRO, endRO, startE1, endE1, grappa_segSize_RO.value(),grappa_segSize_E1.value(), seg, slice, convKer);

	}//end for (p < ref_E1)	
	}//end for (r< ref_RO)

}//end for (ii<num)

	// Save Grappa weights to a file
	FILE *f = fopen("/tmp/gadgetron/debug_out_df/total_weight_set_df.bin", "w");
	if ( f == NULL ) {
		std::cerr << "Cannot open file /tmp/gadgetron/debug_out_df/total_weight_set_df.bin to write!" << std::endl;
	}
	fwrite(total_weight_set_.get_data_ptr(), sizeof(std::complex<float>),kRO*kNE1*srcCHA*(accel_factor_E1-1)*dstCHA*ref_E1/accel_factor_E1*ref_RO*ref_S*ref_SLC,f);
	fclose(f);
	GDEBUG("Exported weights_df\n");

        }// end try
        catch (...)
        {
            GADGET_THROW("Errors happened in RadialGrappaCombinedCalibrationGadget::perform_calib(...) ... ");
        }
    } //end perform_calib

void RadialGrappaCombinedCalibrationGadget::grappa2d_calib_multiplication_kernel(const hoNDArray<std::complex<float> >& acsSrc, const hoNDArray<std::complex<float> >& acsDst, size_t accelFactor, double thres, size_t kRO, size_t kNE1, size_t start_RO,size_t  end_RO, size_t start_E1, size_t end_E1, size_t rseg, size_t pseg, size_t current_seg_, size_t current_slice_, hoNDArray<std::complex<float> >& convKer)
{
    try
    {
        std::vector<int> kE1, oE1;
        bool fitItself = false;
        if (&acsSrc != &acsDst) fitItself = true;

        size_t convkRO, convkE1;

	fitItself = false;
        Gadgetron::grappa2d_kerPattern(kE1, oE1, convkRO, convkE1, accelFactor, kRO, kNE1, fitItself);

        hoNDArray<std::complex<float>> ker;
        RadialGrappaCombined2d_calib(acsSrc, acsDst, thres, kRO, kE1, oE1, start_RO, end_RO, start_E1, end_E1, rseg, pseg, current_seg_, current_slice_, ker);

	//allocate the convKer
	convKer.copyFrom(ker);           

    }
    catch (...)
    {
        GADGET_THROW("Errors in grappa2d_calib_multiplication_kernel(...) ... ");
    }
}//end grappa2d_calib_multiplication_kernel


template <typename T> 
void RadialGrappaCombinedCalibrationGadget::RadialGrappaCombined2d_calib(const hoNDArray<T>& acsSrc, const hoNDArray<T>& acsDst, double thres, size_t kRO, const std::vector<int>& kE1, const std::vector<int>& oE1, size_t start_RO, size_t end_RO, size_t start_E1, size_t end_E1, size_t rseg, size_t pseg, size_t current_seg_, size_t current_slice_, hoNDArray<T>& ker)
{
    try
    {

        GADGET_CHECK_THROW(acsSrc.get_size(0)==acsDst.get_size(0));
        GADGET_CHECK_THROW(acsSrc.get_size(1)==acsDst.get_size(1));
        GADGET_CHECK_THROW(acsSrc.get_size(2)>=acsDst.get_size(2));

        size_t RO = acsSrc.get_size(0);
        size_t E1 = acsSrc.get_size(1);
	size_t E2 = acsSrc.get_size(2);
        size_t srcCHA = acsSrc.get_size(3);
        size_t dstCHA = acsDst.get_size(3);
	size_t numReps = acsSrc.get_size(4);
	size_t numSegs = acsSrc.get_size(5);
	size_t numSlices = acsSrc.get_size(6);

	std::vector<int> rseg_indices;
	int index_r=-(rseg/2);
	for (int i=0; i < rseg; i++){
		rseg_indices.push_back(index_r);
		index_r++;
	}

	std::vector<int> pseg_indices;
	int index_p=-(pseg/2);
	for (int i=0; i < pseg; i++){
		pseg_indices.push_back(index_p);
		index_p++;
	}

	//don't handle edges of kspace yet
	int oNE1 = oE1.size(); 
	size_t kE1_size=kE1.size();
	long kernelReps = rseg * pseg * numReps * E2;
	ker.create((accel_factor_E1-1),kRO*kE1_size*srcCHA,dstCHA);
	Gadgetron::clear(ker);

	size_t dimA = kernelReps;
	size_t dimB = kRO*kE1_size*srcCHA;
	size_t dimC = (accel_factor_E1-1)*dstCHA;

	hoMatrix<T> source;	
	hoMatrix<T> target;
	hoMatrix<T> weights(dimB, dimC);

	hoNDArray<T> source_mem(dimA, dimB);
	source.createMatrix(dimA, dimB, source_mem.begin());
	T* pSource = source.begin();

	hoNDArray<T> targ_mem(dimA, dimC);
	target.createMatrix(dimA,dimC, targ_mem.begin());
	T* pTarg = target.begin();

	int count_outer = 0;
		for (int rr=0; rr<rseg; rr++){
		   for (int pp=0; pp<pseg; pp++){
			int rind = rseg_indices[rr] + start_RO;
			int pind = pseg_indices[pp] + start_E1;

			//=========== loop over rep, ch, kernel, target pts b/c can't access ranges in arrays at once
			
			for (int part=0; part<E2; part++){
				for (int rep=0;rep<numReps; rep++){
					for(int ch=0; ch<srcCHA; ch++){
					int count_inner=0;

					   //for (int kro=-2; kro<3; kro += 2){ //Use this if readout oversampling is not removed before GRAPPA
					for (int kro=-1; kro<2; kro += 1){
					   for (int kp=0; kp<accel_factor_E1+1; kp+=accel_factor_E1){
						if(pind+kp == E1){ //trying to access edge uncollected projection (144)
						    source(count_outer + rep + part*numReps, ch*kRO*kE1_size + count_inner) = acsSrc((RO-1)-(rind+kro), 0, part,ch,rep,current_seg_,current_slice_);  
						}
						else{
						    source(count_outer + rep + part*numReps, ch*kRO*kE1_size + count_inner) = acsSrc(rind+kro, pind+kp, part,ch,rep,current_seg_,current_slice_); 
						}
						count_inner++;
						}//end for (kp)
					   }//end for(kro)

					for (int af=0; af<accel_factor_E1-1; af++){
						target(count_outer + rep + part*numReps, ch*(accel_factor_E1-1) + af) = acsDst(rind,pind+(af+1),part,ch,rep,current_seg_,current_slice_);
					}//end for (af)

					}//end for (ch)
				}//end for (rep)
			}//end for (part)
			//=========== loop over rep, ch, kernel, target pts b/c can't access ranges in arrays at once

			count_outer += numReps*E2;

		   }//end for (pp)
		}//end for (rr)

        SolveLinearSystem_Tikhonov(source, target, weights, thres);

std::vector<size_t> reshape_dims;
reshape_dims.push_back(kRO*kE1_size*srcCHA);
reshape_dims.push_back(accel_factor_E1-1);
reshape_dims.push_back(dstCHA);
weights.reshape(&reshape_dims);

ker.copyFrom(weights);

    }
    catch(...)
    {
        GADGET_THROW("Errors in RadialGrappaCombined2d_calib(...) ... ");
    }

    return;
}//end RadialGrappaCombined2d_calib

    GADGET_FACTORY_DECLARE(RadialGrappaCombinedCalibrationGadget)
}
