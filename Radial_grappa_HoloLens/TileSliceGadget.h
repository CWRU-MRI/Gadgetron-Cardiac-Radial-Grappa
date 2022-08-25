/** \file   GenericReconFieldOfViewAdjustmentGadget.h
    \brief  This is the class gadget for both 2DT and 3DT reconstruction, working on the IsmrmrdImageArray.
            This gadget will adjust the image field-of-view and/or image size to the protocol prescribed values.

            This class is a part of general cartesian recon chain.

\author     Hui Xue


TileSliceGadget: Modified 02/2017 by DNF from GenericReconFieldOfViewAdjustmentGadget to make a Gadget that tiles partitions for viewing 3D volumes in real time at the scanner
Also removes partition oversampling when both oversampled slices are at the beginning of the stack (adds a crop offset, rather than cropping out center)
Remove other FOV adjustments

*/

#pragma once

#include "GenericReconBase.h"

namespace Gadgetron {

    class EXPORTGADGETSMRICORE TileSliceGadget : public GenericReconImageBase
    {
    public:
        GADGET_DECLARE(TileSliceGadget);

        typedef GenericReconImageBase BaseClass;

        TileSliceGadget();
        ~TileSliceGadget();

    protected:

        // --------------------------------------------------
        // variables for protocol
        // --------------------------------------------------

        // encoding FOV and recon FOV
        std::vector< std::vector<float> > encoding_FOV_;
        std::vector< std::vector<float> > recon_FOV_;
        // recon size
        std::vector< std::vector<size_t> > recon_size_;

	//DNF added
	bool dataset3D, performTile;
	double exportToVis, cropROOS, sendInit;
	GADGET_PROPERTY(perform_tile,bool,"If true, tile the partitions", true);
	unsigned int numSlices;

        // --------------------------------------------------
        // variable for recon
        // --------------------------------------------------

        // kspace filter
        hoNDArray< std::complex<float> > filter_RO_;
        hoNDArray< std::complex<float> > filter_E1_;
        hoNDArray< std::complex<float> > filter_E2_;

        // kspace buffer
        hoNDArray< std::complex<float> > kspace_buf_;
        hoNDArray< std::complex<float> > kspace_buf2_;

        // results of filtering
        hoNDArray< std::complex<float> > res_;

        // number of times the process function is called
        size_t process_called_times_;

        // --------------------------------------------------
        // functional functions
        // --------------------------------------------------

        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1);

        // close call
        int close(unsigned long flags);

        // adjust FOV
        int adjust_FOV(IsmrmrdImageArray& data);
        
        // perform fft or ifft
        void perform_fft(size_t E2, const hoNDArray< std::complex<float> >& input, hoNDArray< std::complex<float> >& output);
        void perform_ifft(size_t E2, const hoNDArray< std::complex<float> >& input, hoNDArray< std::complex<float> >& output);

	//tile partitions
	int tile_partitions(IsmrmrdImageArray& data);

	//flipud 2D ims
	int flipudlr2d(IsmrmrdImageArray& data);	
	
	//crop image to remove artifact ring
	int crop_artifact_ring(IsmrmrdImageArray& data);

    };
}
