#ifndef ImageFinishExportHoloLensGadget_H
#define ImageFinishExportHoloLensGadget_H

#include "Gadget.h"
#include "hoNDArray.h"
#include "GadgetMRIHeaders.h"
#include "GadgetStreamController.h"
#include "gadgetron_mricore_export.h"

#include <ismrmrd/ismrmrd.h>
#include "ismrmrd/xml.h"
#include "ismrmrd/meta.h"
#include <complex>

#include "ImageIOAnalyze.h"
#include "hoNDArray_fileio.h"

#include "ace/SOCK_Connector.h"
#include "ace/INET_Addr.h"
#include "ace/Log_Msg.h"
#include "GadgetronTimer.h"

namespace Gadgetron{

    class EXPORTGADGETSMRICORE ImageFinishExportHoloLensGadget : public Gadget1 < ISMRMRD::ImageHeader >
    {
    protected:
        virtual int process(GadgetContainerMessage<ISMRMRD::ImageHeader>* m1);
        GadgetronTimer gt_timer_;

	int prep_client(char *hostname, int port);
	int connect_to_server();
	int send_to_server(GadgetContainerMessage<hoNDArray<unsigned short> > *data);
	int send_to_server_transforms(unsigned int current_slice_, float position[], float quat[], bool refreshRender);
	int send_to_server_init(unsigned int matrix_size_[], float field_of_view_[], bool dataset3D, bool catheterPlot, float planar_ST, unsigned int data_buffer_size_);
	char *host;
	int port_init;
	int port_image;
	ACE_SOCK_Stream client_stream_;
	ACE_INET_Addr remote_addr_;
	ACE_SOCK_Connector connector_;
	char filename_socket_address[256] = "/home/mri/Documents/Dominique/Gadgetron_code/socket_addresses/socket_address_image.txt";
	ifstream socket_address_file;
	
	int process_called_times_ = 0;
	GADGET_PROPERTY(perform_timing, bool, "Whether to perform timing", false);
	GADGET_PROPERTY(verbose, bool, "Whether to output debugging statements", false);

    };
}

#endif //ImageFinishExportHoloLensGadget_H
