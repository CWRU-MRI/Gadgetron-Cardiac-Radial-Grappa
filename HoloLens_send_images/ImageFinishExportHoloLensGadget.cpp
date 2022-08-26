#include "GadgetIsmrmrdReadWrite.h"
#include "ImageFinishExportHoloLensGadget.h"
#include "mri_core_data.h"
#include "log.h"

#include "hoNDArray_elemwise.h"
#include "hoNDArray_reductions.h"
#include "hoNDArray_math.h"

#include "mri_core_utility.h"
#include "mri_core_kspace_filter.h"
#include "mri_core_def.h"

#include "ImageIOAnalyze.h"
#include "hoNDArray_fileio.h"

#include <iostream>
#include <fstream>
#include <ctime>

#include "hoNDFFT.h"
#include "cuNDFFT.h"

//for ethernet cable communication
#include "ace/SOCK_Connector.h"
#include "ace/INET_Addr.h"
#include "ace/Log_Msg.h"
#include <unistd.h>

//for serial port communication
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>

namespace Gadgetron{

    int ImageFinishExportHoloLensGadget::process(GadgetContainerMessage<ISMRMRD::ImageHeader>* m1)
    {
	process_called_times_ ++;
	gt_timer_.set_timing_in_destruction(false);
     	if (perform_timing.value()) { gt_timer_.start("ImageFinishExportHoloLens::process"); }

     	
     	//NOTE: The information for the following two sections could be collected on the first process call only, and would be valid unless we start implementing real-time scan parameter control
	float * userFloats = m1->getObjectPtr()->user_float;
	float exportToVis = userFloats[0]; //This value is set in the TileSliceGadget.cpp
	float sendInit = userFloats[1]; //This value is set in the TileSliceGadget.cpp
	int * userInts = m1->getObjectPtr()->user_int;
	int num_slices = userInts[0]; //This value is set in the TileSliceGadget.cpp
	bool refreshRender = m1->getObjectPtr()->isFlagSet(57); //This value is set in the SpiralGrappaRealtimeReconstructionGadget of the multislice recon pipeline

     	//Collect initialization information	
     	unsigned int matrix_size_[3] = {};
     	for(int ii=0; ii<3; ii++){
    		 matrix_size_[ii] = m1->getObjectPtr()->matrix_size[ii];
	}

     	bool dataset3D;
     	if(matrix_size_[2] > 1) {
			dataset3D = true;
			num_slices = matrix_size_[2];
		}
     	else {dataset3D = false; matrix_size_[2] = num_slices;}
     	
     	float field_of_view_[3] = {};
     	for(int ii=0; ii<3; ii++){
    		 field_of_view_[ii] = m1->getObjectPtr()->field_of_view[ii];
	}
	
	float planar_ST = field_of_view_[2];	

	unsigned int data_buffer_size_ = matrix_size_[0] * matrix_size_[1] * matrix_size_[2]; //amt of data to send per call to server

	if(verbose.value()){
		std::cout<<"\n\n\nmatrix size: "<<matrix_size_[0]<<", "<<matrix_size_[1]<<", "<<matrix_size_[2]<<std::endl;
		std::cout<<"dataset3D: "<<dataset3D<<std::endl;
		std::cout<<"fov: "<<field_of_view_[0]<<", "<<field_of_view_[1]<<", "<<field_of_view_[2]<<std::endl;
		std::cout<<"planar slice thickness: "<<planar_ST<<std::endl;
		std::cout<<"num_slices: "<<num_slices<<std::endl;
		std::cout<<"data buffer size: "<<data_buffer_size_<<"\n\n"<<std::endl;
	}	

	//Collect position and  information
	unsigned int current_slice_ =  m1->getObjectPtr()->slice;
	unsigned int repetition_ = m1->getObjectPtr()->repetition;


	//THIS IS A TEMPORARY FIX!!!  until all other sequences are updated with correct refreshRender pattern
	if(current_slice_ == (num_slices-1)){
		refreshRender = true;
	}
		 
	float slice_position[3] = {};
	for(int ii=0; ii<3; ii++){
	    slice_position[ii] = m1->getObjectPtr()->position[ii];
	}

	float phase_axis_dir[3] = {};
	for(int ii=0; ii<3; ii++){
	    phase_axis_dir[ii] = m1->getObjectPtr()->phase_dir[ii];
	}

	float read_axis_dir[3] = {};
	for(int ii=0; ii<3; ii++){
	    read_axis_dir[ii] = m1->getObjectPtr()->read_dir[ii];
	}

	float slice_axis_dir[3] = {};
	for(int ii=0; ii<3; ii++){
	    slice_axis_dir[ii] = m1->getObjectPtr()->slice_dir[ii];
	}

	float quat[4] = {};
	ISMRMRD::ismrmrd_directions_to_quaternion(phase_axis_dir,read_axis_dir,slice_axis_dir,quat);

	if(verbose.value()){
		std::cout<<"\n\n\ncurrent slice: "<<current_slice_<<"\nslice_position: "<<slice_position[0]<<", "<<slice_position[1]<<", "<<slice_position[2]<<"\n"<<std::endl;
		std::cout<<"phase_axis directional cosines: "<<phase_axis_dir[0]<<", "<<phase_axis_dir[1]<<", "<<phase_axis_dir[2]<<"\n"<<std::endl;
		std::cout<<"read_axis directional cosines: "<<read_axis_dir[0]<<", "<<read_axis_dir[1]<<", "<<read_axis_dir[2]<<"\n"<<std::endl;
		std::cout<<"slice_axis directional cosines: "<<slice_axis_dir[0]<<", "<<slice_axis_dir[1]<<", "<<slice_axis_dir[2]<<"\n"<<std::endl;
		std::cout<<"quaternion: "<<quat[0]<<", "<<quat[1]<<", "<<quat[2]<<", "<<quat[3]<<"\n\n\n"<<std::endl;		
	}
	

  	GadgetContainerMessage<hoNDArray<unsigned short> >* m2 = AsContainerMessage<hoNDArray<unsigned short> >(m1->cont());
	
	//at this point, image should be of size [nx,ny,nz,nrep]
	std::vector<size_t> dims = *(m2->getObjectPtr()->get_dimensions()); 

	if(exportToVis){
		//Get address of host, and port of socket
		std::string host_name = hololens_ip.value();
		std::string port_name_init = hololens_init_port.value();
		std::string port_name_image =hololens_image_port.value();
		host = new char[host_name.length()+1];
		std::strcpy(host,host_name.c_str());
		port_init = std::stoi(port_name_init);
		port_image = std::stoi(port_name_image);
		//host = "10.10.0.110";
		//port = 4447;

		//Send  to socket, hack to send the max number of slices at beginning
		if(process_called_times_ == 1 && sendInit) {	
			//prep_client(host,port);
			prep_client(host,port_init);
			connect_to_server();
			bool catheterPlot = false;
			matrix_size_[2] = 12;
			send_to_server_init(matrix_size_, field_of_view_, dataset3D, catheterPlot, planar_ST, data_buffer_size_);
			if (client_stream_.close () == -1)
				ACE_ERROR_RETURN ((LM_ERROR,"(%P|%t) %p\n","close"),-1);
			usleep(1000000);
			matrix_size_[2] = num_slices;
		} 


		//Send  to socket 
		if(process_called_times_ == 1 && sendInit) {	
			//prep_client(host,port);
			prep_client(host,port_init);
			connect_to_server();
			bool catheterPlot = false;
			send_to_server_init(matrix_size_, field_of_view_, dataset3D, catheterPlot, planar_ST, data_buffer_size_);
			if (client_stream_.close () == -1)
				ACE_ERROR_RETURN ((LM_ERROR,"(%P|%t) %p\n","close"),-1);
			usleep(1000000); 
		}
		
		
		prep_client(host,port_image);
		connect_to_server();
		if(!dataset3D) {send_to_server_transforms(current_slice_, slice_position, quat, refreshRender);}
		//send_to_server_transforms(current_slice_, slice_position, quat, refreshRender);
		send_to_server(m2);
		if (client_stream_.close () == -1)
			ACE_ERROR_RETURN ((LM_ERROR,"(%P|%t) %p\n","close"),-1);
		//usleep(25000); 	
		
	}//end if(exportToVis)

/*
	// Write to serial port
	int serial_port = ::open("/dev/ttyUSB0",O_RDWR);
	if(serial_port < 0){
		printf("\n\n\nError %i from open: %s\n\n\n",errno,strerror(errno));
	} 
	else{
		printf("\n\n\n\n opened serial port\n\n\n");
	}
	

	struct termios tty;
	memset(&tty,0,sizeof tty);
	if(tcgetattr(serial_port, &tty) != 0){
		printf("\n\n\nError %i from tcgetattr %s\n\n\n", errno, strerror(errno));
	}
	else{
		printf("\n\n\n\n\nGot serial attributes\n\n\n\n");
	}
	//Some of these flags can probably be removed if port is set up to write only
	tty.c_cflag |= PARENB;  //use parity bit
	tty.c_cflag &= ~PARODD; //even parity
	tty.c_cflag |= CSTOPB; //two stop bits
	tty.c_cflag |= CS8; //8 bits per byte
	tty.c_cflag &= ~CRTSCTS; //Disable RTS/CTS hardware flow control
	tty.c_cflag |= CREAD | CLOCAL; //Turn on read and ignore control lines
	tty.c_lflag &= ~ICANON; //Disable canonical mode
	tty.c_lflag &= ~ECHO; //Disable echo
	tty.c_lflag &= ~ECHOE; //Disable erasure
	tty.c_lflag &= ~ECHONL; //Disable new line echo
	tty.c_lflag &= ~ISIG; //Disable interpretation of INTR, QUIT, SUSP
	tty.c_iflag &= ~(IXON | IXOFF | IXANY); //Turn off software flow control
	tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL); //Disable special handling of received bits
	tty.c_oflag &= ~OPOST; //Prevent special interpretation of output bytes
	tty.c_oflag &= ~ONLCR; //Prevent conversion of newline to carriage return/line feed
	tty.c_cc[VTIME] = 0;
	tty.c_cc[VMIN] = 0;
	cfsetispeed(&tty,B115200);
	cfsetospeed(&tty,B115200);
	if(tcsetattr(serial_port,TCSANOW,&tty) != 0){
		printf("\n\n\nError %i from tcsetattr: %s\n\n\n", errno, strerror(errno));
	}
	else{
		printf("\n\n\n\n\nSet serial attributes\n\n\n\n");
	}

	std::stringstream frame;
	frame << std::internal << std::setfill('0') << std::setw(3) << repetition_;
	std::stringstream slice;
	slice << std::internal << std::setfill('0') << std::setw(2) << current_slice_;
	std::stringstream msg_stream;
	msg_stream << 'c' << 'f' << frame.str() << 's' << slice.str() <<'.';
	const std::string tmp = msg_stream.str(); 
	const char* msg = tmp.c_str();
	//int numBytesSent = write(serial_port,msg,sizeof(msg));
	int numBytesSent = write(serial_port,msg,9);
    if (numBytesSent < 0) {
	printf("\n\n\n\nError from write: %d, %s\n\n\n\n", errno, strerror(errno));
    }
    else{
    	printf("\n\n\n\nSent %d bytes\n\n\n\n",numBytesSent);
    }


	close(serial_port);
*/
	
	//Return to Gadgetron framework controller
        if (!this->controller_)
        {
            GERROR("Cannot return result to controller, no controller set");
            return -1;
        }

        GadgetContainerMessage<GadgetMessageIdentifier>* mb = new GadgetContainerMessage<GadgetMessageIdentifier>();
        mb->getObjectPtr()->id = GADGET_MESSAGE_ISMRMRD_IMAGE;
        mb->cont(m1);
        int ret = this->controller_->output_ready(mb);

        if ((ret < 0))
        {
            GERROR("Failed to return massage to controller\n");
            return GADGET_FAIL;
        }

	if (perform_timing.value()) {gt_timer_.stop();}
        return GADGET_OK;
    } //end process

//Functions for exporting image
  int ImageFinishExportHoloLensGadget::prep_client(char *hostname, int port)
  {
	remote_addr_ = *(new ACE_INET_Addr(port, hostname));
	return 0;

  }//end prep_client


  int ImageFinishExportHoloLensGadget::connect_to_server()
  {
	//ACE_DEBUG ((LM_DEBUG, "(%P|%t) Starting connect to %s:%d\n",remote_addr_.get_host_name(),remote_addr_.get_port_number()));
	if (connector_.connect (client_stream_, remote_addr_) == -1)
		ACE_ERROR_RETURN ((LM_ERROR,"(%P|%t) %p\n","connection failed"),-1);
	else
		//ACE_DEBUG ((LM_DEBUG,"(%P|%t) connected to %s\n",remote_addr_.get_host_name ()));
	return 0;
  } //end connect_to_server


int ImageFinishExportHoloLensGadget::send_to_server(GadgetContainerMessage< hoNDArray<unsigned short> >* data)
{

	ssize_t send_cnt = 0;
	size_t bytes_transferred;
	if ((send_cnt = client_stream_.send_n(data->getObjectPtr()->get_data_ptr(), sizeof(unsigned short)*data->getObjectPtr()->get_number_of_elements(),0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send image data\n");
		return -1;
	}

	else
	{
		//ACE_DEBUG((LM_DEBUG,"Image data: Send_cnt value: %d, bytes_transferred value: %d\n",send_cnt, bytes_transferred));
	}

return 0;
} //end send_to_server



int ImageFinishExportHoloLensGadget::send_to_server_transforms(unsigned int current_slice_, float position[], float quat[], bool refreshRender)
{

	float deliminator = 9000.0;
	ssize_t send_cnt = 0;
	size_t bytes_transferred;
    
	if ((send_cnt = client_stream_.send_n(&current_slice_, sizeof(unsigned int)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send current_slice_ info\n");
		return -1;
	}
			else
	{
		//GDEBUG("\n\n\nSent current slice info: %d, bytes_transferred value: %d\n\n\n\n",current_slice_, bytes_transferred);
	}

	if ((send_cnt = client_stream_.send_n(&deliminator, sizeof(float)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send deliminator\n");
		return -1;
	}
			else
	{
		//GDEBUG("\n\n\ndeliminator bytes_transferred value: %d\n\n\n\n",bytes_transferred);
	}

	if ((send_cnt = client_stream_.send_n(&position[0], sizeof(float)*3,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send slice position info\n");
		return -1;
	}
			else
	{
		//GDEBUG("\n\n\nslice position bytes_transferred value: %d\n\n\n\n",bytes_transferred);
	}	

	if ((send_cnt = client_stream_.send_n(&deliminator, sizeof(float)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send deliminator\n");
		return -1;
	}
			else
	{
		//GDEBUG("\n\n\ndeliminator bytes_transferred value: %d\n\n\n\n",bytes_transferred);
	}

	if ((send_cnt = client_stream_.send_n(&quat[0], sizeof(float)*4,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send quaternion info\n");
		return -1;
	}
			else
	{
		//GDEBUG("\n\n\nquaternion bytes_transferred value: %d\n\n\n\n",bytes_transferred);
	}

	if ((send_cnt = client_stream_.send_n(&deliminator, sizeof(float)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send deliminator\n");
		return -1;
	}
			else
	{
		//GDEBUG("\n\n\ndeliminator bytes_transferred value: %d\n\n\n\n",bytes_transferred);
	}

	if ((send_cnt = client_stream_.send_n(&refreshRender, sizeof(bool)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send refreshRender bool flag info\n");
		return -1;
	}
			else
	{
		//GDEBUG("\n\n\nrefreshRender bool bytes_transferred value: %d\n\n\n\n",bytes_transferred);
	}

	if ((send_cnt = client_stream_.send_n(&deliminator, sizeof(float)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send deliminator\n");
		return -1;
	}
			else
	{
		//GDEBUG("\n\n\ndeliminator bytes_transferred value: %d\n\n\n\n",bytes_transferred);
	}

	return 0;
} //end send_to_server_transforms


	int ImageFinishExportHoloLensGadget::send_to_server_init(unsigned int matrix_size_[], float field_of_view_[], bool dataset3D, bool catheterPlot, float planar_ST, unsigned int data_buffer_size_){

	float deliminator_data = 9000.0;
	float deliminator_init_start = 9001.0;
	float deliminator_init_end = 9002.0;
	ssize_t send_cnt = 0;
	size_t bytes_transferred;
	
/*
	if ((send_cnt = client_stream_.send_n(&deliminator_init_start, sizeof(float)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send start deliminator\n");
		return -1;
	}
		else
	{
		GDEBUG("\n\n\nstart deliminator bytes_transferred value: %d\n\n\n\n",bytes_transferred);
	}
*/
	
	if ((send_cnt = client_stream_.send_n(&matrix_size_[0], sizeof(unsigned int)*3,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send matrix size info\n");
		return -1;
	}
		else
	{
		//GDEBUG("\n\n\nmatrix size bytes_transferred value: %d\n",bytes_transferred);
	}
/*	
	if ((send_cnt = client_stream_.send_n(&deliminator_data, sizeof(float)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send deliminator\n");
		return -1;
	}
		else
	{
		//GDEBUG("\n\n\ndelim bytes_transferred value: %d\n\n\n\n",bytes_transferred);
	}
*/
	
	if ((send_cnt = client_stream_.send_n(&field_of_view_[0], sizeof(float)*3,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send field of view info\n");
		return -1;
	}
			else
	{
		//GDEBUG("fov bytes_transferred value: %d\n",bytes_transferred);
	}

/*	
	if ((send_cnt = client_stream_.send_n(&deliminator_data, sizeof(float)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send deliminator\n");
		return -1;
	}
		else
	{
		GDEBUG("\n\n\ndelim bytes_transferred value: %d\n\n\n\n",bytes_transferred);
	}
*/
	
	if ((send_cnt = client_stream_.send_n(&dataset3D, sizeof(bool)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send dataset3D info\n");
		return -1;
	}
		else
	{
		//GDEBUG("dataset3D bytes_transferred value: %d\n",bytes_transferred);
	}
	
	
/*	
	if ((send_cnt = client_stream_.send_n(&catheterPlot, sizeof(bool)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send dataset3D info\n");
		return -1;
	}
		else
	{
		//GDEBUG("dataset3D bytes_transferred value: %d\n",bytes_transferred);
	}
	*/
	
/*	
	if ((send_cnt = client_stream_.send_n(&deliminator_data, sizeof(float)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send deliminator\n");
		return -1;
	}
		else
	{
		GDEBUG("\n\n\ndelim bytes_transferred value: %d\n\n\n\n",bytes_transferred);
	}
*/
	
	if ((send_cnt = client_stream_.send_n(&planar_ST, sizeof(float)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send planar slice thickness info\n");
		return -1;
	}
		else
	{
		//GDEBUG("planar_ST bytes_transferred value: %d\n",bytes_transferred);
	}

/*	
	if ((send_cnt = client_stream_.send_n(&deliminator_data, sizeof(float)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send deliminator\n");
		return -1;
	}
		else
	{
		GDEBUG("\n\n\ndelim bytes_transferred value: %d\n\n\n\n",bytes_transferred);
	}
*/	
	
	if ((send_cnt = client_stream_.send_n(&data_buffer_size_, sizeof(unsigned int)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send data buffer size info\n");
		return -1;
	}
		else
	{
		//GDEBUG("data buffer size bytes_transferred value: %d\n",bytes_transferred);
	}
	
	if ((send_cnt = client_stream_.send_n(&deliminator_init_end, sizeof(float)*1,0,&bytes_transferred)) <= 0)
	{
		GERROR("Unable to send end deliminator\n");
		return -1;
	}
		else
	{
		//GDEBUG("end delim bytes_transferred value: %d\n",bytes_transferred);
	}

	return 0;
} //end send_to_server_transforms

    GADGET_FACTORY_DECLARE(ImageFinishExportHoloLensGadget);
}
