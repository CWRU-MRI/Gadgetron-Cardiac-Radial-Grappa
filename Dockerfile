FROM gadgetronimages.azurecr.io/gadgetron:3.14.1-add-klt-setter

COPY Radial_grappa_HoloLens /opt/case/Radial_grappa_HoloLens
RUN cd /opt/case/Radial_grappa_HoloLens && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make install 

COPY Basic_radial_gridding /opt/case/Basic_radial_gridding
RUN cd /opt/case/Basic_radial_gridding && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make install 

COPY HoloLens_send_images /opt/case/HoloLens_send_images
RUN cd /opt/case/HoloLens_send_images && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make install 
RUN /sbin/ldconfig -v
