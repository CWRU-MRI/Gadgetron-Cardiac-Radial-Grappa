#pragma once

#ifndef RadialGrappaCombinedRECONSTRUCTION_GPUFXNS_H
#define RadialGrappaCombinedRECONSTRUCTION_GPUFXNS_H

#include "cuNDArray.h"
#include "gpucore_export.h"
#include "mri_core_data.h"

namespace Gadgetron{
  
int perform_ksp_recon_gpu(hoNDArray<float_complext>* host_data, hoNDArray<std::complex<float> >* host_recon, cuNDArray<complext<float> >* dev_weights_ptr,size_t e, int accel_factor_E1, size_t current_slice_);

int apply_grappa_weights(cuNDArray<complext<float> >* recon_data_, cuNDArray<complext<float> >* recon_res_, cuNDArray<complext<float> >* weights_, size_t pts_to_recon_);

template <class T>__global__ void perform_dot_product(const T* __restrict__ recon_data_, T* __restrict__ recon_res_, const T* __restrict__ weights_, size_t pts_to_recon_, size_t RO,size_t us_E1,size_t fs_E1,size_t us_E2,size_t fs_E2,size_t srcCHA,size_t dstCHA, size_t wpp, size_t af_E1_m1, size_t current_slice_);

}
#endif 
