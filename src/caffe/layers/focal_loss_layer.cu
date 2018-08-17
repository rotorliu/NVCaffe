#include <algorithm>
#include <device_launch_parameters.h>

#include "caffe/layers/focal_loss_layer.hpp"
#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/net.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FocalLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts, float alpha_, float gamma_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      //loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
      //                Dtype(FLT_MIN)));
      Dtype pt = prob_data[n * dim + label_value * spatial_dim + s];
      loss[index] = -alpha_ * powf(1 - pt, gamma_) * log(max(pt, min_dtype<Dtype>()));
      counts[index] = 1;
    }
  }
}

template <>
__global__ void FocalLossForwardGPU<half>(const int nthreads,
	const half* prob_data, const half* label, half* loss,
	const int num, const int dim, const int spatial_dim,
	const bool has_ignore_label_, const int ignore_label_,
	half* counts, float alpha_, float gamma_) {
	const float minh = __half2float(min_dtype<half>());
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int n = index / spatial_dim;
		const int s = index % spatial_dim;
		const int label_value = static_cast<int>(__half2float(label[n * spatial_dim + s]));
		if (has_ignore_label_ && label_value == ignore_label_) {
			loss[index].setx(0U);
			counts[index].setx(0U);
		}
		else {
			float pt = __half2float(prob_data[n * dim + label_value * spatial_dim + s]);
			loss[index] = float2half_clip (-alpha_ * powf(1 - pt, gamma_) * log(max(pt, minh)));
			counts[index].setx(0x3c00U);  // set to 1
		}
	}
}

template <typename Ftype, typename Btype>
void FocalLossLayer<Ftype, Btype>::Forward_gpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Ftype* prob_data = prob_->template gpu_data<Ftype>();
  const Ftype* label = bottom[1]->gpu_data<Ftype>();
  const int dim = prob_->count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Ftype* loss_data = bottom[0]->mutable_gpu_diff<Ftype>();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Ftype* counts = prob_->template mutable_gpu_diff<Ftype>();
  cudaStream_t stream = Caffe::thread_stream();
  if (tp<Ftype>() == FLOAT16) {
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  FocalLossForwardGPU<half> << <CAFFE_GET_BLOCKS(nthreads),
		  CAFFE_CUDA_NUM_THREADS, 0, stream >> >(nthreads, reinterpret_cast<const half*>(prob_data),
			  reinterpret_cast<const half*>(label), reinterpret_cast<half*>(loss_data),
			  outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, 
			  reinterpret_cast<half*>(counts),
			  alpha_, gamma_);
  }
  else {
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  FocalLossForwardGPU << <CAFFE_GET_BLOCKS(nthreads),
		  CAFFE_CUDA_NUM_THREADS, 0, stream >> >(nthreads, prob_data, label, loss_data,
			  outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts,
			  alpha_, gamma_);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float loss;
  caffe_gpu_asum(nthreads, loss_data, &loss, 0);
  float valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count, 0);
  }
  top[0]->mutable_cpu_data<Ftype>()[0] = loss / get_normalizer(normalization_, valid_count);
  if (top.size() == 2) {
    top[1]->ShareData(*prob_);
  }
}

template <typename Dtype>
__global__ void FocalLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts, float alpha_, float gamma_) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      Dtype pt = bottom_diff[n * dim + label_value * spatial_dim + s];
      for (int c = 0; c < channels; ++c) {
        if(c == label_value){
            bottom_diff[n * dim + c * spatial_dim + s] = alpha_ *
            powf(1 - pt, gamma_) * (gamma_ * pt * log(max(pt, Dtype(FLT_MIN))) + pt - 1);
        }
        else{
            Dtype pc = bottom_diff[n * dim + c * spatial_dim + s];
            bottom_diff[n * dim + c * spatial_dim + s] = alpha_ * 
                (powf(1 - pt, gamma_ - 1) * (-gamma_ * log(max(pt, Dtype(FLT_MIN))) * pt * pc) +
                powf(1 - pt, gamma_) * pc);
        }
      }
      counts[index] = 1;
    }
  }
}

template <>
__global__ void FocalLossBackwardGPU<half>(const int nthreads, const half* top,
	const half* label, half* bottom_diff, const int num, const int dim,
	const int spatial_dim, const bool has_ignore_label_,
	const int ignore_label_, half* counts, float alpha_, float gamma_) {
	const int channels = dim / spatial_dim;
	const float minh = __half2float(min_dtype<half>());
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int n = index / spatial_dim;
		const int s = index % spatial_dim;
		const int label_value = static_cast<int>(__half2float(label[n * spatial_dim + s]));

		if (has_ignore_label_ && label_value == ignore_label_) {
			for (int c = 0; c < channels; ++c) {
				bottom_diff[n * dim + c * spatial_dim + s].setx(0U);
			}
			counts[index].setx(0U);
		}
		else {
			float pt = __half2float(bottom_diff[n * dim + label_value * spatial_dim + s]);
			for (int c = 0; c < channels; ++c) {
				if (c == label_value) {
					bottom_diff[n * dim + c * spatial_dim + s] = float2half_clip(alpha_ *
						powf(1 - pt, gamma_) * (gamma_ * pt * log(max(pt, minh)) + pt - 1));
				}
				else {
					float pc = __half2float(bottom_diff[n * dim + c * spatial_dim + s]);
					bottom_diff[n * dim + c * spatial_dim + s] = float2half_clip(alpha_ *
						(powf(1 - pt, gamma_ - 1) * (-gamma_ * log(max(pt, minh)) * pt * pc) +
							powf(1 - pt, gamma_) * pc));
				}
			}
			counts[index].setx(0x3c00U);  // 1.
		}
	}
}

template <typename Ftype, typename Btype>
void FocalLossLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    const Btype* prob_data = prob_->template gpu_data<Btype>();
    const Btype* top_data = top[0]->gpu_data<Btype>();
    caffe_gpu_memcpy(prob_->count() * sizeof(Btype), prob_data, bottom_diff);
    const Btype* label = bottom[1]->gpu_data<Btype>();
    const int dim = prob_->count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Btype* counts = prob_->template mutable_gpu_diff<Btype>();
    // NOLINT_NEXT_LINE(whitespace/operators)
    FocalLossBackwardGPU<<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream() >>>(nthreads, top_data, label, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts,
        alpha_, gamma_);
	CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
    int valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
	  float float_count;
      caffe_gpu_asum(nthreads, counts, &float_count, 0);
	  valid_count = int(float_count);
    }
    float loss_weight = float(top[0]->cpu_diff<Btype>()[0]) / get_normalizer(normalization_, valid_count);
	if (this->parent_net() != NULL) {
		loss_weight *= this->parent_net()->global_grad_scale();
	}
    caffe_gpu_scal<Btype>(prob_->count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(FocalLossLayer);

}  // namespace caffe
