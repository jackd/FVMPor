#ifndef TIMER_CUDA_H
#define TIMER_CUDA_H

//#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

namespace util{

    // timing function
    class TimerCuda{
        public:
            TimerCuda() : recording_(false) {};

            // destructor
            // required for the case where a timer falls out of scope
            // while recording
            ~TimerCuda(){
                if(recording_){
                    cudaEventDestroy(start_);
                    cudaEventDestroy(stop_);
                }
            }

            // start the timer
            inline void tic(){
                if(recording_){
                    cudaEventDestroy(start_);
                    cudaEventDestroy(stop_);
                }
                recording_ = true;
                cudaEventCreate(&start_);
                cudaEventCreate(&stop_);
                cudaEventRecord(start_,0);
            };

            // stop the timer and return time
            inline float toc(){
                assert(recording_);
                recording_ = false;
                cudaEventRecord(stop_,0);
                cudaEventSynchronize(stop_);
                cudaEventElapsedTime(&elapsed_time, start_, stop_);
                cudaEventDestroy(start_);
                cudaEventDestroy(stop_);
                // convert from milli seconds to seconds
                return 1.e-3*elapsed_time;
            };

            bool is_recording(){
                return recording_;
            }

        private:
            cudaEvent_t start_, stop_;
            float elapsed_time;
            bool recording_;
    };
}
#endif

