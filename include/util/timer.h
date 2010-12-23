#ifndef TIMER_H
#define TIMER_H

#include <mkl.h>

namespace util{

    // timing function
    class Timer{
        public:
            Timer() : recording_(false) {};

            // start the timer
            void tic(){
                recording_ = true;
                t_tic_ =  dsecnd();
            };

            // stop the timer and return time
            double toc(){
                assert(recording_);
                recording_ = false;
                return dsecnd()-t_tic_;
            };

            // return time since last tic
            // but don't stop the timer
            double toc_continue(){
                assert(recording_);
                return dsecnd()-t_tic_;
            };

            void pause(double t){
                tic();
                while(toc_continue()<t);
            };

            bool is_recording(){
                return recording_;
            }

        private:
            double t_tic_;
            bool recording_;
    };
}
#endif
