#include "boundary_detection.h"

int main( int argc, char* argv[] ) {
    if (argc > 1) {
        // std::unique_ptr<Boundary_detection> detection(new Boundary_detection(argv[1], 0, 16.0, 0.5));
        std::unique_ptr<Boundary_detection> detection(new Boundary_detection(argv[1], 0, 16.0, 1.125));
        // std::unique_ptr<Boundary_detection> detection(new Boundary_detection(argv[1], 0, 0.0, 0.5)); // for iljoo pcap
        vector<bool> result = detection->run_detection(true);    
    }
    else {
        std::unique_ptr<Boundary_detection> detection(new Boundary_detection("velodynes/", 0, 16.0, 1.125));
        vector<bool> result = detection->run_detection(true);    
    }    

    /* Karun's        
    if (argc > 1) {
        std::unique_ptr<Boundary_detection> detection(new Boundary_detection("test2/", 3, 17, 0.5));
        vector<bool> result = detection->run_detection(true);    
    }
    else {
        std::unique_ptr<Boundary_detection> detection(new Boundary_detection("kesselRun.pcap", 0, 16.0, 0.5))t ;
        vector<bool> result = detection->run_detection(true);    
    }
    */
    return 0;
}
