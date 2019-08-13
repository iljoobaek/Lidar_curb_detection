#include<vector>

namespace fusion {
    struct Line {
        float b, m, r2;
    };
    class FusionController {
        private:
            std::vector<Line> boundaryLines;
        public:
            FusionController(){}
            
            
    }
}