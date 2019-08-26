#include <vector>
#include <math.h>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

namespace fusion {

class FusionController {
public:
    FusionController() {}

    struct Line {
        float b, m, r2, lowerBound, upperBound, avgX;

        bool operator==(const Line& a) const //Fuzzy
        {
            return isInPercentTolerance(avgX, a.avgX, 15) && isInPercentTolerance(b, a.b, 15);
        }
    };

    struct RadarBoundary {
        float x, y;
        int lowerBound, upperBound;

        bool operator==(const RadarBoundary& a) const //Fuzzy
        {
            return isInPercentTolerance(x, a.x, 5) && isInPercentTolerance(y, a.y, 5);
        }
        bool operator==(const Line& a) const //Fuzzy
        {
            return isInPercentTolerance(x, a.avgX, 15);
        }
    };

    static bool isInPercentTolerance(float measured, float actual,
        float percentTolerance)
    {
        return (fabs((measured - actual) / actual) * 100. <= percentTolerance);
    }

    bool isInRange(float measured, float mode, float range)
    {
        return measured <= mode + range && measured >= mode - range;
    }

    void addRadarData(std::vector<cv::Vec3f>& points)
    {
        RadarBoundary boundary = { points.back()[0], points.back()[1], (int)points.front()[2], (int)points.back()[2] };
        rightRadarBoundary.push_back(boundary);
        if (rightRadarBoundary.size() > 10) {
            rightRadarBoundary.pop_front();
        }
    }

    void addLidarData(Line leftLine, Line rightLine)
    {
        leftLidarLines.push_back(leftLine);
        rightLidarLines.push_back(rightLine);
        if (leftLidarLines.size() > 25) {
            leftLidarLines.pop_front();
            rightLidarLines.pop_front();
        }
    }

    std::vector<cv::Vec3f> drawRadarBoundary(RadarBoundary boundary)
    {
        std::vector<cv::Vec3f> points;
        for (int i = boundary.lowerBound; i <= boundary.upperBound; i++) {
            points.push_back(cv::Vec3f(boundary.x, boundary.y, i));
            points.push_back(cv::Vec3f(boundary.x, boundary.y - 2, i));
            points.push_back(cv::Vec3f(boundary.x, boundary.y - 1, i));
            points.push_back(cv::Vec3f(boundary.x, boundary.y + 1, i));
            points.push_back(cv::Vec3f(boundary.x, boundary.y + 2, i));
        }
        return points;
    }

    void updatePrevRadarBoundaries(std::vector<float> velocity, float time = 0.1)
    {
        float vx = velocity[0], vy = velocity[1], vz = velocity[2];
        float deltaPitch = velocity[3], deltaRoll = velocity[4], deltaYaw = velocity[5];
        for (int i = 0; i < rightRadarBoundary.size(); i++) {
            rotateEulerAngles(rightRadarBoundary[i], deltaYaw, deltaPitch, deltaRoll);
            translatePoint(rightRadarBoundary[i], vx, vy, vz);
        }
    }

    std::vector<cv::viz::WLine> generateDisplayLine(std::vector<cv::Vec3f>& lidarPoints, std::vector<cv::Vec3f>& radarPoints)
    {
        if (rightLidarLines.size() < 25) {
            return generateLidarLine(lidarPoints);
        }
        return generateLidarLine(lidarPoints);
    }

    std::vector<float> rotatedPointParameters(std::vector<cv::Vec3f>& points)
    {
        std::vector<float> left, right, thetaVecL, thetaVecR;
        std::vector<cv::Vec3f> tempL, tempR;
        std::sort(points.begin(), points.end(), [](const cv::Vec3f& a, const cv::Vec3f& b) {
            return (a[1] < b[1]);
        });
        cv::Vec3f newOriginL = cv::Vec3f(0, 0, 0), newOriginR = cv::Vec3f(0, 0, 0);
        for (int i = 0; i < points.size(); i++) {
            if (newOriginL != cv::Vec3f(0, 0, 0) && newOriginR != cv::Vec3f(0, 0, 0)) {
                break;
            }
            if (points[i][0] > 0 && newOriginR == cv::Vec3f(0, 0, 0)) {
                newOriginR = points[i];
            }
            if (points[i][0] < 0 && newOriginL == cv::Vec3f(0, 0, 0)) {
                newOriginL = points[i];
            }
        }
        for (int i = 0; i < points.size(); i++) {
            if (points[i][0] > 0) {
                tempR.push_back(cv::Vec3f(points[i][0] - newOriginR[0], points[i][1] - newOriginR[1], points[i][2]));
            }
            if (points[i][0] < 0) {
                tempL.push_back(cv::Vec3f(points[i][0] - newOriginL[0], points[i][1] - newOriginL[1], points[i][2]));
            }
        }
        for (int i = 0; i < tempR.size(); i++) {
            thetaVecR.push_back((int)(10 * atan(tempR[i][0] / tempR[i][1]) + 0.5) / 10.);
        }
        for (int i = 0; i < tempL.size(); i++) {
            thetaVecL.push_back((int)(10 * atan(tempL[i][0] / tempL[i][1]) + 0.5) / 10.);
        }
        float thetaLmode = getMode(thetaVecL);
        float thetaRmode = getMode(thetaVecR);
        for (int i = 0; i < tempR.size(); i++) {
            right.push_back((int)(10 * yaw(tempR[i], thetaRmode)[0] + 0.5) / 10.);
        }
        for (int i = 0; i < tempL.size(); i++) {
            left.push_back((int)(10 * yaw(tempL[i], thetaLmode)[0] + 0.5) / 10.);
        }
        float modeL = getMode(left);
        float modeR = getMode(right);
        std::vector<float> values = { thetaLmode, thetaRmode, modeL, modeR };
        return values;
    }

    std::vector<cv::viz::WLine> generateLidarLine(std::vector<cv::Vec3f>& lidarPoints)
    {
        std::vector<std::vector<cv::Vec3f> > lines = findLidarLine(lidarPoints);
        std::vector<cv::viz::WLine> WLines;
        cv::Point3d lower, upper;
        Line leftLine = linearlsq(lines[0]);
        Line rightLine = linearlsq(lines[1]);
        addLidarData(leftLine, rightLine);
        if (leftLidarLines.size() == 0) {
            lower = cv::Point3d(leftLine.b + leftLine.lowerBound * leftLine.m, leftLine.lowerBound, 0);
            upper = cv::Point3d(leftLine.b + leftLine.upperBound * leftLine.m, leftLine.upperBound, 0);
            cv::viz::WLine leftWLine = cv::viz::WLine(lower, upper, cv::viz::Color::green());
            lower = cv::Point3d(rightLine.b + rightLine.lowerBound * rightLine.m, rightLine.lowerBound, 0);
            upper = cv::Point3d(rightLine.b + rightLine.upperBound * rightLine.m, rightLine.upperBound, 0);
            cv::viz::WLine rightWLine = cv::viz::WLine(lower, upper, cv::viz::Color::green());
            WLines.push_back(leftWLine);
            WLines.push_back(rightWLine);
        }
        else {
            if (confidenceLeft > 0.4) {
                lower = cv::Point3d(leftLine.b + leftLine.lowerBound * leftLine.m, leftLine.lowerBound, 0);
                upper = cv::Point3d(leftLine.b + leftLine.upperBound * leftLine.m, leftLine.upperBound, 0);
                cv::viz::WLine leftWLine = cv::viz::WLine(lower, upper, cv::viz::Color::green());
                WLines.push_back(leftWLine);
            }
            else {
                cv::viz::WLine leftWLine = cv::viz::WLine(cv::Point3d(0, 0, 0), cv::Point3d(0, 0, 0), cv::viz::Color::green());
                WLines.push_back(leftWLine);
            }
            if (confidenceRight > 0.4) {
                lower = cv::Point3d(rightLine.b + rightLine.lowerBound * rightLine.m, rightLine.lowerBound, 0);
                upper = cv::Point3d(rightLine.b + rightLine.upperBound * rightLine.m, rightLine.upperBound, 0);
                cv::viz::WLine rightWLine = cv::viz::WLine(lower, upper, cv::viz::Color::green());
                WLines.push_back(rightWLine);
            }
            else {
                cv::viz::WLine rightWLine = cv::viz::WLine(cv::Point3d(0, 0, 0), cv::Point3d(0, 0, 0), cv::viz::Color::green());
                WLines.push_back(rightWLine);
            }
        }
        return WLines;
    }

    std::vector<cv::viz::WPolyLine> displayThirdOrder(std::vector<cv::Vec3f>& lidarPoints) {
        std::vector<std::vector<cv::Vec3f> > lines = findLidarLine(lidarPoints);
        cv::viz::WPolyLine left = thirdOrderlsq(lines[0]), right = thirdOrderlsq(lines[1]);
        std::vector<cv::viz::WPolyLine> displayLines = { left, right };            
        return displayLines;
    }

    cv::viz::WPolyLine thirdOrderlsq(std::vector<cv::Vec3f>& points)
    {
        std::vector<float> x, y;
        for (int i = 0; i < points.size(); i++) {
            //It is reversed, the xDistance is y in the y = mx+b, and yDistance is x
            y.push_back(points[i][0]);
            x.push_back(points[i][1]);
        }
        if (x.size() == 0) {
            cv::Mat pointsMat = cv::Mat(static_cast<int>(points.size()), 1, CV_32FC3, &points[0]);
            return cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
        }
          size_t N = x.size();
        int n = 3;
        int np1 = n + 1;
        int np2 = n + 2;
        int tnp1 = 2 * n + 1;
        float tmp;

        // X = vector that stores values of sigma(xi^2n)
        std::vector<float> X(tnp1);
        for (int i = 0; i < tnp1; ++i) {
            X[i] = 0;
            for (int j = 0; j < N; ++j)
            X[i] += (float)pow(x.at(j), i);
        }

        // a = vector to store final coefficients.
        std::vector<float> a(np1);

        // B = normal augmented matrix that stores the equations.
        std::vector<std::vector<float> > B(np1, std::vector<float> (np2, 0));

        for (int i = 0; i <= n; ++i) 
            for (int j = 0; j <= n; ++j) 
            B[i][j] = X[i + j];

        // Y = vector to store values of sigma(xi^n * yi)
        std::vector<float> Y(np1);
        for (int i = 0; i < np1; ++i) {
            Y[i] = (float)0;
            for (int j = 0; j < N; ++j) {
            Y[i] += (float)pow(x[j], i)*y[j];
            }
        }

        // Load values of Y as last column of B
        for (int i = 0; i <= n; ++i) 
            B[i][np1] = Y[i];

        n += 1;
        int nm1 = n-1;

        // Pivotisation of the B matrix.
        for (int i = 0; i < n; ++i) 
            for (int k = i+1; k < n; ++k) 
            if (B[i][i] < B[k][i]) 
                for (int j = 0; j <= n; ++j) {
                tmp = B[i][j];
                B[i][j] = B[k][j];
                B[k][j] = tmp;
                }

        // Performs the Gaussian elimination.
        // (1) Make all elements below the pivot equals to zero
        //     or eliminate the variable.
        for (int i=0; i<nm1; ++i)
            for (int k =i+1; k<n; ++k) {
            float t = B[k][i] / B[i][i];
            for (int j=0; j<=n; ++j)
                B[k][j] -= t*B[i][j];         // (1)
            }

        // Back substitution.
        // (1) Set the variable as the rhs of last equation
        // (2) Subtract all lhs values except the target coefficient.
        // (3) Divide rhs by coefficient of variable being calculated.
        for (int i=nm1; i >= 0; --i) {
            a[i] = B[i][n];                   // (1)
            for (int j = 0; j<n; ++j)
            if (j != i)
                a[i] -= B[i][j] * a[j];       // (2)
            a[i] /= B[i][i];                  // (3)
        }
        std::vector<float> coeffs;
        for (size_t i = 0; i < a.size(); ++i) 
            coeffs.push_back(a[i]);
        std::vector<cv::Vec3f> linePoints;
        for (int i = *std::min_element(y.begin(), y.end()) * 100; i <= *std::max_element(y.begin(), y.end()) * 200; i++) {
            linePoints.push_back(cv::Vec3f(coeffs[0] + coeffs[1] * i/100. + coeffs[2] * powf(i/100., 2) + coeffs[3] * powf(i/100., 3), i/100., 0));
        }
        cv::Mat pointsMat = cv::Mat(static_cast<int>(linePoints.size()), 1, CV_32FC3, &linePoints[0]);
        return cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
    }

    Line linearlsq(std::vector<cv::Vec3f>& points)
    {
        std::vector<float> xVec, yVec;
        for (int i = 0; i < points.size(); i++) {
            //It is reversed, the xDistance is y in the y = mx+b, and yDistance is x
            yVec.push_back(points[i][0]);
            xVec.push_back(points[i][1]);
        }
        if (xVec.size() == 0) {
            return Line{ 0, 0, 0, 0, 0, 0 };
        }
        float xMean = std::accumulate(std::begin(xVec), std::end(xVec), 0.0) / xVec.size();
        float yMean = std::accumulate(std::begin(yVec), std::end(yVec), 0.0) / yVec.size();
        float tss = 0.;
        float rss = 0.;
        float top = 0.;
        float bot = 0.;
        for (int i = 0; i < xVec.size(); i++) {
            top += (xVec[i] - xMean) * (yVec[i] - yMean);
            bot += (xVec[i] - xMean) * (xVec[i] - xMean);
        }
        float m = top / bot;
        float b = yMean - m * xMean;
        for (int i = 0; i < xVec.size(); i++) {
            tss += powf((yVec[i] - yMean), 2);
            rss += powf(yVec[i] - (m * xVec[i] + b), 2);
        }
        float r2 = 1. - rss / tss;
        float xMin = *std::min_element(xVec.begin(), xVec.end()), xMax = *std::max_element(xVec.begin(), xVec.end());
        if (xMax - xMin < 1.5) {
            xMax += 1;
        }
        Line lidarLine = { b, m, r2, xMin, xMax, b + m * (xMin + xMax) / 2 };
        if (fabsf(xMax - xMin) < fabsf((m * xMax + b) - (m * xMin + b))) {
            lidarLine = { 0, 0, 0, 0, 0, 0 };
        }
        return lidarLine;
    }

    std::vector<cv::viz::WText3D> displayConfidence(std::vector<cv::Vec3f>& points)
    {
        zeroConfidence(points);
        cv::viz::WText3D left = cv::viz::WText3D("Left Confidence: " + std::to_string(confidenceLeft), cv::Point3d(-3, -1, 0), 0.1);
        cv::viz::WText3D right = cv::viz::WText3D("Right Confidence: " + std::to_string(confidenceRight), cv::Point3d(1, -1, 0), 0.1);
        std::vector<cv::viz::WText3D> out = { left, right };
        return out;
    }

    std::vector<std::vector<cv::Vec3f> > findLidarLine(std::vector<cv::Vec3f>& points)
    {
        cv::Vec3f newOriginL = cv::Vec3f(0, 0, 0), newOriginR = cv::Vec3f(0, 0, 0);
        for (int i = 0; i < points.size(); i++) {
            if (newOriginL != cv::Vec3f(0, 0, 0) && newOriginR != cv::Vec3f(0, 0, 0)) {
                break;
            }
            if (points[i][0] > 0 && newOriginR == cv::Vec3f(0, 0, 0)) {
                newOriginR = points[i];
            }
            if (points[i][0] < 0 && newOriginL == cv::Vec3f(0, 0, 0)) {
                newOriginL = points[i];
            }
        }
        std::vector<float> rotationValues = rotatedPointParameters(points);
        float thetaL = rotationValues[0], thetaR = rotationValues[1], modeL = rotationValues[2], modeR = rotationValues[3];
        std::vector<cv::Vec3f> linePointsR, linePointsL;
        std::vector<float> leftX, rightX;
        int leftCount = 0, rightCount = 0;
        for (int i = 0; i < points.size(); i++) {
            if (points[i][0] > 0) {
                rightCount++;
                if (isInRange(yaw(cv::Vec3f(points[i][0] - newOriginR[0], points[i][1] - newOriginR[1], points[1][2]), thetaR)[0], modeR, .4) && points[i][1] < 10) {
                    linePointsR.push_back(points[i]);
                    rightX.push_back(points[i][0]);
                }
            }
            if (points[i][0] < 0) {
                leftCount++;
                if (isInRange(yaw(cv::Vec3f(points[i][0] - newOriginL[0], points[i][1] - newOriginL[1], points[1][2]), thetaL)[0], modeL, .4) && points[i][1] < 10 && points[i][0] < 0) {
                    linePointsL.push_back(points[i]);
                    leftX.push_back(points[i][0]);
                }
            }
        }
        if (linePointsL.size() && leftCount) {
            confidenceLeft = linePointsL.size() / static_cast<float>(leftCount);
        }
        else {
            confidenceLeft = 0.0;
        }
        if (linePointsR.size() && rightCount) {
            confidenceRight = linePointsR.size() / static_cast<float>(rightCount);
        }
        else {
            confidenceRight = 0.0;
        }
        float rRange, lRange;
        if (linePointsR.size()) {
            if (*std::max_element(rightX.begin(), rightX.end()) - *std::min_element(rightX.begin(), rightX.end()) > 4) {
                confidenceRight = 0.;
            }
        }
        if (linePointsL.size()) {
            if (*std::max_element(leftX.begin(), leftX.end()) - *std::min_element(leftX.begin(), leftX.end()) > 4) {
                confidenceLeft = 0.;
            }
        }
        std::vector<std::vector<cv::Vec3f> > linePoints;
        linePoints.push_back(linePointsL);
        linePoints.push_back(linePointsR);
        return linePoints;
    }

    float lineCoherency(std::deque<Line> lines)
    {
        float matches = 0;
        for (int i = 1; i < lines.size() - 1; i++) {
            if (lines[i] == lines[i - 1]) {
                matches++;
            }
        }
        return matches / lines.size();
    }

    void zeroConfidence(std::vector<cv::Vec3f>& points)
    {
        std::vector<float> xVecL, xVecR;
        for (int i = 0; i < points.size(); i++) {
            if (points[i][0] > 0 && points[i][1] < 15) {
                xVecR.push_back((int)(10 * points[i][0] + 0.5) / 10.);
            }
            if (points[i][0] < 0 && points[i][1] < 15) {
                xVecL.push_back((int)(10 * points[i][0] + 0.5) / 10.);
            }
        }
        float leftCoherency = lineCoherency(leftLidarLines), rightCoherency = lineCoherency(rightLidarLines);
        if (leftCoherency > 0.8) {
            confidenceLeft = 1.;
        }
        if (rightCoherency > 0.8) {
            confidenceRight = 1.;
        }
        /* if (leftCoherency < 0.3) {
            confidenceLeft = 0.;
        }
        if (rightCoherency < 0.3) {
            confidenceRight = 0.;
        } */
    }

    RadarBoundary rotateEulerAngles(RadarBoundary& boundary, float a, float b, float c)
    {
        float x = (cosf(a) * cosf(b) * boundary.x) + (boundary.y) * (cosf(a) * sinf(b) * sinf(c) - sinf(a) * cosf(c)) + (boundary.lowerBound) * (cosf(a) * sinf(b) * cosf(c) + sinf(a) * sinf(c));
        float y = (sinf(a) * cosf(b) * boundary.x) + (boundary.y) * (sinf(a) * sinf(b) * sinf(c) + cosf(a) * cosf(c)) + (boundary.lowerBound) * (sinf(a) * sinf(b) * cosf(c) - cosf(a) * sinf(c));
        //float z = (-sinf(b) * boundary.x) + (boundary.y)*(cosf(b)*sinf(c)) + (boundary.lowerBound)*(cosf(b)*cosf(c));
        boundary.x = x;
        boundary.y = y;
        return boundary;
    }

    RadarBoundary translatePoint(RadarBoundary& boundary, float vx, float vy, float vz, float time = 0.1)
    {
        //All the velocities are negated because if we keep the vehicle constant, as it travels forward the point has a velocity backwards
        boundary.x += -vx * time;
        boundary.y += -vy * time;
        boundary.upperBound += -vz * time;
        boundary.lowerBound += -vz * time;
        return boundary;
    }

    cv::Vec3f rotateEulerAngles(cv::Vec3f point, float a, float b, float c)
    {
        float x = (cosf(a) * cosf(b) * point[0]) + (point[1]) * (cosf(a) * sinf(b) * sinf(c) - sinf(a) * cosf(c)) + (point[2]) * (cosf(a) * sinf(b) * cosf(c) + sinf(a) * sinf(c));
        float y = (sinf(a) * cosf(b) * point[0]) + (point[1]) * (sinf(a) * sinf(b) * sinf(c) + cosf(a) * cosf(c)) + (point[2]) * (sinf(a) * sinf(b) * cosf(c) - cosf(a) * sinf(c));
        float z = (-sinf(b) * point[0]) + (point[1]) * (cosf(b) * sinf(c)) + (point[2]) * (cosf(b) * cosf(c));
        return cv::Vec3f(x, y, z);
    }

    cv::Vec3f yaw(cv::Vec3f point, float a)
    {
        float x = cosf(a) * point[0] + -sinf(a) * point[1];
        float y = sinf(a) * point[0] + cosf(a) * point[1];
        return cv::Vec3f(x, y, point[2]);
    }

    float getMode(std::vector<float> data)
    {
        if (data.size() == 0) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        int* ipRepetition = new int[data.size()];
        for (int i = 0; i < data.size(); ++i) {
            ipRepetition[i] = 0;
            int j = 0;
            bool bFound = false;
            while ((j < i) && (data[i] != data[j])) {
                if (data[i] != data[j]) {
                    ++j;
                }
            }
            ++(ipRepetition[j]);
        }
        int iMaxRepeat = 0;
        for (int i = 1; i < data.size(); ++i) {
            if (ipRepetition[i] > ipRepetition[iMaxRepeat]) {
                iMaxRepeat = i;
            }
        }
        delete[] ipRepetition;
        return static_cast<float>(data[iMaxRepeat]);
    }

private:
    std::deque<Line> leftLidarLines;
    std::deque<Line> rightLidarLines;
    std::deque<RadarBoundary> leftRadarBoundary;
    std::deque<RadarBoundary> rightRadarBoundary;
    float confidenceLeft, confidenceRight;
};
}