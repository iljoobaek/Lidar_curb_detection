#include <vector>
#include <math.h>

#include <opencv2/opencv.hpp>

namespace fusion {

class FusionController {
public:
    FusionController() {}

    struct Line {
        float b, m, r2, lowerBound, upperBound, avgX;

        bool operator==(const Line& a) const //Fuzzy
        {
            return (isInPercentTolerance(b, a.b, 5) && isInPercentTolerance(m, a.m, 5)) && isInPercentTolerance(avgX, a.avgX, 5) && isInPercentTolerance(r2, a.r2, 5);
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
        if (leftLidarLines.size() > 16) {
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

    bool isLidarConfident(std::deque<Line>& lines)
    {
        int matches = 0;
        for (int i = 0; i < lines.size() - 1; i++) {
            if (lines.back() == lines[i]) {
                matches++;
            }
        }
        return matches > 5;
    }

    std::vector<cv::viz::WLine> generateDisplayLine(std::vector<cv::Vec3f>& lidarPoints, std::vector<cv::Vec3f>& radarPoints)
    {
        if (rightLidarLines.size() < 15) {
            return generateLidarLine(lidarPoints);
        }
        if (isLidarConfident(rightLidarLines)) {
            if (rightRadarBoundary.back() == rightLidarLines.back()) { //Is the point Approximately on the previous line
                lidarPoints.push_back(radarPoints.front()); //Combines the Data, Radar can be Repeated or Weighted
                std::cout << "matched" << std::endl;
            }
            std::cout << "radar is not on line" << std::endl;
            return generateLidarLine(lidarPoints);
        }
        else {
            std::cout << "not confident" << std::endl;
            return generateLidarLine(lidarPoints);
        }
    }

    std::vector<cv::viz::WLine> generateLidarLine(std::vector<cv::Vec3f>& lidarPoints)
    {
        std::vector<std::vector<cv::Vec3f> > lines = findLidarLine(lidarPoints);
        std::vector<cv::viz::WLine> WLines;
        cv::Point3d lower, upper;
        Line leftLine = lidarllsq(lines[0]);
        Line rightLine = lidarllsq(lines[1]);
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
            if (isLidarConfident(leftLidarLines)) {
                lower = cv::Point3d(leftLine.b + leftLine.lowerBound * leftLine.m, leftLine.lowerBound, 0);
                upper = cv::Point3d(leftLine.b + leftLine.upperBound * leftLine.m, leftLine.upperBound, 0);
                cv::viz::WLine leftWLine = cv::viz::WLine(lower, upper, cv::viz::Color::green());
                WLines.push_back(leftWLine);
            }
            else {
                cv::viz::WLine leftWLine = cv::viz::WLine(cv::Point3d(0, 0, 0), cv::Point3d(0, 0, 0), cv::viz::Color::green());
                WLines.push_back(leftWLine);
            }
            if (isLidarConfident(rightLidarLines)) {
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

    Line lidarllsq(std::vector<cv::Vec3f>& points)
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
            return Line{ 0, 0, 0, 0, 0, 0 };
        }
        Line lidarLine = { b, m, r2, xMin, xMax, b + m * (xMin + xMax) / 2 };
        return lidarLine;
    }

    cv::Vec3f yaw(cv::Vec3f point, float a)
    {
        float x = cosf(a) * point[0] + -sinf(a) * point[1];
        float y = sinf(a) * point[0] + cosf(a) * point[1];
        return cv::Vec3f(x, y, point[2]);
    }

    std::vector<float> findThetaToRotate(std::vector<cv::Vec3f>& points)
    {
        std::vector<float> thetaVecL, thetaVecR;
        for (int i = 0; i < points.size(); i++) {
            if (points[i][0] > 0) {
                thetaVecR.push_back((int)(10 * atan(points[i][0] / points[i][1]) + 0.5) / 10.);
            }
            else {
                thetaVecL.push_back((int)(10 * atan(points[i][0] / -points[i][1]) + 0.5) / 10.);
            }
        }
        float thetaRmode = getMode(thetaVecR);
        float thetaLmode = getMode(thetaVecL);
        std::vector<float> thetas;
        thetas.push_back(thetaLmode);
        thetas.push_back(thetaRmode);
        return thetas;
    }

    std::vector<std::vector<cv::Vec3f> > findLidarLine(std::vector<cv::Vec3f>& points)
    {
        std::vector<float> xVecL, xVecR, thetas; //xDistance
        thetas = findThetaToRotate(points);
        //float thetaL = CV_PI/4. - thetas[0], thetaR = CV_PI/4. - thetas[1];
        float thetaL = 0, thetaR = 0;
        for (int i = 0; i < points.size(); i++) {
            if (yaw(points[i], thetaR)[0] > 0) {
                xVecR.push_back((int)(10 * points[i][0] + 0.5) / 10.);
            }
            if (yaw(points[i], thetaL)[0] < 0) {
                xVecL.push_back((int)(10 * points[i][0] + 0.5) / 10.);
            }
        }
        float xRmode = getMode(xVecR);
        float xLmode = getMode(xVecL);
        std::vector<cv::Vec3f> linePointsR;
        std::vector<cv::Vec3f> linePointsL;
        for (int i = 0; i < points.size(); i++) {
            if (linePointsR.size() && linePointsL.size()) {
                if (isInRange(yaw(points[i], thetaR)[0], linePointsR.back()[0], .1) && points[i][1] < 10 && yaw(points[i], thetaR)[0] > 0) {
                    linePointsR.push_back(points[i]);
                }
                if (isInRange(yaw(points[i], thetaL)[0], linePointsL.back()[0], .1) && points[i][1] < 10 && yaw(points[i], thetaL)[0] < 0) {
                    linePointsL.push_back(points[i]);
                }
            }
            else {
                if (isInRange(yaw(points[i], thetaR)[0], xRmode, .25) && points[i][1] < 10 && yaw(points[i], thetaR)[0] > 0) {
                    linePointsR.push_back(points[i]);
                }
                if (isInRange(yaw(points[i], thetaL)[0], xLmode, .25) && points[i][1] < 10 && yaw(points[i], thetaL)[0] < 0) {
                    linePointsL.push_back(points[i]);
                }
            }
        }
        std::vector<std::vector<cv::Vec3f> > linePoints;
        linePoints.push_back(linePointsL);
        linePoints.push_back(linePointsR);
        return linePoints;
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
};
}