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

    static bool isInPercentTolerance(float measured, float actual,
        float percentTolerance)
    {
        return (fabs((measured - actual) / actual) * 100. <= percentTolerance);
    }

    bool isInRange(float measured, float mode, float range)
    {
        return measured <= mode + range && measured >= mode - range;
    }

    std::vector<cv::viz::WLine> generateDisplayLine(std::vector<cv::Vec3f>& points)
    {
        if (leftLidarLines.size() > 10) {
            leftLidarLines.pop_front();
            rightLidarLines.pop_front();
        }
        std::vector<std::vector<cv::Vec3f> > lines = findLidarLine(points);
        std::vector<cv::viz::WLine> WLines;
        cv::Point3d lower, upper;
        Line leftLine = lidarllsq(lines[0]);
        Line rightLine = lidarllsq(lines[1]);
        leftLidarLines.push_back(leftLine);
        rightLidarLines.push_back(rightLine);
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
            int matches = 0;
            for (Line line : leftLidarLines) {
                if (leftLine == line) {
                    matches++;
                }
            }
            if (matches > 5) {
                lower = cv::Point3d(leftLine.b + leftLine.lowerBound * leftLine.m, leftLine.lowerBound, 0);
                upper = cv::Point3d(leftLine.b + leftLine.upperBound * leftLine.m, leftLine.upperBound, 0);
                cv::viz::WLine leftWLine = cv::viz::WLine(lower, upper, cv::viz::Color::green());
                WLines.push_back(leftWLine);
            }
            else {
                cv::viz::WLine leftWLine = cv::viz::WLine(cv::Point3d(0, 0, 0), cv::Point3d(0, 0, 0), cv::viz::Color::green());
                WLines.push_back(leftWLine);
            }
            matches = 0;
            for (Line line : rightLidarLines) {
                if (rightLine == line) {
                    matches++;
                }
            }
            if (matches > 5) {
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

    std::vector<std::vector<cv::Vec3f> > findLidarLine(std::vector<cv::Vec3f>& points)
    {   
        std::vector<float> xVecL, xVecR; //xDistance
        for (int i = 0; i < points.size(); i++) {
            if (points[i][0] > 0) {
                xVecR.push_back((int)(10 * points[i][0] + 0.5) / 10.);
            }
            else {
                xVecL.push_back((int)(10 * points[i][0] + 0.5) / 10.);
            }
        }
        float xRmode = getMode(xVecR);
        float xLmode = getMode(xVecL);
        std::vector<cv::Vec3f> linePointsR;
        std::vector<cv::Vec3f> linePointsL;
        for (int i = 0; i < points.size(); i++) {
            if (isInRange(points[i][0], xRmode, .25) && points[i][1] < 10 && points[i][0] > 0) {
                linePointsR.push_back(points[i]);
            }
            if (isInRange(points[i][0], xLmode, .25) && points[i][1] < 10 && points[i][0] < 0) {
                linePointsL.push_back(points[i]);
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
};
}