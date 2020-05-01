#include "fastvirtualscan/fastvirtualscan.h"
#include "../../timers.h"
#include <omp.h>

#define MAXVIRTUALSCAN 1e6
#define USEOMP

extern Timers timers;
/*std::string optType = "openmp_cpu";
std::string optInfo = "openmp_cpu";*/

void FastVirtualScan::printsvs() {
	int size = 160;
	for(int i =0; i<beamnum*size; ++i) {
		cout << i << ": [" << i/size << "] [" << i%size << "]: ";
		cout << (svs+i)->rotid << " " << (svs+i)->rotlength << " " << 
		(svs+i)->rotheight << " " << (svs+i)->length << " " << 
		(svs+i)->height << endl;
	}
}

FastVirtualScan::FastVirtualScan(int beamNum, double heightStep, double minFloor, double maxCeiling)
{
    beamnum=beamNum;
    step=heightStep;
    minfloor=minFloor;
    maxceiling=maxCeiling;
    rotation=0;
    minrange=0;
    
    int size=int((maxceiling-minfloor)/step+0.5); // 160
    
    svs = new SimpleVirtualScan[beamnum*size];
    svsback = new SimpleVirtualScan[beamnum*size];
}

FastVirtualScan::~FastVirtualScan()
{
	delete[] svs;
	delete[] svsback;
}

bool compareDistance(const SimpleVirtualScan & svs1, const SimpleVirtualScan & svs2)
{
    if(svs1.rotlength==svs2.rotlength)
    {
        return svs1.rotid>svs2.rotid;
    }
    else
    {
        return svs1.rotlength<svs2.rotlength;
    }
}

void FastVirtualScan::calculateVirtualScans(const std::vector<cv::Vec3f> &pointcloud, int beamNum, double heightStep, double minFloor, double maxCeiling, double obstacleMinHeight, double maxBackDistance, double beamRotation, double minRange)
{

    assert(minFloor<maxCeiling);

    beamnum=beamNum; // 720
    step=heightStep;
    minfloor=minFloor;
    maxceiling=maxCeiling;
    rotation=beamRotation;
    minrange=minRange;
    double c=cos(rotation);
    double s=sin(rotation);

    double PI=3.141592654;
    double density=2*PI/beamnum;

    int size=int((maxceiling-minfloor)/step+0.5); // 160


    //initial Simple Virtual Scan
    timers.resetTimer("initial_simple_virtual_scan");
    {
/*#ifdef USEOMP
#ifndef QT_DEBUG
#pragma omp parallel for
#endif
#endif*/
        for(int i=0;i<beamnum*size;i++)
        {
        	int j = i%size;

            svs[i].rotid=j;
            svs[i].length=MAXVIRTUALSCAN;
            svs[i].rotlength=MAXVIRTUALSCAN;
            svs[i].rotheight=minfloor+(j+0.5)*step;
            svs[i].height=minfloor+(j+0.5)*step;
        }
    }
    timers.pauseTimer("initial_simple_virtual_scan");

    //set SVS
    timers.resetTimer("set_svs");
    {
        int i,n=pointcloud.size();

        //O(P)
        for(i=0;i<n;i++)
        {
            const cv::Vec3f &point = pointcloud[i];
		    double length=sqrt(point[0]*point[0]+point[1]*point[1]);
		    double rotlength=length*c-point[2]*s;
		    double rotheight=length*s+point[2]*c;
		    int rotid=int((rotheight-minfloor)/step+0.5);
		    if(rotid>=0&&rotid<size)
		    {
		        double theta=atan2(point[1],point[0]);
		        int beamid=int((theta+PI)/density);
		        if(beamid<0)
		        {
		            beamid=0;
		        }
		        else if(beamid>=beamnum)
		        {
		            beamid=beamnum-1;
		        }
		        if(length > minrange && svs[beamid*size + rotid].rotlength > rotlength)
		        {
		            svs[beamid*size + rotid].rotlength=rotlength;
		            svs[beamid*size + rotid].length=svs[beamid*size + rotid].rotlength*c+svs[beamid*size + rotid].rotheight*s;
		            svs[beamid*size + rotid].height=-svs[beamid*size + rotid].rotlength*s+svs[beamid*size + rotid].rotheight*c;
		        }
		    }
        }
    }
    timers.pauseTimer("set_svs");

    //sorts
    timers.resetTimer("sorts");
    {
#ifdef USEOMP
#ifndef QT_DEBUG
#pragma omp parallel for \
    default(shared) \
    schedule(static)
#endif
#endif
        for(int i=0;i<beamnum;i++) {
		    int j;
		    bool flag=1;
		    int startid=0;
		    for(j=0;j<size;j++)
		    {
		        if(flag)
		        {
		            if(svs[i*size + j].rotlength<MAXVIRTUALSCAN)
		            {
		                flag=0;
		                startid=j;
		            }
		            continue;
		        }
		        if(svs[i*size + j].rotlength<MAXVIRTUALSCAN && startid==j-1)
		        {
		            startid=j;
		        }
		        else if(svs[i*size + j].rotlength<MAXVIRTUALSCAN)
		        {
		            if(svs[i*size + j].height-svs[i*size + startid].height<obstacleMinHeight&&svs[i*size + j].rotlength-svs[i*size + startid].rotlength>-maxBackDistance)
		            {
		                double delta=(svs[i*size + j].rotlength-svs[i*size + startid].rotlength)/(j-startid);
		                int k;
		                for(k=startid+1;k<j;k++)
		                {
		                    svs[i*size + k].rotlength = svs[i*size + j].rotlength-(j-k)*delta;
		                    svs[i*size + k].length = svs[i*size + k].rotlength*c+svs[i*size + k].rotheight*s;
		                    svs[i*size + k].height = -svs[i*size + k].rotlength*s+svs[i*size + k].rotheight*c;
		                }
		            }
		            startid=j;
		        }
		    }
		    svs[(i*size)+size-1].rotlength=MAXVIRTUALSCAN;
		    std::copy(svs+(i*size), svs+(i*size)+size, svsback+(i*size));
		    std::sort(svs+(i*size),svs+(i*size)+size,compareDistance);
		}
    }
    timers.pauseTimer("sorts");
}


void FastVirtualScan::getVirtualScan(double thetaminheight, double thetamaxheight, double maxFloor, double minCeiling, double passHeight, QVector<double> &virtualScan)
{
    virtualScan.fill(MAXVIRTUALSCAN,beamnum);
    minheights.fill(minfloor,beamnum);
    maxheights.fill(maxceiling,beamnum);

    QVector<double> rotVirtualScan;
    rotVirtualScan.fill(MAXVIRTUALSCAN,beamnum);

    int size=int((maxceiling-minfloor)/step+0.5);
    double deltaminheight=fabs(step/tan(thetaminheight));
    double deltamaxheight=fabs(step/tan(thetamaxheight));

#ifdef USEOMP
#ifndef QT_DEBUG
#pragma omp parallel for \
    default(shared) \
    schedule(static)
#endif
#endif
    for(int i=0;i<beamnum;i++)
    {
        int candid=0;
        bool roadfilterflag=1;
        bool denoiseflag=1;
        while(candid<size&&svs[i*size+candid].height>minCeiling)
        {
            candid++;
        }
        if(candid>=size||svs[i*size+candid].rotlength==MAXVIRTUALSCAN)
        {
            virtualScan[i]=0;
            minheights[i]=0;
            maxheights[i]=0;
            continue;
        }
        if(svs[i*size+candid].height>maxFloor)
        {
            virtualScan[i]=svs[i*size+candid].length;
            minheights[i]=svs[i*size+candid].height;
            denoiseflag=0;
            roadfilterflag=0;
        }
        int firstcandid=candid;
        for(int j=candid+1;j<size;j++)
        {
            if(svs[i*size+j].rotid<=svs[i*size+candid].rotid)
            {
                continue;
            }
            int startrotid=svs[i*size+candid].rotid;
            int endrotid=svs[i*size+j].rotid;

            if(svs[i*size+j].rotlength==MAXVIRTUALSCAN)
            {
                if(roadfilterflag)
                {
                    virtualScan[i]=MAXVIRTUALSCAN;
                    minheights[i]=0;//svsback[i][startrotid].height;
                    maxheights[i]=0;//svsback[i][startrotid].height;
                }
                else
                {
                    maxheights[i]=svsback[i*size+startrotid].height;
                }
                break;
            }
            else
            {
                if(denoiseflag)
                {
                    if(startrotid+1==endrotid)
                    {
                        if(svs[i*size+j].rotlength-svs[i*size+candid].rotlength>=deltaminheight)
                        {
                            denoiseflag=0;
                            roadfilterflag=1;
                        }
                        else if(svs[i*size+j].height>maxFloor)
                        {
                            virtualScan[i]=svs[i*size+firstcandid].length;
                            minheights[i]=svs[i*size+firstcandid].height;
                            denoiseflag=0;
                            roadfilterflag=0;
                        }
                    }
                    else
                    {
                        if(svs[i*size+j].height-svs[i*size+candid].height<=passHeight)
                        {
                            if(svs[i*size+j].rotlength-svs[i*size+candid].rotlength<=deltaminheight)
                            {
                                virtualScan[i]=svsback[i*size+startrotid].length;
                                minheights[i]=svsback[i*size+startrotid].height;
                                denoiseflag=0;
                                roadfilterflag=0;
                            }
                            else
                            {
                                virtualScan[i]=svs[i*size+j].length;
                                for(int k=startrotid+1;k<endrotid;k++)
                                {
                                    if(virtualScan[i]>svsback[i*size+k].length)
                                    {
                                        virtualScan[i]=svsback[i*size+k].length;
                                    }
                                }
                                minheights[i]=svsback[i*size+startrotid+1].height;
                                denoiseflag=0;
                                roadfilterflag=0;
                            }
                        }
                        else
                        {
                            continue;
                        }
                    }
                }
                else
                {
                    if(roadfilterflag)
                    {
                        if(startrotid+1==endrotid)
                        {
                            if(svs[i*size+j].rotlength-svs[i*size+candid].rotlength<=deltaminheight)
                            {
                                virtualScan[i]=svsback[i*size+startrotid].length;
                                minheights[i]=svsback[i*size+startrotid].height;
                                roadfilterflag=0;
                            }
                        }
                        else
                        {
                            if(svs[i*size+j].height-svs[i*size+candid].height<=passHeight)
                            {
                                if(svs[i*size+j].rotlength-svs[i*size+candid].rotlength<=deltaminheight)
                                {
                                    virtualScan[i]=svsback[i*size+startrotid].length;
                                    minheights[i]=svsback[i*size+startrotid].height;
                                    roadfilterflag=0;
                                }
                                else
                                {
                                    virtualScan[i]=svs[i*size+j].length;
                                    for(int k=startrotid+1;k<endrotid;k++)
                                    {
                                        if(virtualScan[i]>svsback[i*size+k].length)
                                        {
                                            virtualScan[i]=svsback[i*size+k].length;
                                        }
                                    }
                                    minheights[i]=svsback[i*size+startrotid+1].height;
                                    roadfilterflag=0;
                                }
                            }
                            else
                            {
                                continue;
                            }
                        }
                    }
                    else
                    {
                        if(svs[i*size+j].rotlength-svs[i*size+candid].rotlength>deltamaxheight)
                        {
                            maxheights[i]=svsback[i*size+startrotid].height;
                            break;
                        }
                    }
                }
            }
            candid=j;
        }
        if(virtualScan[i]<=0)
        {
            virtualScan[i]=0;
            minheights[i]=0;
            maxheights[i]=0;
        }
    }
}
