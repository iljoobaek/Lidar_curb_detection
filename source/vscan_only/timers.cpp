#include "timers.h"
using namespace std;

void Timers::addTimer(string name, bool start) {
  Timer timer = Timer(start, name);
  timersMap[name] = timer;
  timerMetrics[name] = vector<double>{1e10f,0.0f,0.0f,0.0f}; // min,max,total,count
}

void Timers::resetTimer(string name) {
  auto timer = getTimerFromName(name);
  if(timer != NULL) timer->reset();
}

void Timers::pauseTimer(string name) {
  auto timer = getTimerFromName(name);
  if(timer != NULL) {
    timer->pause();
    updateTimer(name);
  }
}

void Timers::resumeTimer(string name) {
  auto timer = getTimerFromName(name);
  if(timer != NULL) timer->resume();
}

double Timers::getTimerElapsed(string name) {
  auto timer = getTimerFromName(name);
  if(timer != NULL) return timer->getElapsed();
  return -1;
}

void Timers::removeTimer(string name) {
  if(getTimerFromName(name) != NULL) {
    timersMap.erase(name);
    timerMetrics.erase(name);
  }
}

void Timers::updateTimer(string name) {
  auto timer = getTimerFromName(name);
  if(timer != NULL) {
    auto timerMetric = timerMetrics.at(name);
    if(timer->getElapsed() < timerMetrics[name][0]) {
      timerMetrics[name][0] = timer->getElapsed();
    }
    if(timer->getElapsed() > timerMetrics[name][1]) {
      timerMetrics[name][1] = timer->getElapsed();
    }
    timerMetrics[name][2] += timer->getElapsed();
    timerMetrics[name][3] += 1;
  }
}

Timer* Timers::getTimerFromName(string name) {
  try {
    return &timersMap.at(name);
  } catch(exception& out_of_range) {
    return NULL;
  }
}

void Timers::printToFile(string fileName, string optType, string optInfo) {
  vector<string> vec{"pointcloud_preprocessing", "runDetection", "line_fitting", "tracking", "virtualscan"};
  
  ofstream mf;
  mf.open(fileName);
  
  mf << optType << ":\n";
  mf << "===========================================================================" << "\n\n";
  mf << "Info: \n" << optInfo << "\n";
  mf << "===========================================================================" << "\n\n";

  for(int i = 0; i < vec.size(); ++i) {
	  mf << vec[i] << ": avg= " << timerMetrics[vec[i]][2] / timerMetrics[vec[i]][3] << " min= " << timerMetrics[vec[i]][0] << " max= " << timerMetrics[vec[i]][1] << " total= " << timerMetrics[vec[i]][2] << " count= " << timerMetrics[vec[i]][3] << "\n";
  }
  
  mf.close();
}

/*
min: 0
max: 1
total: 2
count: 3
*/

void Timers::print() {
  cout << "\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
  for(auto iter = timerMetrics.begin(); iter != timerMetrics.end(); ++iter) {
    if(iter->second[0] != 0) {
      cout << iter->first << " avg= " 
      << iter->second[2] / iter->second[3] << ": min= " << iter->second[0] 
      << " max= " << iter->second[1] << " total= " << iter->second[2] << " count= " 
      << iter->second[3] << endl;
    }
  }
  cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n";
}


void setupTimers(Timers &timers) {
  timers.addTimer("virtualscan");
  timers.addTimer("visualization");
  timers.addTimer("frame");

  // Timers in fasatvirtualscan.cpp
  timers.addTimer("calculateVirtualScans");
  timers.addTimer("initial_simple_virtual_scan");
  timers.addTimer("set_svs");
  timers.addTimer("sorts");
  timers.addTimer("getVirtualScan");
  timers.addTimer("getVscanResult");
}
