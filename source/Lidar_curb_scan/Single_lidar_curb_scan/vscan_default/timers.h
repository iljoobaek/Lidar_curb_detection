#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <typeinfo>

#include "timer.cpp"

class Timers {
  map<string, Timer> timersMap;
  map<string, vector<double>> timerMetrics;

public:
  void addTimer(string name, bool start=false);
  void resetTimer(string name);
  void pauseTimer(string name);
  void resumeTimer(string name);
  double getTimerElapsed(string name);
  void removeTimer(string name);
  void updateTimer(string name);
  Timer* getTimerFromName(string name);
  void printToFile(string fileName, string optType, string optInfo);
  void print();
};

void setupTimers(Timers &timers);
