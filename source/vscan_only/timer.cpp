#include <iostream>
#include <chrono>
#include <string>

using namespace std;
using namespace std::chrono;

class Timer {
    high_resolution_clock::time_point start;
    string title;
    bool is_paused;
    double elapsed;

public:
    Timer(bool _start=false, string _title="") : title(_title) {
        if(_start) {
            this->start = high_resolution_clock::now();
            this->is_paused = false;
        } else {
            this->is_paused = true;
        }
        this->elapsed = 0;
    }

    void pause() {
        if(!this->is_paused) {
            auto stop = high_resolution_clock::now();
            auto elapsed_ = duration_cast<duration<double>>(stop - this->start);
            this->elapsed += elapsed_.count();
            this->is_paused = true;
        }
    }

    void resume() {
        if(this->is_paused) {
            this->start = high_resolution_clock::now();
            this->is_paused = false;
        }
    }

    void reset(bool _start=true) {
        this->elapsed = 0;
        if(_start) {
            this->start = high_resolution_clock::now();
            this->is_paused = false;
        } else {
            this->is_paused = true;
        }
    }

    double getElapsed() {
        if(this->is_paused) return this->elapsed * 1000;

        duration<double> elapsed;
        auto stop = high_resolution_clock::now();
        elapsed = duration_cast<duration<double>>(stop - this->start);
        return (this->elapsed + elapsed.count()) * 1000;
    }

    void changeTitle(string _title) {
        this->title = _title;
    }

    string getTitle() {
        return this->title;
    }
};