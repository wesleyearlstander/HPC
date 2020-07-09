#pragma include once
#ifndef __METRICS_H__
#define __METRICS_H__
#include <math.h>

typedef struct Metrics{
    double serialTime;
    double parallelTime;
    int processes;
    double speedup;
    double efficiency;
} Metrics;

static void Speedup (Metrics* m) {
    m->speedup = m->serialTime/m->parallelTime;
}

static void Efficiency (Metrics* m) {
    m->efficiency = m->speedup / (float)m->processes;
}

static void RunTests (Metrics* m, double serialTime, double parallelTime, int processes) {
    m->serialTime = serialTime;
    m->parallelTime = parallelTime;
    m->processes = processes;
    Speedup(m);
    Efficiency(m);
}

static void RunTestsOnly (Metrics* m) {
    Speedup(m);
    Efficiency(m);
}


#endif
