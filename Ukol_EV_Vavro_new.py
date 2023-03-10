# -*- coding: utf-8 -*-
import sys, os
from math import sqrt, sin, cos, pi
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime
from statistics import mean, median, stdev
#import yappi
from numba import jit

cm = 1/2.54  # centimeters in inches

# benchmarking functions
@jit
def computeDeJong1(x):
    return sum([xi**2 for xi in x])

@jit
def computeDeJong2(x):
    return sum([100 * (x[i]**2 - x[i+1])**2 + (1 - x[i])**2 for i in range(len(x)-1)])

@jit
def computeSchwefel(x):
    return sum([-xi * sin(sqrt(abs(xi))) for xi in x])

@jit
def computeRastrigin(x):
    return 10 * len(x) + sum([xi**2 - 10 * cos(2*pi*xi) for xi in x])

# list of dimensions
dimensions = [10, 30]

# dict of bench. functions
functions = {
    0 : computeDeJong1,
    1 : computeDeJong2,
    2 : computeSchwefel,
    3 : computeRastrigin
}

functionNames = {
    0 : 'DeJong 1st',
    1 : 'DeJong 2nd',
    2 : 'Schwefel',
    3 : 'Rastrigin'
}

# x-es, dimensions 1 .. N-1
functionRanges = {
    0 : (-5.12, 5.12),
    1 : (-5.12, 5.12),
    2 : (-500.0, 500.0),
    3 : (-5.12, 5.12)
}

# x-es times dimension N
functionMins = {
    0 : 0.0, # (0.0, 0.0, 0.0, ..., 0.0)
    1 : 1.0,
    2 : 420.9687,
    3 : 0.0
}
functionMaxs = {
    0 : functionRanges[0][1],
    1 : functionRanges[1][0],
    2 : 301.923,
    3 : 4.5
}

functionMinMax = {
    d:{
        f:[
            functions[f]([functionMins[f] for x in range(d)]), # compute Min
            functions[f]([functionMaxs[f] for x in range(d)])  # compute Max
        ]
        for f in functions.keys()
    } for d in dimensions
}

# for each method a dimensional dictionary will contain
# a list of 30 lists for each function which will contain results
results = {
    "Particle Swarm Optimization" :
        {
            d:{
                f:[[] for i in range(30)] for f in functions.keys()
            } for d in dimensions 
        }
}

class Stats:
    values = []
    minS = sys.float_info.max
    maxS = sys.float_info.min
    meanS = 0.0
    medianS = 0.0
    stdDevS = 0.0

statsDict = {
    "Particle Swarm Optimization" :
        {
            d:{
                f:Stats() for f in functions.keys()
            } for d in dimensions 
        }
}

# single particle update
@jit
def particleSwarmOptimizationSingle(func, ranges, dims, particlePosition, particleVelocity, particleBestPos, swarmBestPos, params):
    w = params[0]
    ro_p = params[1]
    ro_g = params[2]
    rp = np.random.uniform(0.0, 1.0, dims)
    rg = np.random.uniform(0.0, 1.0, dims)
    particleVelocity = w * particleVelocity + ro_p * rp * (particleBestPos - particlePosition) + ro_g * rg * (swarmBestPos - particlePosition)
    particlePosition = ranges[0] + (
                (ranges[0] + particlePosition + particleVelocity) % (ranges[1] - ranges[0]))
    result = func(list(particlePosition))
    if result < func(list(particleBestPos)):
        particleBestPos = particlePosition.copy()
        if result < func(list(swarmBestPos)):
            swarmBestPos = particleBestPos.copy()
    return (particlePosition, particleVelocity, particleBestPos, swarmBestPos)

# swarm update
def particleSwarmOptimization(fes = 5000, particleCount = 50, w_init=0.9, w_end=0.4, ro_p=2.0, ro_g=2.0):
    algName = "Particle Swarm Optimization"
    for d in dimensions:
        print("DIM=" + str(d))
        for fk, fv in functionRanges.items():
            print("*FUNC=" + str(fk))
            global statsDict
            statsDict[algName][d][fk].values = []
            for r in range(30):
                global results
                results[algName][d][fk][r] = []
                values = np.random.uniform(fv[0], fv[1], d)
                bestValues = np.copy(values)
                particles = np.random.uniform(fv[0], fv[1], (particleCount, d))
                particleVelocities = np. random.uniform(-abs(fv[1] - fv[0]), abs(fv[1] - fv[0]), (particleCount, d))
                particlesBestPositions = np.copy(particles)
                print("{rnd:2d}.[".format(rnd=r+1), end='')
                sys.stdout.flush()
                for j in range(fes):
                    for p in range(particleCount): # particles
                        global functions
                        (particles[p, :], particleVelocities[p, :], particlesBestPositions[p, :], bestValues) = particleSwarmOptimizationSingle(functions[fk], functionRanges[fk], d, particles[p, :], particleVelocities[p, :], particlesBestPositions[p, :], bestValues, (w_end + (j/fes)*(w_init - w_end), ro_p, ro_g))
                    results[algName][d][fk][r].append(functions[fk](bestValues))
                    statsDict[algName][d][fk].values.extend(results[algName][d][fk][r])
                    if (((j / fes) * 100) % 10) == 0:
                        print('*', end='')
                        sys.stdout.flush()
                print(']')
            statsDict[algName][d][fk].minS = min(statsDict[algName][d][fk].values)
            statsDict[algName][d][fk].maxS = max(statsDict[algName][d][fk].values)
            statsDict[algName][d][fk].meanS = mean(statsDict[algName][d][fk].values)
            statsDict[algName][d][fk].medianS = median(statsDict[algName][d][fk].values)
            statsDict[algName][d][fk].stdDevS = stdev(statsDict[algName][d][fk].values)
            print(algName + ", Dims=" + str(d) + ", Bench=" + functionNames[fk] + ":\n" + str(statsDict[algName][d][fk].minS) + ", " + str(statsDict[algName][d][fk].maxS) + ", " + str(statsDict[algName][d][fk].meanS) + ", " + str(statsDict[algName][d][fk].medianS) + ", " + str(statsDict[algName][d][fk].stdDevS))

def plotGraphs():
    global results, cm
    with PdfPages('Ukol_EV_Vavro.pdf') as pdf:
        firstPage = plt.figure(figsize=(20.0*cm,28.7*cm))
        firstPage.clf()
        firstPage.text(0.5, 0.55, '??kol do p??edm??tu EV', transform=firstPage.transFigure, size=24, ha="center")
        firstPage.text(0.5, 0.45, 'Ondrej Vavro, 2022/2023', transform=firstPage.transFigure, size=18, ha="center")
        pdf.savefig()
        plt.close()
        
        # comparison graphs
        comparisons = [plt.subplots(1, len(functions.keys()), figsize=(28.7*cm,20.0*cm)) for d in dimensions]
        
        for a in [("Particle Swarm Optimization", 1)]:#("Random Search", 1), ("Hill Climbing", neighbours)]: # algorithms and neighbourhoods
            titlePage = plt.figure(figsize=(20.0*cm,28.7*cm))
            titlePage.clf()
            titlePage.text(0.5, 0.5, a[0], transform=titlePage.transFigure, size=24, ha="center")
            pdf.savefig()
            plt.close()
        
            for d in dimensions:
                comparisons[dimensions.index(d)][0].text(0.5, 0.90, 'Dimensions='+str(d), transform=comparisons[dimensions.index(d)][0].transFigure, size=24, ha="center")
                comparisons[dimensions.index(d)][0].subplots_adjust(wspace=0.4)
                (fig, ax) = plt.subplots(len(functions.keys()), 2, figsize=(20.0*cm,28.7*cm))
                plt.subplots_adjust(wspace=0.4, hspace=0.4)
                fig.clf()
                fig.text(0.5, 0.94, 'Dimensions='+str(d), transform=fig.transFigure, size=24, ha="center")
                for f in functions.keys():
                    avg = [0.0 for i in range(len(results[a[0]][d][f][0]))]
                    plt.subplot(len(functions.keys()), 2, f*2 + 1)
                    plt.ylim(functionMinMax[d][f])
                    for i in results[a[0]][d][f]:
                        avg = [x + y for x, y in zip(avg, i)]
                        plt.plot([x for x in range(0,len(i)*a[1],a[1])], i)
                    plt.xlabel('Generations')
                    plt.ylabel('CF Value')
                    plt.title('Benchmark=' + functionNames[f] + ', 30 runs')
                    plt.subplot(len(functions.keys()), 2, f*2 + 2)
                    avg = [x / len(results[a[0]][d][f]) for x in avg]
                    plt.ylim(functionMinMax[d][f])
                    plt.plot([x for x in range(0,len(results[a[0]][d][f][0])*a[1],a[1])], avg)
                    plt.xlabel('Generations')
                    plt.ylabel('CF Value')
                    plt.title('Benchmark=' + functionNames[f] + ', Average')
                    #comparisons[dimensions.index(d)][1][f].axis(ymin=functionMinMax[d][f][0],ymax=functionMinMax[d][f][1])
                    comparisons[dimensions.index(d)][1][f].set_box_aspect(1)
                    comparisons[dimensions.index(d)][1][f].set_xlabel('Generations')
                    comparisons[dimensions.index(d)][1][f].set_ylabel('CF Value')
                    comparisons[dimensions.index(d)][1][f].set_title('Benchmark=' + functionNames[f])                    
                    comparisons[dimensions.index(d)][1][f].plot([x for x in range(0,len(results[a[0]][d][f][0])*a[1],a[1])], avg, label=a[0])
                pdf.savefig()
                plt.close()
                
        statsPage = plt.figure(figsize=(28.7*cm,20.0*cm))
        statsPage.clf()
        statsPage.text(0.5, 0.5, 'Porovn??n?? statistik', transform=statsPage.transFigure, size=24, ha="center")
        pdf.savefig()
        plt.close()
        
        # tabulky
        (tablePage, ax) = plt.subplots(2, 1, figsize=(28.7*cm, 20.0*cm))
        tablePage.patch.set_visible(False)
        for a in statsDict:
            ax[list(statsDict.keys()).index(a)].axis('off')
            ax[list(statsDict.keys()).index(a)].axis('tight')
            cols = ('Min', 'Max', 'Mean', 'Median', 'StdDev')
            rows = ["D={},F={}".format(d,functionNames[f]) for d in statsDict[a] for f in statsDict[a][d]]
            cell_text = []
            for d in statsDict[a]:
                for fk,fv in statsDict[a][d].items():
                    cell_text.append([fv.minS, fv.maxS, fv.meanS, fv.medianS, fv.stdDevS])
            plt.subplot(2,1,list(statsDict.keys()).index(a)+1)
            tablePage.text(0.5, 0.94-list(statsDict.keys()).index(a)*0.5, a, transform=tablePage.transFigure, size=24, ha="center")
            statTable = plt.table(cellText=cell_text,
                                  rowLabels=rows,
                                  colLabels=cols,
                                  loc='center')
        tablePage.tight_layout()
        pdf.savefig()
        plt.close()
        
        compPage = plt.figure(figsize=(28.7*cm,20.0*cm))
        compPage.clf()
        compPage.text(0.5, 0.5, 'Porovn??n?? algoritm??', transform=compPage.transFigure, size=24, ha="center")
        pdf.savefig()
        plt.close()
        for d in dimensions:
            for f in functions.keys():
                comparisons[dimensions.index(d)][1][f].legend()
            pdf.savefig(comparisons[dimensions.index(d)][0])
            
        
        info = pdf.infodict()
        info['Title'] = '??kol do EV 2022/2023'
        info['Author'] = 'Ondrej Vavro'
        info['Subject'] = 'B??h evolu??n??ho algoritmu na vybran??ch ????elov??ch funkc??ch'
        info['Keywords'] = 'Particle Swarm Optimization, 1st DeJong, 2nd DeJong, Schweffel, Rastrigin'
        info['CreationDate'] = datetime.datetime.today()
        info['ModDate'] = datetime.datetime.today()

def main():
    #yappi.set_clock_type("cpu")
    #yappi.start()
    #np.seterr(all='raise')
    particleSwarmOptimization(5000)
    #yappi.get_func_stats().print_all()
    #yappi.get_thread_stats().print_all()
    #randomSearch(10000)
    #hillClimbing(10000, neighbours)
    plotGraphs()    
    
if __name__ == "__main__":
    main()