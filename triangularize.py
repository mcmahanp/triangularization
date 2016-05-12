#!/usr/bin/python
# -*- coding: utf8 -*-

import scipy as sp 
import numba
import sys

'''heuristic matrix triangularization from
    Tommaso Schiavinotto and Thomas Stützle.
    The linear ordering problem: Instances, 
    search space analysis and algorithms. 
    Journal of Mathematical Modelling and 
    Algorithms, 3(4):367–402, 2005.'''





@numba.jit(nopython=True)
def randMat(n,d1=.05,d2=.4):
    res = sp.zeros((n,n),dtype=sp.int64)
    for i in range(n):
        for j in range(n):
            if i>j:
                res[i,j] = sp.random.uniform(0,1) < d1
            elif i<j:
                res[i,j] = sp.random.uniform(0,1) < d2
    return(res)






#####
# functions to calculate objective function and deltas
#####
@numba.jit(nopython=True,nogil=True)
def obj(mat,p):
    n = len(p)
    res = 0
    for i in range(0,n):
        for j in range(i+1,n):
            res += mat[p[i],p[j]]
    return(res)

@numba.jit(nopython=True,nogil=True)
def delta_swap(mat,p,i):
    '''change in objective function for a swap at i'''
    pi = p[i]
    pii = p[i+1]
    return(mat[pii,pi] - mat[pi,pii])


#####
# functions to alter permutations for local search
#####
@numba.jit(nopython=True,nogil=True)
def swap(p,i):
    '''swap p[i] and p[i+1]
    IN PLACE'''
    pi = p[i]
    p[i] = p[i+1]
    p[i+1] = pi


# 0,1,2,3,4
# 1,2,3,0,4
@numba.jit(nopython=True,nogil=True)
def insert(p,i,j):
    '''if i<j insert p[i] after j, otherwise insert p[i] before j.
    IN PLACE'''
    if i<j:
        pi = p[i]
        for k in range(i,j):
            p[k] = p[k+1]
        p[j] = pi
    else:
        pi = p[i]
        for k in range(i,j,-1):
            p[k] = p[k-1]
        p[j] = pi

#####
# local search
#####
@numba.jit(nopython=True,nogil=True)
def visit(mat,p):
    '''single step of local search with first pivot.
    returns 1 if update occured, 0 otherwise.'''
    obj0 = obj(mat,p)
    n = len(p)
    for i in range(n):
        rbar = i
        objbar = 0
        # backwards
        obj1 = obj0
        for j in range(i-1,-1,-1):
            obj1 += delta_swap(mat,p,j)
            swap(p,j)
            if obj1 > objbar:
                rbar = j
                objbar = obj1
        insert(p,0,i) # put it back
        # forwards
        obj1 = obj0
        for j in range(i,n-1):
            obj1 += delta_swap(mat,p,j)
            swap(p,j)
            if obj1 > objbar:
                rbar = j+1
                objbar = obj1
        insert(p,n-1,i) # put it back
        if objbar > obj0:
            insert(p,i,rbar)
            return(1)
    return(0)

@numba.jit(nopython=True,nogil=True)
def localsearch(mat,p,maxiter = 100000):
    '''run visit(mat,p) until no more improvement'''
    for i in range(maxiter):
        if visit(mat,p)==0:
            break




#####
# memetic (selection) algorithm
#####
def crossover(p1,p2):
    n = len(p1)
    res = p1.copy()
    inds = [i for i in range(n) if sp.random.uniform(0,1) < 0.5]
    res[inds] = p1[inds][sp.argsort(-p2[inds])]
    return(res)

def select_best(pop,m):
    '''select the m best items from population'''
    inds = sp.argsort([-s for (p,s) in pop])[:m]
    return([pop[i] for i in inds])

def MA(mat,npopulation=25,noffspring=11,diversify_after=30,stop_after=150):
    '''use memetic algorithm to find approx optimal permutation'''
    # initial population
    n = mat.shape[0]
    population = []
    for i in range(npopulation):
        p = sp.arange(n,dtype=sp.int64)
        sp.random.shuffle(p)
        localsearch(mat,p)
        population.append((p,obj(mat,p)))

    # repeat until no improvement for 10 diversifications
    iter = 0
    av_score = sp.zeros(diversify_after)
    best = max([s for (p,s) in population])
    stale_count = 0
    while stale_count < stop_after:
        if (iter % diversify_after) == 0:
            sys.stdout.write('\r%6d (%3d): %d' % (iter,stale_count,best))
            sys.stdout.flush()
        iter += 1
        # crossover offspring
        for i in range(noffspring):
            a1,a2 = sp.random.choice(npopulation,2,False)
            p1 = population[a1][0]
            p2 = population[a2][0]
            p3 = crossover(p1,p2)
            localsearch(mat,p3)
            population.append((p3,obj(mat,p3)))
        sp.random.shuffle(population)
        population = select_best(population,npopulation)
        gen_best = max([s for (p,s) in population])
        if gen_best > best:
            stale_count = 0
            best = gen_best
        else:
            stale_count += 1
        av_score[iter % diversify_after] = sp.mean([s for (p,s) in population])
        if (iter % diversify_after) == 0 and max(av_score) == min(av_score):
            # diversify
            population = select_best(population,1)
            for i in range(npopulation-1):
                p = sp.arange(n,dtype=sp.int64)
                sp.random.shuffle(p)
                localsearch(mat,p)
                population.append((p,obj(mat,p)))
    sys.stdout.write('\r%6d (%3d): %d\n' % (iter,stale_count,best))
    sys.stdout.flush()
    return(select_best(population,1)[0][0])






