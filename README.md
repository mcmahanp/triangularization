### matrix triangularization in python ###
Implementation of the memetic algorithm from:

Tommaso Schiavinotto and Thomas Stützle. The linear ordering problem: Instances, search space analysis and algorithms. _Journal of Mathematical Modelling and Algorithms_, 3(4):367–402, 2005.

**NOTE**: Requires `numba` to be installed, which can be tricky.

    # make a random 80x80 matrix to test on
    mat = randMat(80,.05,.15)

    # get an approximately optimal permutation (with default parameterss)
    p = MA(mat)

    opt_mat = mat[p,:][:,p]

Default options have been tested work fast enough on matrices up to 200x200.
