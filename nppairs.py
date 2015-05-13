import numpy as np
from numba import vectorize, double

class NumpyPairCounter(object):

    def __init__(self, Lbox, cell_size):

        self.reset_grid(Lbox, cell_size)

    def retrieve_tree(self, x, y, z):
        """
        Parameters 
        ----------
        x, y, z : arrays
            Length-Npts arrays containing the spatial position of the Npts points. 

        Returns 
        --------
        x, y, z : arrays
            Length-Npts arrays containing the spatial position of the Npts points, 
            after sorting the points according to their cellID. 

        slice_array: array
            Array of slice objects. The x-coordinates of points residing in 
            cellID = i can be accessed as x[slice_array[i]]. 

        """

        idx_sorted, slice_array = self.compute_cell_structure(x, y, z)

        return x[idx_sorted], y[idx_sorted], z[idx_sorted], slice_array


    def reset_grid(self, Lbox, cell_size):
        """ Reset the internal variables specifying the grid. 

        Parameters 
        ----------
        Lbox : float
            Length scale defining the periodic boundary conditions

        cell_size : float 
            The approximate cell size into which the box will be divided. 
        """

        self.Lbox = Lbox
        self.cell_size = cell_size
        self.num_divs = np.floor(Lbox/float(cell_size)).astype(int)
        self.dL = Lbox/float(self.num_divs)

    def num_pairs(self, x, y, z, rbins):
        """
        Parameters 
        ----------
        x, y, z : arrays
            Length-Npts arrays containing the spatial position of the Npts points. 

        rbins : array 
            Defines the bins of separation into which pair distances will be discretized. 

        Returns 
        -------
        num_pairs : array 
            Number of pairs of points residing in the bins. 

        Notes 
        -----
        Super slow. Durn. 
        """

        x, y, z, slice_array = self.retrieve_tree(x, y, z)

        # Loop over all subvolumes in the entire box
        for icell in range(self.num_divs**3):
            x_icell, y_icell, z_icell = (
                x[slice_array[icell]], y[slice_array[icell]], z[slice_array[icell]]
                )
            ix, iy, iz = np.unravel_index(icell, 
                (self.num_divs, self.num_divs, self.num_divs))
            adj_cell_arr = self.adjacent_cells(ix, iy, iz)

            # Loop over each of the 27 subvolumes neighboring (and including) cellID=icell
            for icell2 in adj_cell_arr:
                x_icell2 = x[slice_array[icell2]]
                y_icell2 = z[slice_array[icell2]]
                z_icell2 = x[slice_array[icell2]]
                
                # Loop over all points in icell2
                for x2, y2, z2 in zip(x_icell2, y_icell2, z_icell2):
                    pass
                    #xdist = self.numba_vectorized_periodic_distance(x2, x_icell, self.Lbox)
                    #ydist = self.numba_vectorized_periodic_distance(y2, y_icell, self.Lbox)
                    #zdist = self.numba_vectorized_periodic_distance(z2, z_icell, self.Lbox)

    def compute_cell_structure(self, x, y, z):
        """ Method divides the periodic box into regular, 
        cubical subvolumes, and assigns a subvolume index to each point. 
        The returned arrays can be used to efficiently access only those 
        points in a given subvolume. 

        Parameters 
        ----------
        x, y, z : arrays
            Length-Npts arrays containing the spatial position of the Npts points. 

        Returns 
        -------
        idx_sorted : array
            Array of indices that sort the points according to the dictionary 
            order of the 3d subvolumes. 

        slice_array : array 
            array of slice objects used to access the elements of x, y, and z 
            of points residing in a given subvolume. 

        Notes 
        -----
        The dictionary ordering of 3d cells where :math:`dL = L_{\\rm box} / 2` 
        is defined as follows:

            * (0, 0, 0) <--> 0

            * (0, 0, 1) <--> 1

            * (0, 1, 0) <--> 2

            * (0, 1, 1) <--> 3

        And so forth. Each of the Npts thus has a unique triplet, 
        or equivalently, unique integer specifying the subvolume containing the point. 
        The unique integer is called the *cellID*. 
        In order to access the *x* positions of the points lying in subvolume *i*, 
        x[idx_sort][slice_array[i]]. 

        In practice, because fancy indexing with `idx_sort` is not instantaneous, 
        it will be more efficient to use `idx_sort` once to sort the x, y, and z arrays 
        in-place, and then access the sorted arrays with the relevant slice_array element. 
        This is the strategy used in the `retrieve_tree` method. 

        """

        ix = np.floor(x/self.dL).astype(int)
        iy = np.floor(y/self.dL).astype(int)
        iz = np.floor(z/self.dL).astype(int)

        particle_indices = np.ravel_multi_index((ix, iy, iz), 
            (self.num_divs, self.num_divs, self.num_divs))
        
        idx_sorted = np.argsort(particle_indices)
        bin_indices = np.searchsorted(particle_indices[idx_sorted], 
                                      np.arange(self.num_divs**3))
        bin_indices = np.append(bin_indices, None)
        
        slice_array = np.empty(self.num_divs**3, dtype=object)
        for icell in range(self.num_divs**3):
            slice_array[icell] = slice(bin_indices[icell], bin_indices[icell+1])
            
        return idx_sorted, slice_array

    def adjacent_cells(self, *args):
        """ Given a subvolume specified by the input arguments,  
        return the length-27 array of cellIDs of the neighboring cells. 
        The input subvolume can be specified either by its ix, iy, iz triplet, 
        or by its cellID. 

        Parameters 
        ----------
        ix, iy, iz : int, optional
            Integers specifying the ix, iy, and iz triplet of the subvolume. 
            If ix, iy, and iz are not passed, then ic must be passed. 

        ic : int, optional
            Integer specifying the cellID of the input subvolume
            If ic is not passed, the ix, iy, and iz must be passed. 

        Returns 
        -------
        result : int array
            Length-27 array of cellIDs of neighboring subvolumes. 

        Notes 
        -----
        If one argument is passed to `adjacent_cells`, this argument will be 
        interpreted as the cellID of the input subvolume. 
        If three arguments are passed, these will be interpreted as 
        the ix, iy, iz triplet of the input subvolume. 

        """

        ixgen, iygen, izgen = np.unravel_index(np.arange(3**3), (3, 3, 3)) 

        if len(args) >= 3:
            ix, iy, iz = args[0], args[1], args[2]
        elif len(args) == 1:
            ic = args[0]
            ix, iy, iz = np.unravel_index(ic, (self.num_divs, self.num_divs, self.num_divs))

        ixgen = (ixgen + ix - 1) % self.num_divs
        iygen = (iygen + iy - 1) % self.num_divs
        izgen = (izgen + iz - 1) % self.num_divs

        return np.ravel_multi_index((ixgen, iygen, izgen), 
            (self.num_divs, self.num_divs, self.num_divs))

    def adjacent_points(self, x, y, z, slice_array, *args): 
        """ Return the set of all x, y, z points in the subvolumes 
        including and neighboring the input subvolume.  

        Parameters 
        ----------
        x, y, z : array_like 
            coordinates of all points in the entire box 

        slice_array : array_like 
            Cell structure created by the `cell_structure` method. 

        ix, iy, iz : integers, optional 
            Integers specifying the ix, iy, and iz triplet of the subvolume. 
            If ix, iy, and iz are not passed, then icell must be passed. 

        icell : integer, optional
            Integer specifying the cellID of the input subvolume
            If icell is not passed, the ix, iy, and iz must be passed. 

        Returns 
        --------
        adjx, adjy, adjz : array_like 
            x, y, z coordinates of all points in the volumes 
            including and neighboring the input subvolume. 

        Notes 
        ------
        Method assumes x, y, and z have already been sorted according their cellID. 

        """
        if len(args) >= 3:
            ix, iy, iz = args[0], args[1], args[2]
            ic = np.ravel_multi_index(ix, iy, iz, self.num_divs)
        elif len(args) == 1:
            ic = args[0]

        icells = self.adjacent_cells(ic)
        adjx = np.concatenate([x[slice_array[i]] for i in icells])
        adjy = np.concatenate([y[slice_array[i]] for i in icells])
        adjz = np.concatenate([z[slice_array[i]] for i in icells])

        return adjx, adjy, adjz

    @vectorize([double(double, double, double)])
    def numba_vectorized_periodic_distance(x1, x2, Lbox):
        diff = abs(x2-x1)
        if diff > Lbox/2.:
            diff = abs(diff-Lbox)
        return diff



