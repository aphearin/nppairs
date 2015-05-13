import numpy as np

class NumpyPairCounter(object):

    def __init__(self, Lbox, cell_size):

        self.reset_grid(Lbox, cell_size)
        self.isSorted = False

    def __call__(self, x, y, z, dbins):

        idx_sorted, slice_array = self.cell_structure(x, y, z)
        x, y, z = x[idx_sorted], y[idx_sorted], z[idx_sorted]

        for icell in range(self.num_divs**3):
            x_icell, y_icell, z_icell = (
                x[slice_array[icell]], y[slice_array[icell]], z[slice_array[icell]]
                )
            ix, iy, iz = np.unravel_index(icell, 
                (self.num_divs, self.num_divs, self.num_divs))
            adj_cell_arr = self.adjacent_cells(ix, iy, iz)

            for icell2 in adj_cell_arr:
                x_icell2, y_icell2, z_icell2 = (
                    x[slice_array[icell2]], y[slice_array[icell2]], z[slice_array[icell2]]
                    )
                
                for x2, y2, z2 in zip(x_icell2, y_icell2, z_icell2):
                    dsq = (
                        (x2 - x_icell)*(x2 - x_icell) + 
                        (y2 - y_icell)*(y2 - y_icell) + 
                        (z2 - z_icell)*(z2 - z_icell)
                        )

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

    def cell_structure(self, x, y, z):
        """ Method divides the periodic box into regular, 
        cubical subvolumes, and assigns a subvolume index to each point. 
        The returned arrays can be used to efficiently access only those 
        points in a given subvolume. 

        Parameters 
        ----------
        x, y, z : arrays
            Length-Npts arrays containing the spatial position of the Npts points. 

        Lbox : float
            Length scale defining the periodic boundary conditions

        cell_size : float 
            The approximate cell size into which the box will be divided. 

        Returns 
        -------
        idx_sort : array
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
        In order to access the *x* positions of the points lying in subvolume *i*, 
        x[idx_sort][slice_array[i]]. 

        In practice, because fancy indexing with `idx_sort` is not instantaneous, 
        it will be more efficient to use `idx_sort` to sort the x, y, and z arrays 
        in-place, and then access the sorted arrays with the relevant slice_array element. 

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
        """ Given a subvolume specified by ix, iy, and iz, 
        return the length-27 array of dictionary-order indices 
        of the neighboring cells. 

        Parameters 
        ----------
        ix, iy, iz : int, optional
            integers specifying the x, y, and z subvolume index. 

        ic : int, optional
            integer specifying the dictionary-order of the (ix, iy, iz) triplet

        Returns 
        -------
        result : int array
            Length-27 array of indices. Each element corresponds to the 
            dictionary-order index of a subvolume that is adjacent to 
            the input cell specified by (ix, iy, iz). 

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
        """ Given either a set of x, y, z indices, or a set of cell indices, 
        return all points in adjacent cells. 

        Parameters 
        ----------
        x, y, z : array_like 
            coordinates of all points in the volume 

        slice_array : array_like 
            Cell structure created by the `cell_structure` method. 

        ix, iy, iz : array_like, optional 
            3 integer arrays containing the x, y, and z indices of a set of points. 
            If ix, iy, and iz are not passed, then icell must be passed. 

        icell : array_like, optional
            Integer array containing the cell id of a set of points. 
            If icell is not passed, the ix, iy, and iz must be passed. 

        Returns 
        --------
        adjx, adjy, adjz : array_like 
            Coordinates of all adjacent points

        Notes 
        ------
        Method assumes x, y, and z have already been sorted according their cellid. 

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



