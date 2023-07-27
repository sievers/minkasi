import sys
from . import MapType, SkyMapTwoRes
from ..parallel import have_mpi

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class _MapsetBase:
    """
    Class to store and operate on a set of map objects.
    This class is made to house shared functions to build subclasses with.

    Attributes
    ----------
    nmap : int
        Number of maps in the set.
    maps : list[MapType | SkyMapTwoRes])
        Maps stored in this set.
    """

    def __init__(self):
        """
        Initialize the Mapset.
        By default nmap is 0 and maps in empty.
        """
        self.nmap: int = 0
        self.maps: list[MapType | SkyMapTwoRes] = []

    def add_map(self, map: MapType | SkyMapTwoRes):
        """
        Add a map to the Mapset.

        Parameters
        ----------
        map : MapType | SkyMapTwoRes
            The map to add.
        """
        self.maps.append(map.copy())
        self.nmap = self.nmap + 1

    def copy(self) -> Self:
        """
        Make a copy of this Mapset.

        Returns
        -------
        new_mapset : Mapset
            A copy of this Mapset
        """
        new_mapset: Self = self.__class__()
        for i in range(self.nmap):
            new_mapset.add_map(self.maps[i].copy())
        return new_mapset


class Mapset(_MapsetBase):
    """
    Class to store and operate on a set of map objects.
    This class is made to house shared functions to build subclasses with.

    Attributes
    ----------
    nmap : int
        Number of maps in the set.
    maps : list[MapType])
        Maps stored in this set.
    """

    maps: list[MapType]

    def clear(self):
        """
        Clear all maps in the Mapset.
        """
        for i in range(self.nmap):
            self.maps[i].clear()

    def dot(self, mapset: Self) -> float:
        """
        Take the dot product of this Mapset with another and return the sum.

        Parameters
        ----------
        mapset : Mapset
            Mapset to tahe dot product with.
            Should have the same nmap as this Mapset.

        Returns
        -------
        tot : float
            The sum of the dot products between corresponding maps in the Mapsets.

        Raises
        ------
        ValueError
            If the size of the Mapsets don't match.
        """
        if self.nmap != mapset.nmap:
            raise ValueError(
                f"nmaps don't match, this Mapset has {self.nmap} but the provided Mapset has {mapset.nmap}"
            )
        tot: float = 0.0
        for i in range(self.nmap):
            tot = tot + self.maps[i].dot(mapset.maps[i])
        return tot

    def axpy(self, mapset: Self, a: float):
        """
        Apply a*x + y for all maps in Mapset.

        Parameters
        ----------
        mapset : Mapset
            The set of maps to add in (y).
            Should have the same nmap as this Mapset.

        a : float
            Number to multiply the maps in this Mapset by.

        Raises
        ------
        ValueError
            If the size of the Mapsets don't match.
        """
        if self.nmap != mapset.nmap:
            raise ValueError(
                f"nmaps don't match, this Mapset has {self.nmap} but the provided Mapset has {mapset.nmap}"
            )
        for i in range(self.nmap):
            self.maps[i].axpy(mapset.maps[i], a)

    def __add__(self, mapset: Self) -> Self:
        """
        Add maps in two mapsets.

        Parameters
        ----------
        mapset : Mapset
            Mapset with maps to add to maps in this mapset.
            Should have the same nmap as this Mapset.

        Returns
        -------
        mm : Mapset
            Mapset containing the summed maps.
        """
        mm = self.copy()
        mm.axpy(mapset, 1.0)
        return mm

    def __sub__(self, mapset: Self) -> Self:
        """
        Subtract maps in two mapsets.

        Parameters
        ----------
        mapset : Mapset
            Mapset with maps to subtract maps in this mapset from.
            Should have the same nmap as this Mapset.

        Returns
        -------
        mm : Mapset
            Mapset containing the subtracted maps.
        """
        mm = self.copy()
        mm.axpy(mapset, -1.0)
        return mm

    def __mul__(self, mapset: Self) -> Self:
        """
        Multiply maps in two mapsets.

        Parameters
        ----------
        mapset : Mapset
            Mapset with maps to multiply maps in this mapset with.
            Should have the same nmap as this Mapset.

        Returns
        -------
        mm : Mapset
            Mapset containing the multiplied maps.

        Raises
        ------
        ValueError
            If the size of the Mapsets don't match.
        """
        if self.nmap != mapset.nmap:
            raise ValueError(
                f"nmaps don't match, this Mapset has {self.nmap} but the provided Mapset has {mapset.nmap}"
            )
        mm = mapset.copy()
        for i in range(self.nmap):
            mm.maps[i] = self.maps[i] * mapset.maps[i]
        return mm

    def get_caches(self):
        """
        Get caches for all maps in Mapset
        """
        for i in range(self.nmap):
            self.maps[i].get_caches()

    def clear_caches(self):
        """
        Clear caches for all maps in Mapset
        """
        for i in range(self.nmap):
            self.maps[i].clear_caches()

    def mpi_reduce(self):
        """
        Reduce all maps in mapset.
        If not running with MPI this does nothing.
        """
        if have_mpi:
            for map in self.maps:
                map.mpi_reduce()


class MapsetTwoRes(_MapsetBase):
    """
    Class to store and operate on a set of SkyMapTwoRes objects.

    Attributes
    ----------
    nmap : int
        Number of maps in the set.
    maps : list[SkyMapTwoRes])
        Maps stored in this set.
    """

    maps: list[SkyMapTwoRes]

    def apply_prior(self, x: Mapset, Ax: Mapset):
        """
        Apply prior to all maps in Mapset.
        This only makes sense to do if maps are SkyMapTwoRes.

        Parameters
        ----------
        x : Mapset
            Map or Mapset to use as prior.
            Should be a Mapset of SkyMaps (no pol support yet).

        Ax : Mapset
            The output mapset with priors applied.
            Should be a Mapset of SkyMaps (no pol support yet).
        """
        for i in range(self.nmap):
            if not (self.maps[i] is None):
                try:
                    if self.maps[i].isglobal_prior:
                        # print('applying global prior')
                        self.maps[i].apply_prior(x, Ax)
                    else:
                        self.maps[i].apply_prior(x.maps[i], Ax.maps[i])
                except:
                    # print('going through exception')
                    self.maps[i].apply_prior(x.maps[i], Ax.maps[i])
