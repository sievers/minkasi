from .mapset import Mapset
from .polmap import HealPolMap, PolMap
from .skymap import HealMap, SkyMap, SkyMapCar, SkyMapCoarse
from .twores import SkyMapTwoRes

MapType = SkyMap | PolMap
