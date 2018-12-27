import logging
import os

# import dask_rasterio
from read import read_raster
import fiona
import numpy as np
import rasterio
import rasterio.mask
from shapely.geometry import box, shape
from skimage import exposure
from skimage.io import imsave

from shapely.geometry import Polygon, mapping

from features import geometry_mask, geometry_window

from shapely.ops import transform
from functools import partial
import pyproj
from rasterio.windows import Window
from shapely.geometry import box, shape

from transform import IDENTITY, guard_transform  #rasterio
from enums import MergeAlg
from dtypes import validate_dtype, can_cast_dtype, get_minimum_dtype

from rasterio._features import _shapes, _sieve, _rasterize, _bounds
from rasterio.crs import CRS
from rasterio.dtypes import validate_dtype, can_cast_dtype, get_minimum_dtype
from rasterio.enums import MergeAlg
from rasterio.env import ensure_env
from rasterio.rio.helpers import coords
from rasterio.transform import Affine
from rasterio.transform import IDENTITY, guard_transform
from rasterio.windows import Window
from rasterio import warp

from tqdm import tqdm

from rasterio.plot import show
from scipy.misc import imsave

from rasterio import features
import matplotlib.pyplot as plt
%matplotlib inline


def createMask(x, threshold):
    
    y = x.copy()
    y[y<threshold] = 0.0
    
    return(y)




def reproject_shape(shape, src_crs, dst_crs):
    """Reprojects a shape from some projection to another"""
    project = partial(
        pyproj.transform,
        pyproj.Proj(init=src_crs['init']),
        pyproj.Proj(init=dst_crs['init']))
    return transform(project, shape)


def getMappingList(x):
    
    mapList = []
    for i in range(len(x)):
        mapList.append( mapping(x[i][1])  )
        
    return mapList


def getOverlappingShapes(raster,shapes, returnCoord = False):
    """
    Given a raster file and a list of shapes, returns the list of shapes that intersect with the raster file
    
    Arguments
    raster : Raster image. Rasterio object
    shapes : Polygon shape files. Fiona object
    
    """
    
    fullBox = Window(0,0,raster.width,raster.height)
    fullWindow = box(*raster.window_bounds(fullBox))
    
    matchingShapes = []
    for i in range(len(shapes)):

        t = reproject_shape(shape(shapes[i]['geometry']), shapes.crs, raster.crs)
        intersect = t.intersection(fullWindow)
        if intersect.area > 0 :
            matchingShapes.append([shapes[i], shape(shapes[i]['geometry']),t])
            
    if returnCoord == True:
        #Return only the geospatial coordinates of the transformed shape
        matchingShapes = [mapping(x[2]) for x in matchingShapes]
    
        
    return(matchingShapes)

    

def getShapeCoords(raster, shape,returnXY = False):
    """
    Given a raster and a shape ( polygon coordinates), returns the polygon coordinates in terms
    of pixel positions.
    This allows the plotting of shapes on raster images
    
    Arguments 
    raster : Image file. Rasterio object
    shape : Dict containing geospatial coordinates
    """
    
    #Check if shapes is in the proper format
    if 'coordinates' not in shape:
        shape = mapping(shape[2])
    
    
    coordList = []
    
    for i in range(len(shape['coordinates'][0])):
        point = shape['coordinates'][0][i]
        newPoint = raster.index(point[0], point[1])
        
        coordList.append(newPoint)
        
    #This step is done because the ordering of x,y axis is reversed while using index.
    #This behavior has been observed at different instances across rasterio
    coordList = [(x[1],x[0]) for x in coordList]
    
    if returnXY==True:
        x = [i[0] for i in coordList]
        y = [i[1] for i in coordList]
        return(x,y)
    else:
        return(coordList)
        
    
        
    

def plotShapeOnRaster(raster, shape, band = 1):
    """
    
    Arguments :
    raster : Raster file object
    shape : Dict containing geospatial coordinates
    band : Which band of the raster to be used
    """
    
    
    q = getShapeCoords(raster,shape, returnXY=True)
    rval = raster.read(band)
    plt.figure(figsize=(20,20))
    plt.imshow(rval,cmap='gray')
    plt.plot(q[0], q[1], color='red', alpha=0.7,
        linewidth=3, solid_capstyle='round', zorder=2)
    



def is_valid_geom(geom):
    """
    Checks to see if geometry is a valid GeoJSON geometry type or
    GeometryCollection.

    Geometries must be non-empty, and have at least x, y coordinates.

    Note: only the first coordinate is checked for validity.

    Parameters
    ----------
    geom: an object that implements the geo interface or GeoJSON-like object

    Returns
    -------
    bool: True if object is a valid GeoJSON geometry type
    """

    geom_types = {'Point', 'MultiPoint', 'LineString', 'LinearRing',
                  'MultiLineString', 'Polygon', 'MultiPolygon'}

    if 'type' not in geom:
        return False

    try:
        geom_type = geom['type']
        if geom_type not in geom_types.union({'GeometryCollection'}):
            return False

    except TypeError:
        return False

    if geom_type in geom_types:
        if 'coordinates' not in geom:
            return False

        coords = geom['coordinates']

        if geom_type == 'Point':
            # Points must have at least x, y
            return len(coords) >= 2

        if geom_type == 'MultiPoint':
            # Multi points must have at least one point with at least x, y
            return len(coords) > 0 and len(coords[0]) >= 2

        if geom_type == 'LineString':
            # Lines must have at least 2 coordinates and at least x, y for
            # a coordinate
            return len(coords) >= 2 and len(coords[0]) >= 2

        if geom_type == 'LinearRing':
            # Rings must have at least 4 coordinates and at least x, y for
            # a coordinate
            return len(coords) >= 4 and len(coords[0]) >= 2

        if geom_type == 'MultiLineString':
            # Multi lines must have at least one LineString
            return (len(coords) > 0 and len(coords[0]) >= 2 and
                    len(coords[0][0]) >= 2)

        if geom_type == 'Polygon':
            # Polygons must have at least 1 ring, with at least 4 coordinates,
            # with at least x, y for a coordinate
            return (len(coords) > 0 and len(coords[0]) >= 4 and
                    len(coords[0][0]) >= 2)

        if geom_type == 'MultiPolygon':
            # Muti polygons must have at least one Polygon
            return (len(coords) > 0 and len(coords[0]) > 0 and
                    len(coords[0][0]) >= 4 and len(coords[0][0][0]) >= 2)

    if geom_type == 'GeometryCollection':
        if 'geometries' not in geom:
            return False

        if not len(geom['geometries']) > 0:
            # While technically valid according to GeoJSON spec, an empty
            # GeometryCollection will cause issues if used in rasterio
            return False

        for g in geom['geometries']:
            if not is_valid_geom(g):
                return False  # short-circuit and fail early

    return True



def rasterize(
        shapes,
        out_shape=None,
        fill=0,
        out=None,
        transform=IDENTITY,
        all_touched=False,
        merge_alg=MergeAlg.replace,
        default_value=1,
        dtype=None):
    
    valid_dtypes = (
        'int16', 'int32', 'uint8', 'uint16', 'uint32', 'float32', 'float64'
    )

    def format_invalid_dtype(param):
        return '{0} dtype must be one of: {1}'.format(
            param, ', '.join(valid_dtypes)
        )

    def format_cast_error(param, dtype):
        return '{0} cannot be cast to specified dtype: {1}'.format(param, dtype)

    if fill != 0:
        fill_array = np.array([fill])
        if not validate_dtype(fill_array, valid_dtypes):
            raise ValueError(format_invalid_dtype('fill'))

        if dtype is not None and not can_cast_dtype(fill_array, dtype):
            raise ValueError(format_cast_error('fill', dtype))

    if default_value != 1:
        default_value_array = np.array([default_value])
        if not validate_dtype(default_value_array, valid_dtypes):
            raise ValueError(format_invalid_dtype('default_value'))

        if dtype is not None and not can_cast_dtype(default_value_array, dtype):
            raise ValueError(format_cast_error('default_vaue', dtype))

    if dtype is not None and np.dtype(dtype).name not in valid_dtypes:
        raise ValueError(format_invalid_dtype('dtype'))

    valid_shapes = []
    shape_values = []
    for index, item in enumerate(shapes):
        if isinstance(item, (tuple, list)):
            geom, value = item
        else:
            geom = item
            value = default_value
        geom = getattr(geom, '__geo_interface__', None) or geom

        # geom must be a valid GeoJSON geometry type and non-empty
        if not is_valid_geom(geom):
            raise ValueError(
                'Invalid geometry object at index {0}'.format(index)
            )

        if geom['type'] == 'GeometryCollection':
            # GeometryCollections need to be handled as individual parts to
            # avoid holes in output:
            # https://github.com/mapbox/rasterio/issues/1253.
            # Only 1-level deep since GeoJSON spec discourages nested
            # GeometryCollections
            for part in geom['geometries']:
                valid_shapes.append((part, value))

        else:
            valid_shapes.append((geom, value))

        shape_values.append(value)

    if not valid_shapes:
        raise ValueError('No valid geometry objects found for rasterize')

    shape_values = np.array(shape_values)

    if not validate_dtype(shape_values, valid_dtypes):
        raise ValueError(format_invalid_dtype('shape values'))

    if dtype is None:
        dtype = get_minimum_dtype(np.append(shape_values, fill))

    elif not can_cast_dtype(shape_values, dtype):
        raise ValueError(format_cast_error('shape values', dtype))

    if out is not None:
        if np.dtype(out.dtype).name not in valid_dtypes:
            raise ValueError(format_invalid_dtype('out'))

        if not can_cast_dtype(shape_values, out.dtype):
            raise ValueError(format_cast_error('shape values', out.dtype.name))

    elif out_shape is not None:

        if len(out_shape) != 2:
            raise ValueError('Invalid out_shape, must be 2D')

        out = np.empty(out_shape, dtype=dtype)
        out.fill(fill)

    else:
        raise ValueError('Either an out_shape or image must be provided')

    if min(out.shape) == 0:
        raise ValueError("width and height must be > 0")

    transform = guard_transform(transform)
    _rasterize(valid_shapes, out, transform, all_touched, merge_alg)
    return out



def getBinaryMask(dataset, shapes, transformed = True, window = None):
    """
    Returns a binary mask with respect to the shape polygons overlayed on the raster file
    
    Arguments : 
    dataset : Raster file object
    shapes : List of Dictionaries, with each dict containing geospatial coordinates of polygon vertices
    window : A window object specifying the specific window in the raster to be concentrating on. The fina;
             binary mask is cropped based on the given window

    """
    
    if type(shapes) is list :
        
        #Check for proper format of shape file
        if 'geometry' in shapes[0]:
            shapes = [mapping(x[2]) for x in shapes]
     
    elif type(shapes) is dict:
        
        if 'geometry' in shapes:
            shapes = [mapping(shapes[2])]
        elif 'coordinates' in shapes:
            shapes = [shapes]
    else:
        print("Invalid shapes provided")
        return
    
    
    mask = rasterize(shapes,out_shape=(int(dataset.height), int(dataset.width)),
        transform=dataset.transform,
        all_touched=False,fill=1,default_value=0)
    
    if window!=None:
        windowed_mask = mask[window.row_off:(window.row_off+window.height) , window.col_off:(window.col_off+window.width)]
        return(windowed_mask)
    else:
        return(mask)
        
    
    
    
    return(mask)


def getBoundingBox(coordList):
    """
    coordList : List of shape coordinates in terms of raster image pixels
    """
    
    minX,maxX = 9999.0,-9999.0
    minY,maxY = 9999.0,-9999.0
    
    for i in range(len(coordList)):
        minX = min(minX,coordList[i][0])
        maxX = max(maxX,coordList[i][0])
        
        minY = min(minY,coordList[i][1])
        maxY = max(maxY,coordList[i][1])
        
    return (minX,maxX,minY,maxY)
        
    
    

def getOverlappingShapes(raster,shapes, returnCoord = False,window=None):
    """
    Given a raster file and a list of shapes, returns the list of shapes that intersect with the raster file
    
    Arguments
    raster : Raster image. Rasterio object
    shapes : Polygon shape files. Fiona object
    
    """
    
    if window!=None:
        fullBox = window
    else:
        fullBox = Window(0,0,raster.width,raster.height)
    
    fullWindow = box(*raster.window_bounds(fullBox))
    
    matchingShapes = []
    for i in range(len(shapes)):

        t = reproject_shape(shape(shapes[i]['geometry']), shapes.crs, raster.crs)
        intersect = t.intersection(fullWindow)
        if intersect.area > 0 :
            matchingShapes.append([shapes[i], shape(shapes[i]['geometry']),t])
            
    if returnCoord == True:
        #Return only the geospatial coordinates of the transformed shape
        matchingShapes = [mapping(x[2]) for x in matchingShapes]
    
        
    return(matchingShapes)


# Test this for windows with two shapes

def getIntersectionArea(raster, window, shapeDict, ratio = True):
    """
    Given a Window based on a raster and a shape object, returns the ratio/total area of overlap between
    the shape and the window
    
    Arguments : 
    raster : Raster object
    window : Rasterio Window object
    shapeDict : Dict containing geospatial coordinates
    ratio : If true, returns the ratio of intersection area to full window area, else returns intersection area
    
    """
    
    window_box = box(*raster.window_bounds(window))
    
    
    if(type(shapeDict)==list):
        shp = shape(shapeDict[0])
        
        for i in range(1,len(shapeDict)):
            shp = shp.union(shape(shapeDict[i]))            
    else:        
        shp = shape(shapeDict)
    
    intersection = shp.intersection(window_box)
    
    boxArea = window_box.area
    interArea = intersection.area
    
    if ratio == True:
        return ( interArea / float(boxArea) )
    else:
        return interArea
    


def generateWindowsWithMasks(raster, vector, output_dir, window_width=1000, window_height = 1000, step_size = 100):
    
    name = raster.name.split("/")[-1][:-4]
    shapes = getOverlappingShapes(raster,vector,returnCoord=True)
    if len(shapes)==0:
        print("No overlapping shapes")
        return
    
    mask = getBinaryMask(raster, shapes)
    
    if (os.path.exists(os.path.join(output_dir,name))==False) :
        os.mkdir(os.path.join(output_dir,name))
            
    os.chdir(os.path.join(output_dir,name))
            
    count = 0

    for i in tqdm(range(0,raster.width - window_width, step_size)):
        for j in range(0,raster.height - window_height, step_size):

            w = Window(i,j,window_width, window_height)
            wbox = box(*raster.window_bounds(w))

            #Reads all 3 bands and concatenates them
            img = np.dstack([raster.read(k,window=w) for k in range(1,4)])

            window_mask = mask[w.row_off:(w.row_off+w.height) , w.col_off:(w.col_off+w.width)]
            window_mask = window_mask*255  #Convert 0-1 array to 0-255

            shapes = getOverlappingShapes(raster,vector,returnCoord=True,window=w)

            if(len(shapes)==0):
                continue

            intersectionArea = getIntersectionArea(raster=raster,shapeDict=shapes,window=w,ratio=True)
        
            
            if ( intersectionArea>0.05):

                count += 1
#                 print("{} {} Area : {}".format(i,j,intersectionArea))

                index = str(i)+"_"+str(j)
                imsave("./"+ index+"img.jpg",img)
                imsave("./"+ index+"mask.jpg",window_mask)
    
    


def generateMasksForShape(raster,shape,vector,window_width= 1000, window_height = 1000, step_size=100):
    
    mask = getBinaryMask(raster, shape)
    minX,maxX,minY,maxY = getBoundingBox( getShapeCoords(raster,shape) )
    
    for i in tqdm(range(int(minX), int(maxX), step_size)):
        for j in range(int(minY),int(maxY),step_size):
            
            w = Window(i,j,window_width, window_height)
            wbox = box(*raster.window_bounds(w))

            #Reads all 3 bands and concatenates them
            img = np.dstack([raster.read(k,window=w) for k in range(1,4)])

            window_mask = mask[w.row_off:(w.row_off+w.height) , w.col_off:(w.col_off+w.width)]
            window_mask = window_mask*255  #Convert 0-1 array to 0-255

            shapes = getOverlappingShapes(raster,vector,returnCoord=True,window=w)

            if(len(shapes)==0):
                continue

            intersectionArea = getIntersectionArea(raster=raster,shapeDict=shapes,window=w,ratio=True)
        
            
            if ( intersectionArea>0.00):

#                 print("{} {} Area : {}".format(i,j,intersectionArea))

                index = str(i)+"_"+str(j)
                imsave("./"+ index+"img.jpg",img)
                imsave("./"+ index+"mask.jpg",window_mask)
            
    