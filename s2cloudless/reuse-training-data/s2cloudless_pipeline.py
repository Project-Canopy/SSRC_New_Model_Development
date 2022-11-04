from cloudpathlib import CloudPath
import rasterio
from pathlib import Path
from rasterio import features as rfeatures
import geopandas as gpd
import pandas as pd
import ee
from shapely.geometry.polygon import Polygon
import json
import os
import time

from typing import List, Union
from tqdm import tqdm
# abbreviations commonly used--
# s2: sentinenel2
# sr: the kind of radar/sensor sentinenl2 uses

class S2CloudlessPipeline:
    """
    Pipeline for downloading satellite images from Google Earth Engine using the S2Cloudess cloud removal algorithm.
    Arguments:
    s2cloudless_config: Dictionary containing various values the algorithm needs. See the "Filter_old_polygons" notebook for an example.
    chips_list: List of tiles to downlod. See the "Filter_old_polygons" notebook for an example.
    """
    def __init__(self, s2cloudless_config, chips_list):
        ee.Initialize()
        self.s2cloudless_config = s2cloudless_config
        self.chips_list = chips_list
        self.band_list = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','TCI_R','TCI_G','TCI_B','AOT','WVP']
        self.NIR_DRK_THRESH = s2cloudless_config.get('NIR_DRK_THRESH',  0.15)
        self.CLOUDY_PIXEL_PERCENTAGE = s2cloudless_config.get('CLOUDY_PIXEL_PERCENTAGE', 60)
        self.CLD_PRB_THRESH = s2cloudless_config.get('CLD_PRB_THRESH',  40)
        self.BUFFER = s2cloudless_config.get('BUFFER',  50)
        self.CLD_PRJ_DIST = s2cloudless_config.get('CLD_PRJ_DIST',  2)


    def get_s2_sr_cld_col(self, aoi: ee.Geometry, start_date:str = '2020-06-01', end_date: str ='2020-07-01'):
        # Import and filter S2 SR.
        s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self.CLOUDY_PIXEL_PERCENTAGE)))

        # Import and filter s2cloudless.
        s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                            .filterBounds(aoi)
                            .filterDate(start_date, end_date))

        # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
        return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
            'primary': s2_sr_col,
            'secondary': s2_cloudless_col,
            'condition': ee.Filter.equals(**{
                'leftField': 'system:index',
                'rightField': 'system:index'
            })
        }))

    def add_cloud_bands(self, img):
        # Get s2cloudless image, subset the probability band.
        cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

        # Condition s2cloudless by the probability threshold value.
        is_cloud = cld_prb.gt(self.CLD_PRB_THRESH).rename('clouds')

        # Add the cloud probability layer and cloud mask as image bands.
        return img.addBands(ee.Image([cld_prb, is_cloud]))

    def add_shadow_bands(self, img):
        # Identify water pixels from the SCL band.
        not_water = img.select('SCL').neq(6)

        # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
        SR_BAND_SCALE = 1e4
        dark_pixels = img.select('B8').lt(self.NIR_DRK_THRESH * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

        # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
        shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

        # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
        cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, self.CLD_PRJ_DIST * 10)
                    .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
                    .select('distance')
                    .mask()
                    .rename('cloud_transform'))

        # Identify the intersection of dark pixels with cloud shadow projection.
        shadows = cld_proj.multiply(dark_pixels).rename('shadows')

        # Add dark pixels, cloud projection, and identified shadows as image bands.
        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    def add_cld_shdw_mask(self, img):
        # Add cloud component bands.
        img_cloud = self.add_cloud_bands(img)

        # Add cloud shadow component bands.
        img_cloud_shadow = self.add_shadow_bands(img_cloud)

        # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
        is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

        # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
        # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
        is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(self.BUFFER * 2 / 20)
                       .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
                       .rename('cloudmask'))

        # Add the final cloud-shadow mask to the image.
        return img_cloud_shadow.addBands(is_cld_shdw)

    def apply_cld_shdw_mask_all_bands(self, img):
        # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
        not_cld_shdw = img.select('cloudmask').Not()

        # Subset reflectance bands and update their masks, return the result.
        return img.select('B.*').updateMask(not_cld_shdw)

    # def export_to_gcs(self, s2_sr_median, AOI, polygon_id, band_list):
    #     time_stamp = "_".join(time.ctime().split(" ")[1:])
    #     time_stamp = time_stamp.replace(':', '_')
    #
    #     export = ee.batch.Export.image.toCloudStorage(
    #         image=s2_sr_median.select(band_list),
    #         description=f'{str(polygon_id)}_full_band_s2cloudless_export',
    #         scale=10,
    #         region=AOI,
    #         fileNamePrefix=f'S2_CloudFree/full_congo_s2cloudless_3/{str(polygon_id)}_{time_stamp}',
    #         bucket='project-canopy-temp-2',
    #         maxPixels=1e13
    #     )
    #     export.start()
    #
    #     return export

    def export_to_gdrive(self, s2_sr_median, AOI:ee.Geometry, polygon_id:str, start_date:str, end_date:str):
        task = ee.batch.Export.image.toCloudStorage(
            image=s2_sr_median,  # .select(band_list),
            fileNamePrefix=f'S2_CloudFree/s2cloudless_null_chips_2/{polygon_id}',
            description=f'null chip {polygon_id}',
            #folder=folder,
            bucket='project-canopy-temp-2',
            crs='EPSG:4326', # crs of _exported_ image
            maxPixels=1e13,
            region=AOI,  # .bounds().getInfo()['coordinates'],
            scale=10, #Resolution in meters per pixel. Defaults to 1000.
            #fileFormat='GeoTIFF',
        )
        task.start()
        return task

    def s2cloudless_process_download(self, start_date, end_date, dup_ids=[]):
        # start_date = f"{year}-01-01"
        # end_date = f"{year}-03-01"

        tasks = []
        count = 0

        for i in tqdm(range(len(self.chips_list))):
            poly_info = self.chips_list[i]
            polygon_id = poly_info['polygon_id']
            aoi = poly_info['aoi']
            
            if polygon_id not in dup_ids:

                s2_sr_cld_col = self.get_s2_sr_cld_col(aoi, start_date, end_date)

                s2_sr_median = (s2_sr_cld_col.map(self.add_cld_shdw_mask)
                                .map(self.apply_cld_shdw_mask_all_bands)
                                .median())

                #         if add_NDVI:

                #             s2_sr_median = add_ndvi(s2_sr_median)

                #         else:

                #             s2_sr_median = s2_sr_median.toInt16()

                #         s2_sr_median = s2_sr_median.toInt16()
                #         s2_sr_median = s2_sr_median.clip(AOI)#.reproject('EPSG:4326', None, 10)

                task = self.export_to_gdrive(s2_sr_median, aoi, polygon_id=polygon_id, start_date=start_date, end_date=end_date)
                tasks.append((i, task))

                count += 1
                if count == 500:
                    time.sleep(3600)
                    count = 0

        return tasks



