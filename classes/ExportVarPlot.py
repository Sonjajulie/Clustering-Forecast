import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
# import numpy as np
# from netCDF4 import Dataset
import os  # needed for loop over files
import logging
# matplotlib.rcParams['backend'] = "Qt4Agg"
# from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

sns.set()

logger = logging.getLogger(__name__)


class ExportVarPlot:
    """save variables and plot data"""

    def __init__(self):
        """ initialize class"""
        self.cmap1 = plt.cm.get_cmap('seismic', 51)
        self.orig_cmap = plt.cm.get_cmap('seismic', 51)
        self.lons = None
        self.lats = None
        self.lat = None
        self.lon = None
        self.latitudes = None
        self.longitudes = None
        self.tmpForecastSave = None
        self.lon_grid, self.lat_grid = None, None
        self.save_string = None
        self.file_out = None
        self.eq_map = None
        self.title_string = None
        self.cbar = None
        self.directory_plots = None
        self.directory_files = None

    def _create_dataset_from_var(self, variable, pred_t, significance):
        """ create dataset for clusters as netcdf using xarray library"""
        logger.info("create dataset for variable")
        self.data_vars = {f"{pred_t.var}-skill": xr.DataArray(variable, dims=('lat', 'lon')),
                          f"{pred_t.var}-significance": xr.DataArray(significance, dims=('lat', 'lon')),
                          }
        self.lons = pred_t.dict_predict[f"{pred_t.var}"].coords["lon"].values
        self.lats = pred_t.dict_predict[f"{pred_t.var}"].coords["lat"].values
        self.lon_min, self.lon_max = min(self.lons), max(self.lons)
        self.lat_min, self.lat_max = min(self.lats), max(self.lats)
        self.ds = xr.Dataset(self.data_vars, coords={'lon': self.lons, 'lat': self.lats})

    def _save_variable(self, variable, pred_t, name, significance):
        """ save clusters using xarray"""
        logger.info("Save variable as netcdf")
        self._create_dataset_from_var(variable, pred_t, significance)
        self.ds.to_netcdf(f'{self.directory_files}/variable_{name}.nc')

    def _save_skill_plot(self, variable, pred_t, name, significance):
        """ save clusters into one plot using xarray library"""
        self._save_variable(variable, pred_t, name, significance)
        # n_cols = max(n, 1)
        map_proj = ccrs.PlateCarree()
        self.ds_arrays = self.ds[f"{pred_t.var}-skill"]
        ax = plt.axes(projection=map_proj)
        self.ds[f"{pred_t.var}-skill"].plot(
            ax=ax,
            transform=ccrs.PlateCarree(),  # the data's projection
            cmap=plt.cm.get_cmap('seismic', 51),
            cbar_kwargs={'shrink': 0.8},
        )

        ax.add_feature(cfeature.BORDERS, linewidth=0.1)
        ax.coastlines()
        ax.set_extent([self.lon_min, self.lon_max, self.lat_min, (2 * self.lat_max - 90)])
        # Without this aspect attributes the maps will look chaotic and the
        # "extent" attribute above will be ignored
        # ax.set_aspect("equal")
        ax.set_aspect(aspect=self.ds.dims["lon"] / self.ds.dims["lat"])
        ax.set_title(f"{name}", fontsize=10)
        ax.contourf(self.lons, self.lats, significance, levels=[ 0.,0.05, 0.5, 0.95, 1],
                    hatches=["/////", ".....", None, None, None], colors='none', transform=ccrs.PlateCarree())
                    # hatches=["/////", ".....", ",,,,,", "/////", "....."], colors='none', transform=ccrs.PlateCarree())
        logger.debug(f"Save in {self.directory_plots}/{pred_t.var}_skill.pdf")
        plt.savefig(f"{self.directory_plots}/{pred_t.var}_skill.pdf")
        plt.close()

    def save_plot_and_time_correlation(self, list_precursors, pred_t, pred_t_corr_reshape, significance,
                                       all_precs_names):
        """ call functions to save and plot data"""
        # define outputname
        if len(list_precursors) == 1:
            self.save_string = all_precs_names[list_precursors[0]]
        else:
            s = "-"
            self.save_string = s.join(list_precursors)
        file_path = str(len(list_precursors)) + '-precursor/'
        self.directory_plots = os.path.dirname("output/" + pred_t.var + "-" + file_path + "/plots/")
        self.directory_files = os.path.dirname("output/" + pred_t.var + "-" + file_path + "/files/")
        Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
        Path(self.directory_files).mkdir(parents=True, exist_ok=True)
        self._save_skill_plot(pred_t_corr_reshape, pred_t, pred_t.var, significance)
