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
from classes.Predictand import Predictand
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
sns.set()



class ExportVarPlot:
    """save variables and plot data"""

    def __init__(self, output_label: str, cl_config: dict):
        """
        initialize class
        :param output_label: string which names the folder for output
        :param cl_config: dictionary, where all information of logger is stored from classes/config
        """
        logging.config.dictConfig(cl_config)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Read ini-file')
        self.output_label = output_label
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

    def _create_dataset_from_var(self, variable: str, pred_t: Predictand, significance: np.ndarray):
        """
        create dataset for clusters as netcdf using xarray library
        :param variable: precursor name
        :param pred_t: object of class Predictand, where the forecast_nn and significance shall be plotted
        :param significance: 2d array of which point of the correlation are significant
        """
        self.logger.info("create dataset for variable")
        self.data_vars = {f"{pred_t.var}-{self.predictor_names}-skill": xr.DataArray(variable, dims=('lat', 'lon')),
                          f"{pred_t.var}-{self.predictor_names}-significance": xr.DataArray(significance, dims=('lat', 'lon')),
                          }
        self.lons = pred_t.dict_predict[f"{pred_t.var}"].coords["lon"].values
        self.lats = pred_t.dict_predict[f"{pred_t.var}"].coords["lat"].values
        self.lon_min, self.lon_max = min(self.lons), max(self.lons)
        self.lat_min, self.lat_max = min(self.lats), max(self.lats)
        self.ds = xr.Dataset(self.data_vars, coords={'lon': self.lons, 'lat': self.lats})

    def _save_variable(self, variable: str, pred_t: Predictand, name: str, significance: np.ndarray):
        """
        save clusters using xarray
        :param variable: precursor name
        :param pred_t: object of class Predictand, where the forecast_nn and significance shall be plotted
        :param pred_t_corr_reshape: correlation 2d array of forecast_nn and observation
        :param name: name of the precuror
        :param significance: 2d array of which point of the correlation are significant
        """
        self.logger.info("Save variable as netcdf")
        self._create_dataset_from_var(variable, pred_t, significance)
        self.ds.to_netcdf(f'{self.directory_files}/{pred_t.var}_{self.predictor_names}_skill.nc')

    def _save_skill_plot(self, variable: str, pred_t: Predictand, name: str, significance: np.ndarray, mean_skill: float):
        """
        save clusters into one plot using xarray library
        :param variable: precursor name
        :param pred_t: object of class Predictand, where the forecast_nn and significance shall be plotted
        :param pred_t_corr_reshape: correlation 2d array of forecast_nn and observation
        :param name: name of the precuror
        :param significance: 2d array of which point of the correlation are significant
        :param mean_skill: mean value of significance array
        """
        self._save_variable(variable, pred_t, name, significance)
        # n_cols = max(n, 1)
        map_proj = ccrs.PlateCarree()
        self.ds_arrays = self.ds[f"{pred_t.var}-{self.predictor_names}-skill"]
        ax = plt.axes(projection=map_proj)
        self.ds[f"{pred_t.var}-{self.predictor_names}-skill"].plot(
            ax=ax,
            transform=ccrs.PlateCarree(),  # the data's projection
            cmap=plt.cm.get_cmap('seismic', 31),
            cbar_kwargs={'shrink': 0.8, 'label':f"{self.predictor_names}-skill"},
        )
        lsize = 14
        axislsize = 10
        ax.add_feature(cfeature.BORDERS, linewidth=0.1)
        ax.coastlines()
        ax.set_extent([self.lon_min, self.lon_max, self.lat_min, (2 * self.lat_max - 90)])
        # Without this aspect attributes the maps will look chaotic and the
        # "extent" attribute above will be ignored
        # ax.set_aspect("equal")
        ax.set_aspect(aspect=self.ds.dims["lon"] / self.ds.dims["lat"])

        ax.set_title(f"{pred_t.var}-skill: {mean_skill:5.3f}",fontsize=14)
        significance_1d = np.reshape(np.array(significance), -1)
        if any(x < 0.05 for x in significance_1d):
            ax.contourf(self.lons, self.lats, significance, levels=[ 0.00, 0.05, 0.5, 0.95, 1],
                        hatches=["////", "....", None, None, None], colors='none', transform=ccrs.PlateCarree())
                    # hatches=["/////", ".....", ",,,,,", "/////", "....."], colors='none', transform=ccrs.PlateCarree())

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.1, color='gray', linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': axislsize, 'color': 'black'}
        gl.ylabel_style = {'size': axislsize, 'color': 'black'}
        gl.xlocator = mticker.FixedLocator([i for i in range(-180, 180, 30)])
        gl.ylocator = mticker.FixedLocator([i for i in range(10, 100, 20)])

        self.logger.debug(f"Save in {self.directory_plots}/{pred_t.var}_skill.pdf")
        plt.savefig(f"{self.directory_plots}/{pred_t.var}_{self.predictor_names}_skill.png")
        plt.close()

    def _save_skill_plotNN(self, variable: str, pred_t: Predictand, name: str, significance: np.ndarray, mean_skill: float,
        nr_neurons: int, opt_method: str, nr_epochs: int, nr_layers: int, lr_rate: float, nr_batch_size: int):
        """
        save clusters into one plot using xarray library
        :param variable: precursor name
        :param pred_t: object of class Predictand, where the forecast_nn and significance shall be plotted
        :param pred_t_corr_reshape: correlation 2d array of forecast_nn and observation
        :param name: name of the precuror
        :param significance: 2d array of which point of the correlation are significant
        :param mean_skill: mean value of significance array
        """
        self._save_variable(variable, pred_t, name, significance)
        # n_cols = max(n, 1)
        map_proj = ccrs.PlateCarree()
        self.ds_arrays = self.ds[f"{pred_t.var}-{self.predictor_names}-skill"]
        ax = plt.axes(projection=map_proj)
        self.ds[f"{pred_t.var}-{self.predictor_names}-skill"].plot(
            ax=ax,
            transform=ccrs.PlateCarree(),  # the data's projection
            cmap=plt.cm.get_cmap('seismic', 31),
            cbar_kwargs={'shrink': 0.8, 'label':f"{self.predictor_names}-skill"},
        )
        lsize = 14
        axislsize = 10
        ax.add_feature(cfeature.BORDERS, linewidth=0.1)
        ax.coastlines()
        ax.set_extent([self.lon_min, self.lon_max, self.lat_min, (2 * self.lat_max - 90)])
        # Without this aspect attributes the maps will look chaotic and the
        # "extent" attribute above will be ignored
        # ax.set_aspect("equal")
        ax.set_aspect(aspect=self.ds.dims["lon"] / self.ds.dims["lat"])

        ax.set_title(f"{pred_t.var}-skill: {mean_skill:5.3f}",fontsize=14)
        significance_1d = np.reshape(np.array(significance), -1)
        if any(x < 0.05 for x in significance_1d):
            ax.contourf(self.lons, self.lats, significance, levels=[ 0.00, 0.05, 0.5, 0.95, 1],
                        hatches=["////", "....", None, None, None], colors='none', transform=ccrs.PlateCarree())
                    # hatches=["/////", ".....", ",,,,,", "/////", "....."], colors='none', transform=ccrs.PlateCarree())

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.1, color='gray', linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': axislsize, 'color': 'black'}
        gl.ylabel_style = {'size': axislsize, 'color': 'black'}
        gl.xlocator = mticker.FixedLocator([i for i in range(-180, 180, 30)])
        gl.ylocator = mticker.FixedLocator([i for i in range(10, 100, 20)])
        filename = f"{pred_t.var}_skill_neurons_{nr_neurons}_{opt_method}_epochs_{nr_epochs}_layers_{nr_layers}_" \
                   f"lr_rate_{lr_rate}_batch_size_{nr_batch_size}"
        self.logger.debug(f"Save in {self.directory_plots}/{filename}.png")
        plt.savefig(f"{self.directory_plots}/{filename}.png")
        plt.close()

    def save_plot_and_time_correlation(self, list_precursors: list, pred_t: Predictand,
                                       pred_t_corr_reshape: np.ndarray, significance: np.ndarray,
                                       all_precs_names: list, mean_skill: np.ndarray):
        """
        call functions to save and plot data
        :param list_precursors: list of precursors which should be plotted
        :param pred_t: object of class Predictand, where the forecast_nn and significance shall be plotted
        :param pred_t_corr_reshape: correlation 2d array of forecast_nn and observation
        :param significance: 2d array of which point of the correlation are significant
        :param all_precs_names: all possible precursors names
        :param mean_skill: mean value of significance array
        """
        # define outputname
        if len(list_precursors) == 1:
            self.predictor_names = list_precursors[0]  # all_precs_names[list_precursors[0]]
        else:
            s = "-"
            self.predictor_names = s.join(list_precursors)
        file_path = str(len(list_precursors)) + '-precursor/'
        self.directory_plots = os.path.dirname(f"output-{self.output_label}/{pred_t.var}-{file_path}/plots/")
        self.directory_files = os.path.dirname(f"output-{self.output_label}/{pred_t.var}-{file_path}/files/")
        Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
        Path(self.directory_files).mkdir(parents=True, exist_ok=True)
        self._save_skill_plot(pred_t_corr_reshape, pred_t, pred_t.var, significance, mean_skill)

    def save_plot_and_time_correlationNN(self, list_precursors: list, pred_t: Predictand,
                                       pred_t_corr_reshape: np.ndarray, significance: np.ndarray,
                                       all_precs_names: list, mean_skill: np.ndarray,
        nr_neurons: int, opt_method: str, nr_epochs: int, nr_layers: int, lr_rate: float, nr_batch_size: int):
        """
        call functions to save and plot data
        :param list_precursors: list of precursors which should be plotted
        :param pred_t: object of class Predictand, where the forecast_nn and significance shall be plotted
        :param pred_t_corr_reshape: correlation 2d array of forecast_nn and observation
        :param significance: 2d array of which point of the correlation are significant
        :param all_precs_names: all possible precursors names
        :param mean_skill: mean value of significance array
        """
        # define outputname
        if len(list_precursors) == 1:
            self.predictor_names = list_precursors[0]  # all_precs_names[list_precursors[0]]
        else:
            s = "-"
            self.predictor_names = s.join(list_precursors)
        file_path = str(len(list_precursors)) + '-precursor/'
        self.directory_plots = os.path.dirname(f"output-{self.output_label}/{pred_t.var}-{file_path}/plots/")
        self.directory_files = os.path.dirname(f"output-{self.output_label}/{pred_t.var}-{file_path}/files/")
        Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
        Path(self.directory_files).mkdir(parents=True, exist_ok=True)
        self._save_skill_plotNN(pred_t_corr_reshape, pred_t, pred_t.var, significance, mean_skill, nr_neurons,
                                opt_method, nr_epochs, nr_layers, lr_rate, nr_batch_size)
