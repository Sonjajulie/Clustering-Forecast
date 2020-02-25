import argparse
import os


class ClusteringParser:
    """ For parsing variables in command line"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='For the Calculation of clusters, composites and forecast_nn algorithm, '
                        'inifile, variables and length of files are needed to be specified '
                        ' via command line')
        self.parser.add_argument('-ini', '--inifile', required=True,
                                 help='path where inifile is stored')
        self.parser.add_argument('-pred', '--predictand', required=True,
                                 help='variable for which cluster analysis should be executed')
        self.parser.add_argument('-n', '--numbers', nargs='?', default=-1, type=int,
                                 help='how many data points per year should be processed. If it is -1 all '
                                      'datapoints will be processed.')
        self.parser.add_argument('-p', '--percentage', type=float, choices=range(0, 100),
                                 help='percentage for bootstrap method. For which percentage the bootstrap '
                                      'method is significant?')
        self.parser.add_argument('-log', '--logfile', type=str, help='log file to save output of program')
        self.parser.add_argument('-o', '--outputlabel', type=str, required=True,
                                 help='The name of the output folder. If it is called standardized, the variable input '
                                      'will be standardized. Note that the folder will be called output-{outputlabel}')
        self.parser.add_argument('-outp', '--outputpath', type=str, required=False, default=os.getcwd(),
                                 help='The name of the output path before output, necessary for year-plots, where you need a lot of storage. If not specified path of progarm is used')

        self.parser.add_argument('-range', '--datarange', type=int, nargs='+', required=False, default=[0,-1],
                                 help='What data points should be used for testing and training data, used in function '
                                      'train_test_split_pred')

        self.args = self.parser.parse_args()
        self.arguments = vars(self.args)
