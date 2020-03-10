
class Config:
    def __init__(self, logfile: str):
        """
        configuration class for logger
        :param logfile: name of the logfile, where log output should be saved
        """
        self.config_dict = {
            'disable_existing_loggers': False,
            'version': 1,
            'formatters': {
                'formatter_short': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
                'formatter_exact': {
                    'format': '%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s'
                },
            },
            'handlers': {
                'console': {
                    'level': 'DEBUG',
                    'formatter': 'formatter_short',
                    'class': 'logging.StreamHandler',
                },
                'file': {
                    'level': 'DEBUG',
                    'formatter': 'formatter_exact',
                    'class': 'logging.FileHandler',
                    'filename': f'{logfile}',
                    'mode': 'a',
                },
            },
            'loggers': {
                '': { # root logger
                    'handlers': ['console'],
                    'propagate': False
                },
                '__main__': {  # if __name__ == '__main__'
                    'handlers': ['console','file'],
                    'level': 'ERROR', # at what level shall call begin (still different for the handler)
                    'propagate': False
                },
                'classes.Clusters': {  # if __name__ == 'classes.Clusters'
                    'handlers': ['console','file'],
                    'level': 'ERROR', # at what level shall call begin (still different for the handler)
                    'propagate': False
                },

                'classes.Composites': {  # if __name__ == 'classes.Composites'
                    'handlers': ['console','file'],
                    'level': 'ERROR', # at what level shall call begin (still different for the handler)
                    'propagate': False
                },

                'classes.Precursors': {  # if __name__ == 'classes.Precursors'
                    'handlers': ['console', 'file'],
                    'level': 'INFO',  # at what level shall call begin (still different for the handler)
                    'propagate': False
                },

                'classes.Predictand': {  # if __name__ == 'classes.Predictand'
                    'handlers': ['console', 'file'],
                    'level': 'INFO',  # at what level shall call begin (still different for the handler)
                    'propagate': False
                },

                'classes.Forecast': {  # if __name__ == 'classes.Forecast'
                    'handlers': ['console', 'file'],
                    'level': 'INFO',  # at what level shall call begin (still different for the handler)
                    'propagate': False
                },
            },
        }
