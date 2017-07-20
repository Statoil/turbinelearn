from .turbine_file import (enum_files, load_single_file, load_data,
                           normalize_column, preprocess_data, split_data_set)
from .turbine_file import FEATURES, TARGET, LIMITS

from .turbine_regression import file_cross_val, individual_cross_validation
from .turbine_regression import train_and_evaluate_single_file
from .turbine_regression import regression, pca
from .turbine_regression import compute_learning_progress
from .turbine_regression import generate_polynomial
from .turbine_regression import detect_outliers

from .dual_linear_regression import DualLinearModel

TURBINELEARN_VERSION = '0.2.1'
__version__ = TURBINELEARN_VERSION

__copyright__ = 'Copyright 2017, Statoil ASA'
__license__ = 'GNU General Public License, version 3 or any later version'

__author__ = 'Statoil ASA, Software Innovation Bergen'
__credits__ = __author__
__maintainer__ = __author__
__email__ = __author__
__status__ = 'Prototype'


__ALL__  = ['load_data']
__ALL__ += ['pca', 'regression', 'file_cross_val', 'individual_cross_validation']
__ALL__ += ['compute_learning_progress']
__ALL__ += ['detect_outliers', 'generate_polynomial']
__ALL__ += ['train_and_evaluate_single_file']
__ALL__ += ['DualLinearModel']
__ALL__ += ['FEATURES', 'TARGET', 'LIMITS']
__ALL__ += ['TURBINELEARN_VERSION']
