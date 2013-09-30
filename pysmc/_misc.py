"""

.. _misc:

++++++++++++++++++++++
Miscellaneous routines
++++++++++++++++++++++

"""


__all__ = ['try_to_array', 'get_var_from_particle_list']


import numpy as np


def try_to_array(data):
    """
    Try to turn the data into a numpy array.

    :returns:   If possible, a :class:`numpy.ndarray` containing the
                data. Otherwise, it just returns the data.
    :rtype:     :class:`numpy.ndarray` or ``type(data)``
    """
    try:
        return np.array(data)
    except:
        return data


def get_var_from_particle_list(particle_list, var_name, type_of_var):
    """
    Get the particles pertaining to variable ``var_name`` of type
    ``type_of_var``.

    :param var_name:    The name of the variable whose particles you want to
                        get.
    :type var_name:     str
    :param type_of_var: The type of variables you want to get. This can be
                        either 'stochastics' or 'deterministics' if you are
                        are using :mod:`pymc`. The default type is 'stochastics'.
                        However, I do not restrict its value, in case you
                        would like to define other types by extending
                        :mod:`pymc`.
    :type type_of_var:  str
    """
    data = [particle_list[i][type_of_var][var_name]
            for i in range(len(particle_list))]
    return try_to_array(data)
