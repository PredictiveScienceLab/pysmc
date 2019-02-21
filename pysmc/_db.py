"""

.. _db:

++++++++
DataBase
++++++++

The class :class:`pysmc.DataBase` implements a simple database for dumping
SMC steps.
"""


__all__ = ['DataBase', 'SerialDataBase']


import pickle
import os
import warnings


class DataBase(object):

    """
    A data base storing the evolution of SMC particles as gamma changes.

    :param gamma_name:      The name we are using for gamma.
    :type gamma_name:       str
    :param filename:        The desired filename for the output. If ``None``
                            then everything is stored on the ram.
    :type filename:         str
    """

    # The gammas visited so far (list)
    _gammas = None

    # The name we used for gamma (str)
    _gamma_name = None

    # The particle approximations associated with each gamma (list)
    _particle_approximations = None

    # The parameters of each step method
    _step_method_params = None

    # The filename you have selected for dumping the data (str)
    _filename = None

    # The file handler
    _fd = None

    # The Pickler
    _pickler = None

    # Last commited particle approximation
    _last_commited = None

    @property
    def gammas(self):
        """
        The list of gammas we have visited.

        :getter:    Get the list of gammas we have visited.
        :type:      list
        """
        return self._gammas

    @property
    def gamma_name(self):
        """
        The name we used for gamma.

        :getter:    Get the name we used for gamma.
        :type:      str
        """
        return self._gamma_name

    @property
    def particle_approximations(self):
        """
        The particle approximations associated with each gamma.

        :getter:    Get the particle approximations associated with each gamma.
        :type:      list
        """
        return self._particle_approximations

    @property
    def step_method_params(self):
        """
        The adaptive scale factors of each step method.

        :getter:    Get the adaptive scale factors associated with each
                    MCMC step method.
        :type:      list
        """
        return self._step_method_params

    @property
    def num_gammas(self):
        """
        The number of gammas added to the database.

        :getter:    Get the number of gammas added to the data base.
        :type:      int
        """
        return len(self._gammas)

    @property
    def filename(self):
        """
        The filename you have selected for dumping the data.

        :getter:    Get the filename you have selected for dumping the data.
        :type:      str
        """
        return self._filename

    @property
    def write_to_disk(self):
        """
        ``True`` if the class writes data to disk, ``False`` otherwise.
        """
        return not self.filename is None

    @property
    def gamma(self):
        """
        The current gamma of the database.

        :getter:    Get the current gamma of the database.
        :type:      unknown
        """
        if self.num_gammas == 0:
            raise RuntimeError(
                    'The db is empty!')
        return self.gammas[-1]

    @property
    def particle_approximation(self):
        """
        The current particle approximation of the database.

        :getter:    Get the current particle approximation of the database.
        :type:      unknown
        """
        return self._particle_approximations[-1]

    @property
    def step_method_param(self):
        return self._step_method_params[-1]

    def __init__(self, gamma_name=None, filename=None):
        """
        Initialize the object.

        See doc string for parameters.
        """
        if gamma_name is not None:
            self._initialize(gamma_name, filename)

    def _initialize(self, gamma_name, filename):
        """
        Initialize the object given a name for gamma and a filename.
        """
        if not isinstance(gamma_name, str):
            raise TypeError(
                'The name of the \'gamma\' parameter must be a str object.')
        self._gamma_name = gamma_name
        if filename is not None:
            if not isinstance(filename, str):
                raise TypeError(
                    'The filename has to be a str object.')
            filename = os.path.abspath(filename)
            if os.path.exists(filename):
                warnings.warn(
            'Filename already exists. I will overwrite it!\n'
            + ' Use pysmc.Database.load(filename) if your intentions are to'
            + ' append the file with new data!')
        self._filename = filename
        if self.write_to_disk:
            with open(self.filename, 'wb') as fd:
                pickle.dump(gamma_name, fd,
                            protocol=pickle.HIGHEST_PROTOCOL)
        self._gammas = []
        self._particle_approximations = []
        self._step_method_params = []
        self._last_commited = 0

    def add(self, gamma, particle_approximation,
            step_method_params):
        """
        Add the ``particle_approximation`` corresponding to ``gamma`` to the
        database.

        :param gamma:                   The gamma parameter.
        :type gamma:                    any
        :param particle_approximation:  particle_approximation
        :type particle_approximation:   any
        """
        self._gammas.append(gamma)
        self._particle_approximations.append(particle_approximation)
        self._step_method_params.append(step_method_params)

    def _dump_part_of_list(self, idx, values, fd):
        """
        Dump part of a list to fd.
        """
        for i in range(len(idx)):
            pickle.dump(values[idx[i]], fd,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def commit(self):
        """
        Commit everything we have so far to the database.
        """
        if self.write_to_disk:
            idx = list(range(self._last_commited, self.num_gammas))
            if len(idx) == 0:
                return
            with open(self.filename, 'ab') as fd:
                self._dump_part_of_list(idx, self.gammas, fd)
                self._dump_part_of_list(idx, self.particle_approximations, fd)
                self._dump_part_of_list(idx, self.step_method_params, fd)
            self._last_commited += len(idx)

    @staticmethod
    def load(filename):
        """
        This is a static method. It loads a database from ``filename``.
        """
        filename = os.path.abspath(filename)
        with open(filename, 'rb') as fd:
            try:
                gamma_name = pickle.load(fd)
            except:
                raise RuntimeError('File %s: Not a valid database!' % filename)
            gammas = []
            particle_approximations = []
            step_method_params = []
            while True:
                try:
                    gammas.append(pickle.load(fd))
                    particle_approximations.append(pickle.load(fd))
                    step_method_params.append(pickle.load(fd))
                except EOFError:
                    break
            last_commited = len(gammas)
        db = DataBase()
        db._gamma_name = gamma_name
        db._gammas = gammas
        db._particle_approximations = particle_approximations
        db._step_method_params = step_method_params
        db._last_commited = last_commited
        db._filename = filename
        return db


class SerialDataBase(object):

    """
    Just like DataBase but stores in memory only the latest particle
    approximation.

    :param gamma_name:      The name we are using for gamma.
    :type gamma_name:       str
    :param filename:        The desired filename for the output. If ``None``
                            then everything is stored on the ram.
    :type filename:         str
    """

    # The last gamma
    _gamma = None

    # The name we used for gamma (str)
    _gamma_name = None

    # The particle approximations associated with gamma
    _particle_approximation = None

    # The parameters of each step method
    _step_method_param = None

    # The filename you have selected for dumping the data (str)
    _filename = None

    # The file handler
    _fd = None

    # The Pickler
    _pickler = None

    # Last commited particle approximation
    _last_commited = None

    # The total number of gammas
    _num_gammas = None

    @property
    def gamma_name(self):
        """
        The name we used for gamma.

        :getter:    Get the name we used for gamma.
        :type:      str
        """
        return self._gamma_name

    @property
    def num_gammas(self):
        """
        The number of gammas added to the database.

        :getter:    Get the number of gammas added to the data base.
        :type:      int
        """
        return self._num_gammas

    @property
    def filename(self):
        """
        The filename you have selected for dumping the data.

        :getter:    Get the filename you have selected for dumping the data.
        :type:      str
        """
        return self._filename

    @property
    def gamma(self):
        """
        The current gamma of the database.

        :getter:    Get the current gamma of the database.
        :type:      unknown
        """
        return self._gamma

    @property
    def particle_approximation(self):
        """
        The current particle approximation of the database.

        :getter:    Get the current particle approximation of the database.
        :type:      unknown
        """
        return self._particle_approximation

    @property
    def step_method_param(self):
        return self._step_method_param

    def __init__(self, gamma_name=None, filename=None):
        """
        Initialize the object.

        See doc string for parameters.
        """
        if gamma_name is not None:
            self._initialize(gamma_name, filename)
        self._num_gammas = 0

    def _initialize(self, gamma_name, filename):
        """
        Initialize the object given a name for gamma and a filename.
        """
        if not isinstance(gamma_name, str):
            raise TypeError(
                'The name of the \'gamma\' parameter must be a str object.')
        self._gamma_name = gamma_name
        if filename is not None:
            if not isinstance(filename, str):
                raise TypeError(
                    'The filename has to be a str object.')
            filename = os.path.abspath(filename)
            if os.path.exists(filename):
                warnings.warn(
            'Filename already exists. I will overwrite it!\n'
            + ' Use pysmc.Database.load(filename) if your intentions are to'
            + ' append the file with new data!')
        self._filename = filename
        with open(self.filename, 'wb') as fd:
            pickle.dump(gamma_name, fd, protocol=pickle.HIGHEST_PROTOCOL)

    def add(self, gamma, particle_approximation,
            step_method_params):
        """
        Add the ``particle_approximation`` corresponding to ``gamma`` to the
        database.

        :param gamma:                   The gamma parameter.
        :type gamma:                    any
        :param particle_approximation:  particle_approximation
        :type particle_approximation:   any
        """
        self._gamma = gamma
        self._particle_approximations = particle_approximation
        self._step_method_param = step_method_params
        self._num_gammas += 1

    def _dump_part_of_list(self, idx, values, fd):
        """
        Dump part of a list to fd.
        """
        for i in range(len(idx)):
            pickle.dump(values[idx[i]], fd,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def commit(self):
        """
        Commit everything we have so far to the database.
        """
        with open(self.filename, 'ab') as fd:
            pickle.dump(self.gamma, fd, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.particle_approximation, fd,
                        protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.step_method_param, fd,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename, go_to_last=True):
        """
        This is a static method. It loads a database from ``filename``.
        """
        filename = os.path.abspath(filename)
        fd = open(filename, 'rb')
        try:
            gamma_name = pickle.load(fd)
        except:
            raise RuntimeError('File %s: Not a valid database!' % filename)
        gamma = pickle.load(fd)
        particle_approximation = pickle.load(fd)
        step_method_param = pickle.load(fd)
        db = SerialDataBase()
        db._fd = fd
        if go_to_last:
            db.go_to_last()
        return db


    def __next__(self):
        """
        Read the next state from the file opened file.
        """
        if self._fd is None:
            return False
        try:
            gamma = pickle.load(self._fd)
            particle_approximation = pickle.load(self._fd)
            step_method_param = pickle.load(self._fd)
        except EOFError:
            self._fd.close()
            self._fd = None
            return False
        if self.num_gammas >= 1:
            del self._gamma
            del self._particle_approximation
            del self._step_method_param
        self._gamma = gamma
        self._particle_approximation = particle_approximation
        self._step_method_param = step_method_param
        self._num_gammas += 1
        return True

    def go_to_last(self):
        """
        Go through all the values of the database until you reach the last one.
        """
        while next(self):
            pass
