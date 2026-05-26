# coding=utf-8
from typing import Generic, TypeVar, Optional

from epstudiosdk.exception import error_code

from epstudiosdk.bean.ResultData import ResultData
from epstudiosdk.exception.exceptions import ClientException
from epstudiosdk.utils.param import version_to_int

T = TypeVar('T')


class EpRequest(Generic[T]):
    """
    abstract class of request
    """
    def __init__(self):
        self._method = None
        self._action_name = None
        self._params = {}
        self._header = {}
        self._body_params = {}
        self._timeout = None
        self._res = None
        self._min_version: str = '0'
        self._max_version: Optional[str] = None

    def add_query_param(self, k, v):
        self._params[k] = v

    def set_query_params(self, params):
        self._params = params

    def get_query_params(self):
        return self._params

    def add_body_params(self, k, v):
        self._body_params[k] = v

    def get_body_params(self):
        return self._body_params

    def set_body_params(self, body_params):
        self._body_params = body_params

    def get_headers(self):
        return self._header

    def set_headers(self, headers):
        self._header = headers

    def add_header(self, k, v):
        self._header[k] = v

    def get_method(self):
        return self._method

    def get_action_name(self):
        return self._action_name

    def get_timeout(self):
        timeout = self._timeout
        if isinstance(timeout, tuple):
            if len(timeout) == 0:
                timeout = None
            elif len(timeout) == 1:
                timeout = (timeout[0], timeout[0])
        return timeout

    def get_info(self):
        return "Request info. Method: %s. Action-name: %s. Request-param: %s. Request-body: %s" \
               % (self._method, self._action_name, self._params, self._body_params)

    def to_result_data(self) -> ResultData[T]:
        return ResultData(result=self._res['result'],
                          code=self._res['code'],
                          desc=self._res['desc'],
                          data=self._res['data'])

    def check_version(self, version: int = None):
        int_min_version = version_to_int(self._min_version)
        int_max_version = version_to_int(self._max_version)
        if version is not None:
            if self._min_version is not None and int_min_version is not None and version < int_min_version:
                raise ClientException(error_code.SDK_VERSION_ERROR, "The server version not support this interface.")
            if self._max_version is not None and int_max_version is not None and version > int_max_version:
                raise ClientException(error_code.SDK_VERSION_ERROR, "The server version not support this interface.")
