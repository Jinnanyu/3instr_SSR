# coding=utf-8
from typing import Optional, TypeVar

import requests
import json
import sys


from epstudiosdk.request.appconfig.SystemConfigQueryRequest import SystemConfigQueryRequest
from epstudiosdk.utils import ConfigUtil, _logging
from epstudiosdk.utils.param import check_result, version_to_int
from epstudiosdk.utils.object import unicode_convert
from epstudiosdk.exception.exceptions import ClientException
from epstudiosdk.request.user.UserLoginRequest import UserLoginRequest
from epstudiosdk.request.guineapig.GuineaPigGetCurrentRequest import GuineaPigGetCurrentRequest
from epstudiosdk.request.guineapig.GuineaPigSetCurrentRequest import GuineaPigSetCurrentRequest
from epstudiosdk.exception import error_code, error_msg
from epstudiosdk.baserequest import EpRequest
from epstudiosdk.websocketclient import EpWebSocketClient

T = TypeVar('T')


class EpClient:
    """
    接口请求工具类

     websocket_client: Optional[EpWebSocketClient] 用于接收实时数据推送的客户端，传入的话会默认开启
     enable_trace: bool = False,
     init_user_status: bool = False, 是否自动登录
     login_id: str = None, 账号
     password: str = None, 密码
     guinea_pig_id: str = None 当前患者
     server_index: int = None 指定EPStudio服务地址索引，默认0，对应config.ini中host和port的值是否用;分割
    """

    def __init__(self,
                 websocket_client: Optional[EpWebSocketClient] = None,
                 enable_trace: bool = False,
                 init_user_status: bool = False,
                 login_id: str = None,
                 password: str = None,
                 guinea_pig_id: str = None,
                 server_index: int = None,
                 ):

        if enable_trace is None:
            enable_trace = False
        _logging.enableTrace(enable_trace)
        # read from configuration
        self._server_index = 0 if server_index is None or server_index < 0 else server_index
        self._parse_host_port()
        # --- 手动覆盖 host 和 port 为实际值 ---
        self._host = 'http://10.10.10.82'   # 注意保留 http://
        self._port = '8080'
        # init session
        self._session = requests.session()
        self._session.headers.update({"accept": "*/*", "Content-Type": "application/json;charset=utf-8"})

        # get the server version
        self.server_version = self._parse_server_version()

        self.init_user_status = init_user_status
        self._login_id = None
        self._password = None
        self._guinea_pig_id = None
        if self.init_user_status:
            self.init_user(login_id, password, guinea_pig_id)

        # start client
        if websocket_client:
            websocket_client.start()

    def _parse_host_port(self):
        hosts = ConfigUtil().get_data('host').split(";")
        ports = ConfigUtil().get_data('port').split(";")
        host_length = len(hosts)
        port_length = len(ports)
        if host_length != port_length:
            raise ClientException(error_code.SDK_INVALID_PARAMS,
                                  "The host and port in config.ini values split by ';' length not same.")
        if self._server_index >= host_length:
            raise ClientException(error_code.SDK_INVALID_PARAMS,
                                  "server_index param out of host and port range.")
        self._host = hosts[self._server_index].strip()
        self._port = ports[self._server_index].strip()
        if len(self._host) == 0 or len(self._port) == 0:
            raise ClientException(error_code.SDK_INVALID_PARAMS,
                                  f"host or port at index[{self._server_index}] is empty.")

    def do_action(self, request: EpRequest):
        """
        发送接口请求
        :param request: 每个接口的参数封装
        :return: 返回字符串
        """
        _logging.log.debug(request.get_info())
        request.check_version(self.server_version)
        status, headers, body, exception = self._implementation_of_do_action(request)
        _logging.log.debug('Response received. Method:%s Response-body: %s' %
                           (request.get_method(), body))
        return body

    def do_action_json(self, request: EpRequest):
        """
        发送接口请求
        :param request: 每个接口的参数封装
        :return: 返回json数据
        """
        body = self.do_action(request)
        if body is not None:
            if sys.version < '3':
                body = unicode_convert(json.loads(body))
            else:
                body = json.loads(body)
        return body

    def do_action_bean(self, request: T) -> T:
        """
        发送接口请求
        :param request: 每个接口的参数封装
        :return: 返回数据对象 ResultData或其子类
        """
        body = self.do_action_json(request)
        request._res = body
        return request

    def _implementation_of_do_action(self, request):
        if not isinstance(request, EpRequest):
            raise ClientException(
                error_code.SDK_INVALID_REQUEST,
                error_msg.get_msg('SDK_INVALID_REQUEST'))
        try:
            status, headers, body = self._handle_single_request(request)
        except IOError as e:
            exception = ClientException(error_code.SDK_HTTP_ERROR, str(e))
            _logging.log.error(str(e))
            raise exception
            # return None, None, None, exception
        return status, headers, body.decode("utf-8"), None

    def _handle_single_request(self, request):
        url = self._host + ":" + self._port + request.get_action_name()
        self._session.headers.update(request.get_headers())
        if request.get_method() == "GET":
            response = self._session.get(url=url,
                                         timeout=request.get_timeout(),
                                         params=request.get_query_params()
                                         )
        elif request.get_method() == "PUT":
            response = self._session.put(url=url,
                                         timeout=request.get_timeout(),
                                         params=request.get_query_params(),
                                         data=json.dumps(request.get_body_params())
                                         )
        elif request.get_method() == "DELETE":
            response = self._session.delete(url=url,
                                            timeout=request.get_timeout(),
                                            params=request.get_query_params()
                                            )
        else:
            response = self._session.post(url=url,
                                          timeout=request.get_timeout(),
                                          params=request.get_query_params(),
                                          data=json.dumps(request.get_body_params())
                                          )
        return response.status_code, response.headers, response.content

    def init_user(self, login_id=None, password=None, guinea_pig_id=None):
        """
        自动登录，并设置患者
        :param login_id:
        :param password:
        :param guinea_pig_id:
        :return:
        """
        # login
        self._login_id = login_id
        if self._login_id is None:
            self._login_id = ConfigUtil().get_data('login_id')
        self._password = password
        if self._password is None:
            self._password = ConfigUtil().get_data('password')
        result = self.do_action_json(UserLoginRequest(self._login_id, self._password))
        check_result(result)

        # guinea_pig_id is not None, check current guinea pig, and reset if different
        # guinea_pig_id is None, check current guinea pig, and set if the current is none
        current_pig_id = ""
        self._guinea_pig_id = guinea_pig_id
        if self._guinea_pig_id is None:
            self._guinea_pig_id = ConfigUtil().get_data('guinea_pig_id')
        result = self.do_action_json(GuineaPigGetCurrentRequest())
        if result['data']:
            current_pig_id = result['data']['id']
        if len(current_pig_id) == 0 or (guinea_pig_id is not None and guinea_pig_id != current_pig_id):
            result = self.do_action_json(GuineaPigSetCurrentRequest(self._guinea_pig_id))
            check_result(result)

    def _parse_server_version(self) -> Optional[int]:
        status, headers, body, exception = self._implementation_of_do_action(SystemConfigQueryRequest())
        if body is None:
            return None
        return version_to_int(json.loads(body)['data']['serverVersion'].replace("v", ""))

    def __del__(self):
        if hasattr(self, "_session"):
            self._session.close()
