# coding=utf-8
from epstudiosdk.exception.exceptions import ClientException
from epstudiosdk.server import EpServer
from epstudiosdk.server.web_client import EventMessageServer, EPSocketClient
from epstudiosdk.utils.config import ConfigUtil
from epstudiosdk.utils.socketclientcheck import SocketClientCheck
from epstudiosdk.utils import _logging
from epstudiosdk.utils.object import unicode_convert
from epstudiosdk.exception import error_code
import websocket
import atexit
import json
import sys
import threading
from typing import Callable, Optional, Any

from epstudiosdk.websocket import EventMessage
from epstudiosdk.websocket.websocket import parse_message, parse_data


class EpWebSocketClient:
    """
    用户接收上位机推送数据的客户端，实时接收
    此客户端只能开启一个，并且开启之后当前运行的程序不会终止。
    如果想要请求其他接口，一：提前写好代码，按业务逻辑自动执行 二：另启文件，重新main函数运行程序

    on_message：消息推送回调
    on_data：数据推送回调

    event_msg： 针对EPStudio服务端的推送监听，若此参数不为空，则以当前参数为准。
                封装 on_message、on_data、on_error、on_close、on_open方法到EventMessage接口中，用户自定义实现
                class UserEvent(EventMessage):
                    def on_message():
                        # 自定义逻辑

    server_index: int = None 指定EPStudio服务地址索引，默认0，对应config.ini中websocket_host和websocket_port的值是否用;分割
    event_msg_server： 针对SDK服务端的推送监听
    """
    def __init__(self,
                 on_message: Optional[Callable[[], Any]] = None,
                 on_data: Optional[Callable[[], Any]] = None,
                 on_error: Optional[Callable[[Any, Any], Any]] = None,
                 on_close: Optional[Callable[[], Any]] = None,
                 on_open: Optional[Callable[[], Any]] = None,
                 event_msg_server: Optional[EventMessageServer] = None,
                 event_msg: Optional[EventMessage] = None,
                 server_index: int = None,
                 enable_trace: bool = None):

        # read from configuration
        self._server_index = 0 if server_index is None or server_index < 0 else server_index
        self._parse_host_port()

        # 强制设置正确的 WebSocket 地址
        self._websocket_host = 'ws://10.10.10.82'   # 注意保留 ws://
        self._websocket_port = '9000'
        
        self._url = self._websocket_host + ":" + self._websocket_port + "/socket.io/?transport=websocket"
        self._on_message = on_message
        self._on_data = on_data
        self._on_error = on_error
        self._on_close = on_close
        self._on_open = on_open
        self._enable_trace = enable_trace
        if self._enable_trace is not None:
            _logging.enableTrace(self._enable_trace)
        else:
            self._enable_trace = False
        self._sdk_server: Optional[EpServer] = None
        self._sdk_client: Optional[EPSocketClient] = None
        self._event_msg = event_msg

        self._status_stop = 0
        self._status_run = 1
        self._event_msg_server = event_msg_server

        def on_open(ws):
            # SocketClientCheck().set_status(self._status_run)
            _logging.log.debug("Opened connection")
            if self._event_msg is not None:
                self._event_msg.ws = ws
                self._callback_msg('on_open')
            else:
                self._callback(self._on_open)

        def on_close(ws, *args):
            _logging.log.debug("Closed connection")
            # SocketClientCheck().set_status(self._status_stop)
            if len(args) == 0:
                if self._event_msg is not None:
                    self._event_msg.ws = None
                    self._callback_msg('on_close', None, None)
                else:
                    self._callback(self._on_close, None, None)
            else:
                if self._event_msg is not None:
                    self._event_msg.ws = None
                    self._callback_msg('on_close', *args)
                else:
                    self._callback(self._on_close, *args)

        def on_error(ws, error):
            _logging.log.debug("Error %s" % error)
            # SocketClientCheck().set_status(self._status_stop)
            if self._event_msg is not None:
                self._callback_msg('on_error', error)
            else:
                self._callback(self._on_error, error)

        def on_message(ws, message):
            if message.startswith("42"):
                message = message[2:]
                if sys.version < '3':
                    message = unicode_convert(json.loads(message))
                else:
                    message = json.loads(message)
                if message[0] == 'msg':
                    if self._event_msg is not None:
                        self._callback_msg('on_message', message[1])
                    else:
                        self._callback(self._on_message, message[1])
                else:
                    if self._event_msg is not None:
                        self._callback_msg('on_data', message[1])
                    else:
                        self._callback(self._on_data, message[1])
            elif message == '3':
                # ping and then pong message
                pass
            else:
                _logging.log.debug("message %s" % message)

        # init client
        # websocket.enableTrace(True)
        self._websocket = websocket.WebSocketApp(self._url,
                                                 on_open=on_open,
                                                 on_message=on_message,
                                                 on_error=on_error,
                                                 on_close=on_close)

        def start_socket():
            self._websocket.run_forever()
        self._websocket_thread = threading.Thread(target=start_socket, name='socket-client')
        self._renewal_thread = None

    def _parse_host_port(self):
        hosts = ConfigUtil().get_data('websocket_host').split(";")
        ports = ConfigUtil().get_data('websocket_port').split(";")
        host_length = len(hosts)
        port_length = len(ports)
        if host_length != port_length:
            raise ClientException(error_code.SDK_INVALID_PARAMS,
                                  "The websocket_host and websocket_port"
                                  " in config.ini values split by ';' length not same.")
        if self._server_index >= host_length:
            raise ClientException(error_code.SDK_INVALID_PARAMS,
                                  "server_index param out of websocket_host and websocket_port range.")
        self._websocket_host = hosts[self._server_index].strip()
        self._websocket_port = ports[self._server_index].strip()
        if len(self._websocket_host) == 0 or len(self._websocket_port) == 0:
            raise ClientException(error_code.SDK_INVALID_PARAMS,
                                  f"websocket_host or websocket_port at index[{self._server_index}] is empty.")

    def start(self):
        """
        开启客户端
        :return:
        """
        # check socket client is or not init
        interval = SocketClientCheck.check_time
        # if SocketClientCheck().check_status():
        #     _logging.log.debug("EpWebSocketClient has started or check_status not expire, please wait %s seconds" % interval)
        #     return
        if not self._websocket_thread.is_alive():
            # start the renewal check_status
            self._websocket_thread.start()
            atexit.register(self._stop_hook)
            event = threading.Event()
            self._renewal_thread = threading.Thread(target=self._renewal_time, args=(event, interval), name='socket-renewal')
            self._renewal_thread.daemon = True
            self._renewal_thread.start()
            if self._event_msg_server:
                # 需要启动SDK服务端，并开启监听
                self._sdk_server = EpServer(enable_trace=self._enable_trace)
                self._sdk_server.start()
                self._sdk_client = EPSocketClient(event=self._event_msg_server)
                self._sdk_client.start()

    def stop(self):
        """
        停止客户端
        :return:
        """
        self._websocket.close()
        if self._sdk_client:
            self._sdk_client.stop()
        if self._sdk_server:
            self._sdk_server.stop()

    def keep_running(self) -> bool:
        """
        客户端和服务端是否链接状态
        :return:
        """
        return self._websocket is not None and self._websocket.keep_running

    def _stop_hook(self, *args):
        self.stop()

    def __del__(self):
        if hasattr(self, "_websocket"):
            self._websocket.close()

    def _renewal_time(self, event, interval):
        while not event.wait(interval):
            # # renewal the check_status
            # SocketClientCheck().set_status(self._status_run)
            # ping to EPStudio push server to maintain the link
            if self._websocket.keep_running:
                self._websocket.send("2", 0x1)

    def _callback(self, callback, *args):
        if callback:
            try:
                callback(self, *args)
            except Exception as e:
                _logging.log.error("error from callback {}: {}".format(callback, e))
                if self._on_error:
                    self._on_error(self, e)

    def _callback_msg(self, callback, *args):
        try:
            if callback == 'on_data' and self._event_msg.impl_data:
                parse_data(self._event_msg, *args)
            elif callback == 'on_message' and self._event_msg.impl_message:
                parse_message(self._event_msg, *args)
            elif callback == 'on_open' and self._event_msg.impl_open:
                self._event_msg.on_open()
            elif callback == 'on_error' and self._event_msg.impl_error:
                self._event_msg.on_error(*args)
            elif callback == 'on_close' and self._event_msg.impl_close:
                self._event_msg.on_close(*args)
        except Exception as e:
            _logging.log.error("error from callback {}: {}".format(callback, e))
            self._event_msg.on_error(e)
