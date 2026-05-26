# -*- coding: utf-8 -*-
import os
import sys
import configparser
import queue
import threading
import time
from collections import deque
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from epstudiosdk.bean.collection.CollectionDeviceBean import CollectionDeviceBean
from epstudiosdk.client import EpClient
from epstudiosdk.collection.Collection import Collection
from epstudiosdk.websocket import DataReceive, DataType, EventMessage
from epstudiosdk.websocketclient import EpWebSocketClient

from epstudiosdk.request.user.UserLoginRequest import UserLoginRequest
from epstudiosdk.request.guineapig.GuineaPigSetCurrentRequest import GuineaPigSetCurrentRequest

import epstudiosdk.client
print(epstudiosdk.client.__file__)

print("Python executable:", sys.executable)

import requests
try:
    r = requests.get('http://10.10.10.111:8080', timeout=5)
    print(f"✅ Connection successful, status code: {r.status_code}")
except Exception as e:
    print(f"❌ Connection failed: {e}")

# ---------- Ensure working directory is script directory ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir,"data","config.ini")
config_path = os.path.abspath(config_path)
# ---------- Read configuration file ----------
config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')

# EPStudio connection configuration
host = config.get('config-data', 'host', fallback='http://localhost').split(';')[0]
port = config.get('config-data', 'port', fallback='8080')
websocket_host = config.get('config-data', 'websocket_host', fallback='ws://localhost').split(';')[0]
websocket_port = config.get('config-data', 'websocket_port', fallback='9000')
login_id = config.get('config-data', 'login_id', fallback='admin')
password = config.get('config-data', 'password', fallback='admin')
guinea_pig_id = config.get('config-data', 'guinea_pig_id', fallback='20220526')

# Application specific configuration
device_mac = config.get('app', 'device_mac', fallback='F9:DC:B2:3E:B4:BD')
channels_str = config.get('app', 'channels', fallback='1,2,3,4,5')
channels = [int(ch.strip()) for ch in channels_str.split(',')]
display_points = int(config.get('app', 'display_points', fallback='1000'))

# ---------- Debug prints ----------
print("host =", host)
print("port =", port)
print("websocket_host =", websocket_host)
print("device_mac =", device_mac)
print("channels =", channels)

# ---------- Global data queues for each channel ----------
data_queues = {ch: queue.Queue() for ch in channels}

# ---------- Custom event handler for WebSocket data ----------
class MyEvent(EventMessage):
    def on_data(self, data: Union[str, dict, DataReceive]):
        """Called when data is received"""
        if isinstance(data, DataReceive) and data.dataType == DataType.EP:
            for device_data in data.data:
                if device_data.deviceId == device_mac:
                    # For each channel, put the data into its queue
                    for ch in channels:
                        ch_str = str(ch)
                        if ch_str in device_data.data:
                            values = device_data.data[ch_str]  # list of sample points
                            data_queues[ch].put(values)

# ---------- Initialize EPStudio client ----------
client = EpClient(init_user_status=True)

# Manual login
login_res = client.do_action_json(UserLoginRequest(login_id, password))
print("Login result:", login_res)

# Set current guinea pig
set_patient_res = client.do_action_json(GuineaPigSetCurrentRequest(guinea_pig_id))
print("Set patient result:", set_patient_res)

# Create WebSocket client (address already forced in modified source)
event_handler = MyEvent()
websocket_client = EpWebSocketClient(
    event_msg=event_handler,
    enable_trace=False
)

# Start WebSocket client
websocket_client.start()

# Prepare collection device with all channels
device = CollectionDeviceBean(device_mac, channelStatus=channels)
collection = Collection(
    client=client,
    device_list=[device],
    record_status=False,
)

# Start collection
print("Starting collection...")
result = collection.start_collection()
print("Collection start result:", result)

# ---------- Real-time plotting ----------
# Create subplots for each channel
n_channels = len(channels)
fig, axes = plt.subplots(nrows=n_channels, ncols=1, sharex=True, figsize=(10, 2*n_channels))
if n_channels == 1:
    axes = [axes]  # ensure axes is always a list

# Initialize data buffers for each channel (deque with fixed max length)
buffers = {ch: deque(maxlen=display_points) for ch in channels}
# Initialize deques with zeros to have initial data
for ch in channels:
    buffers[ch].extend([0]*display_points)

# Create line objects for each channel
lines = {}
for i, ch in enumerate(channels):
    ax = axes[i]
    line, = ax.plot(np.arange(display_points), list(buffers[ch]), lw=1)
    lines[ch] = line
    ax.set_ylim(-2000, 2000)  # adjust as needed
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'Channel {ch}')
    ax.grid(True)

axes[-1].set_xlabel('Sample Point')
fig.suptitle(f'Real-time Waveform - Device {device_mac}', fontsize=14)

def update(frame):
    """Animation update function, called every interval"""
    for ch in channels:
        # Get all new data from the queue for this channel
        new_points = []
        while not data_queues[ch].empty():
            try:
                points = data_queues[ch].get_nowait()
                new_points.extend(points)
            except queue.Empty:
                break
        if new_points:
            # Extend the deque (automatically drops oldest if exceeds maxlen)
            buffers[ch].extend(new_points)
            # Update line data
            lines[ch].set_ydata(list(buffers[ch]))
    return list(lines.values())

# Create animation
ani = FuncAnimation(fig, update, interval=50, blit=True)

def on_close(event):
    """Stop collection and close websocket when window is closed"""
    print("Window closed, stopping collection...")
    collection.stop_collection()
    websocket_client.stop()
    print("Stopped.")

fig.canvas.mpl_connect('close_event', on_close)

print("Displaying real-time waveforms. Close the plot window to exit.")
plt.tight_layout()
plt.show()