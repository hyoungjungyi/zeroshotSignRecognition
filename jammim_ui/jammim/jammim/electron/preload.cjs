const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  startPython: () => ipcRenderer.send('start-python'),
  getWeather: () => ipcRenderer.invoke("get-weather"),
  loadDefaultGestures: () => ipcRenderer.invoke('load-default-gestures'),
  loadCustomGestures: () => ipcRenderer.invoke('load-custom-gestures'),
  saveCustomGestures: (gestures) => ipcRenderer.send('save-custom-gestures', gestures),
  startWebcam: () => ipcRenderer.send('start-webcam-script'),
  trainLSTM: () => ipcRenderer.send('train-LSTM-script'),
  sendCustomGestureData: (data) => ipcRenderer.send("custom-gesture-data", data),
});
