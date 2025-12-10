#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Six-Channel Raw Monitor + Recorder + Per-Channel Linear Calibration + 20-Sample Tare
+ Per-Plot Vertical Zoom Sliders + Simple Low-Pass Filter + Calibration Curves Viewer

Features:
---------
1. Reads “Ch0=xxxxx Ch1=xxxxx … Ch5=xxxxx” from a COM port at 38 400 baud, truncates last three digits.
2. Real-time plots (2×3 grid) with per-plot vertical zoom sliders.
3. “Tare” any channel over 20 samples.
4. “Record 500 Samples” with user-entered loads → CSV.
5. “Run Calibration” → per-channel OLS → slopes.npy / intercepts.npy.
6. “Apply Calibration” toggles use of saved slopes/intercepts.
7. Simple adjustable low-pass filter.
8. “Show Calibration Plots” → scatter + fit line for all six axes.
"""
import sys, os, time, csv
import numpy as np
import serial
from collections import deque
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

class SerialReader(QtCore.QThread):
    raw_line_signal = QtCore.pyqtSignal(str)
    data_signal     = QtCore.pyqtSignal(dict)
    def __init__(self, port: str, baudrate: int):
        super().__init__()
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(0.1)
    def run(self):
        while True:
            line = self.ser.readline().decode(errors='ignore').strip()
            if not line: continue
            self.raw_line_signal.emit(line)
            parsed = {}
            for token in line.split():
                if token.startswith("Ch"):
                    try:
                        ch_label, val_str = token.split("=")
                        ch = int(ch_label[2:])
                        if 0 <= ch <= 5:
                            raw = int(val_str)
                            parsed[ch] = raw // 1000
                    except:
                        pass
            if parsed:
                self.data_signal.emit(parsed)

class LoadInputDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Loads for Recording")
        layout = QtWidgets.QFormLayout(self)
        self.fields = {}
        names = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        for i, name in enumerate(names):
            le = QtWidgets.QLineEdit(self)
            le.setPlaceholderText("0")
            le.setValidator(QtGui.QDoubleValidator(-1e6,1e6,3))
            layout.addRow(f"{name}:", le)
            self.fields[i] = le
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)
    def getLoads(self):
        if self.exec_()==QtWidgets.QDialog.Accepted:
            return {ch: float(le.text()) if le.text() else 0.0
                    for ch,le in self.fields.items()}
        return None

class SixChannelMonitor(QtWidgets.QMainWindow):
    def __init__(self, port="COM6", baudrate=38400):
        super().__init__()
        self.setWindowTitle("6-Channel Monitor & Calibrator")
        self.resize(1200, 900)

        # Buffers, tare, record, calib, filter, zoom state
        self.buffers = {ch: deque(maxlen=100) for ch in range(6)}
        self.tare_collect = {ch: False for ch in range(6)}
        self.tare_count   = {ch: 0 for ch in range(6)}
        self.tare_accum   = {ch: 0.0 for ch in range(6)}
        self.display_offsets = {ch: 0.0 for ch in range(6)}
        self.is_recording = False
        self.record_count = 0
        self.record_buffers = {ch: [] for ch in range(6)}
        self.current_loads = {ch: 0.0 for ch in range(6)}
        self.apply_calib = False
        self.slopes = None
        self.intercepts = None
        self.cal_X_all = None
        self.cal_Y_all = None
        self.enable_filter = False
        self.filter_alpha = 0.1
        self.prev_filtered = np.zeros(6)
        self.yrange = {ch: 10.0 for ch in range(6)}

        # Build UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v_main = QtWidgets.QVBoxLayout(central)
        self.console = QtWidgets.QPlainTextEdit(); self.console.setReadOnly(True)
        v_main.addWidget(self.console, stretch=1)

        # Top controls
        top_h = QtWidgets.QHBoxLayout()
        btn_zero_all = QtWidgets.QPushButton("Zero (Tare) All Axes")
        btn_zero_all.clicked.connect(self.zero_all)
        top_h.addWidget(btn_zero_all)

        btn_record = QtWidgets.QPushButton("Record 500 Samples")
        btn_record.clicked.connect(self.start_record)
        top_h.addWidget(btn_record)

        btn_run_cal = QtWidgets.QPushButton("Run Calibration")
        btn_run_cal.clicked.connect(self.run_batch_calibration)
        top_h.addWidget(btn_run_cal)

        self.btn_apply = QtWidgets.QPushButton("Apply Calibration: OFF")
        self.btn_apply.setCheckable(True)
        self.btn_apply.clicked.connect(self.toggle_apply_calib)
        top_h.addWidget(self.btn_apply)

        btn_show_cal = QtWidgets.QPushButton("Show Calibration Plots")
        btn_show_cal.clicked.connect(self.show_calibration_plots)
        top_h.addWidget(btn_show_cal)

        v_main.addLayout(top_h, stretch=0)

        # Filter controls
        filt_h = QtWidgets.QHBoxLayout()
        self.chk_filter = QtWidgets.QCheckBox("Enable Filter")
        self.chk_filter.stateChanged.connect(self.toggle_filter)
        filt_h.addWidget(self.chk_filter)
        filt_h.addWidget(QtWidgets.QLabel("Filter α (%)"))
        self.slider_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_alpha.setRange(1,100)
        self.slider_alpha.setValue(int(self.filter_alpha*100))
        self.slider_alpha.valueChanged.connect(self.update_alpha)
        filt_h.addWidget(self.slider_alpha)
        v_main.addLayout(filt_h, stretch=0)

        # Plots + zoom sliders
        grid = QtWidgets.QGridLayout()
        v_main.addLayout(grid, stretch=5)
        self.plots = {}; self.curves = {}; self.sliders = {}
        labels = ["Fx","Fy","Fz","Mx","My","Mz"]
        for ch in range(6):
            container = QtWidgets.QWidget()
            hbox = QtWidgets.QHBoxLayout(container)
            left_v = QtWidgets.QVBoxLayout()
            btn = QtWidgets.QPushButton(f"Tare {labels[ch]}")
            btn.clicked.connect(lambda _,c=ch: self.start_tare(c))
            left_v.addWidget(btn, stretch=0)
            pw = pg.PlotWidget()
            pw.setLabel('left', labels[ch]); pw.setLabel('bottom','Sample #')
            pw.showGrid(x=True,y=True,alpha=0.3)
            curve = pw.plot(pen='y', symbol='o', symbolSize=4)
            left_v.addWidget(pw, stretch=1)
            slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
            slider.setRange(1,100)
            slider.setValue(int(self.yrange[ch]))
            slider.valueChanged.connect(lambda val,c=ch: self.update_yrange(c,val))
            hbox.addLayout(left_v, stretch=8)
            hbox.addWidget(slider, stretch=2)
            row, col = divmod(ch,3)
            grid.addWidget(container, row, col)
            self.plots[ch]=pw; self.curves[ch]=curve; self.sliders[ch]=slider

        # Serial reader
        self.reader = SerialReader(port, baudrate)
        self.reader.raw_line_signal.connect(self.log_line)
        self.reader.data_signal.connect(self.handle_data)
        self.reader.start()

        # Plot timer
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.setInterval(50)
        self.plot_timer.timeout.connect(self.redraw_plots)
        self.plot_timer.start()

    # Logging
    def log_line(self, line: str):
        self.console.appendPlainText(f"[RX] {line}")

    # Tare
    def zero_all(self):
        for ch in range(6):
            self.start_tare(ch)
        self.console.appendPlainText("[Tare] Zero all axes over next 20 samples")
    def start_tare(self, ch):
        if not self.tare_collect[ch]:
            self.tare_collect[ch]=True
            self.tare_count[ch]=0
            self.tare_accum[ch]=0.0
            self.console.appendPlainText(f"[Tare] Collecting next 20 samples for Ch{ch}")

    # Recording
    def start_record(self):
        if self.is_recording:
            self.console.appendPlainText("[WARN] Already recording")
            return
        dlg = LoadInputDialog(self)
        loads = dlg.getLoads()
        if loads is None:
            self.console.appendPlainText("[Record] Cancelled")
            return
        self.is_recording=True; self.record_count=0
        for ch in range(6):
            self.current_loads[ch]=loads[ch]
            self.record_buffers[ch].clear()
        self.console.appendPlainText("[Record] Started 500 samples")

    def finish_record(self):
        names=["Fx","Fy","Fz","Mx","My","Mz"]
        parts=[f"{names[ch]}={self.current_loads[ch]}" for ch in range(6)]
        fname="(" + ",".join(parts)+").csv"
        try:
            with open(fname,'w',newline='') as f:
                w=csv.writer(f)
                w.writerow(names)
                for i in range(500):
                    w.writerow([round(self.record_buffers[ch][i],2) for ch in range(6)])
            self.console.appendPlainText(f"[Record] Saved {fname}")
            QtWidgets.QMessageBox.information(self,"Done",f"Saved {fname}")
        except Exception as e:
            self.console.appendPlainText(f"[ERR] Save failed: {e}")
            QtWidgets.QMessageBox.critical(self,"Save Error",str(e))
        self.is_recording=False; self.record_count=0

    # Calibration (integrated)
    def run_batch_calibration(self):
        dlg=QtWidgets.QFileDialog(self,"Select CSVs")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dlg.setNameFilter("CSV Files (*.csv)")
        if dlg.exec_()!=QtWidgets.QDialog.Accepted: return
        files=dlg.selectedFiles()
        if not files: return

        X_list=[]; Y_list=[]
        for path in files:
            fname=os.path.basename(path)
            try:
                core=fname.rstrip(".csv").strip("()")
                parts=[p.strip() for p in core.split(",")]
                true_load=np.zeros(6)
                for part in parts:
                    name,val=part.split("=")
                    idx=["Fx","Fy","Fz","Mx","My","Mz"].index(name.strip())
                    true_load[idx]=float(val)
                data=[]
                with open(path,newline='') as f:
                    rdr=csv.reader(f)
                    hdr=next(rdr)
                    if len(hdr)!=6: raise ValueError("Header≠6 cols")
                    for row in rdr:
                        if len(row)!=6: raise ValueError("Row≠6 cols")
                        data.append([float(x) for x in row])
                arr=np.array(data)
                if arr.shape[0]!=500: raise ValueError("Rows≠500")
                X_list.append(arr)
                Y_list.append(np.tile(true_load,(500,1)))
            except Exception as e:
                QtWidgets.QMessageBox.warning(self,"Parse Error",f"{fname}: {e}")
                continue

        if not X_list:
            QtWidgets.QMessageBox.information(self,"No Data","No CSV parsed.")
            return
        self.cal_X_all=np.vstack(X_list)
        self.cal_Y_all=np.vstack(Y_list)

        # OLS
        slopes=np.zeros(6); intercepts=np.zeros(6)
        for ch in range(6):
            a,b=np.polyfit(self.cal_X_all[:,ch],self.cal_Y_all[:,ch],1)
            slopes[ch]=a; intercepts[ch]=b
        self.slopes, self.intercepts = slopes, intercepts

        txt="\n".join(f"{['Fx','Fy','Fz','Mx','My','Mz'][ch]}: slope={a:.6f}, intercept={b:.6f}"
                      for ch,(a,b) in enumerate(zip(slopes,intercepts)))
        mb=QtWidgets.QMessageBox(self)
        mb.setWindowTitle("Calibration Done")
        mb.setText("Fits:")
        mb.setDetailedText(txt)
        mb.exec_()

        if QtWidgets.QMessageBox.question(self,"Save?","Save slopes.npy & intercepts.npy?",
                                          QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)==QtWidgets.QMessageBox.Yes:
            np.save("slopes.npy",slopes)
            np.save("intercepts.npy",intercepts)
            QtWidgets.QMessageBox.information(self,"Saved","slopes.npy & intercepts.npy")

    def toggle_apply_calib(self):
        if not self.apply_calib:
            try:
                slopes=np.load("slopes.npy")
                intercepts=np.load("intercepts.npy")
                if slopes.shape!=(6,) or intercepts.shape!=(6,):
                    raise ValueError("Bad shapes")
                self.slopes, self.intercepts = slopes, intercepts
                self.apply_calib=True
                self.btn_apply.setText("Apply Calibration: ON")
                self.console.appendPlainText("[Calib] Applied")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self,"Load Error",str(e))
                self.btn_apply.setChecked(False)
        else:
            self.apply_calib=False
            self.btn_apply.setText("Apply Calibration: OFF")
            self.console.appendPlainText("[Calib] Off")

    # Filter
    def toggle_filter(self, state):
        self.enable_filter = (state==QtCore.Qt.Checked)
        if self.enable_filter:
            self.console.appendPlainText(f"[Filter] On α={self.filter_alpha:.2f}")
            for ch in range(6):
                buf=list(self.buffers[ch])
                if buf: self.prev_filtered[ch]=sum(buf)/len(buf)
        else:
            self.console.appendPlainText("[Filter] Off")

    def update_alpha(self, val):
        self.filter_alpha=val/100.0
        self.console.appendPlainText(f"[Filter] α={self.filter_alpha:.2f}")

    # Zoom sliders
    def update_yrange(self,ch,val):
        self.yrange[ch]=float(val)
        buf=list(self.buffers[ch])
        if buf:
            m=sum(buf)/len(buf)
            r=self.yrange[ch]
            self.plots[ch].setYRange(m-r,m+r,padding=0)

    # Data handler
    def handle_data(self,raw_dict):
        raw_vec=np.zeros(6)
        for ch,val in raw_dict.items(): raw_vec[ch]=float(val)
        if self.apply_calib and self.slopes is not None:
            try:
                calib = self.slopes*raw_vec + self.intercepts
            except:
                calib = raw_vec.copy()
        else:
            calib = raw_vec.copy()
        final = np.zeros(6)
        for ch in range(6):
            v=calib[ch]
            if self.tare_collect[ch]:
                self.tare_accum[ch]+=v
                self.tare_count[ch]+=1
                if self.tare_count[ch]>=20:
                    off=self.tare_accum[ch]/20.0
                    self.display_offsets[ch]=off
                    self.tare_collect[ch]=False
                    self.console.appendPlainText(f"[Tare] Ch{ch} off={off:.2f}")
                disp=0.0
            else:
                disp=v-self.display_offsets[ch]
            if self.enable_filter:
                a=self.filter_alpha
                p=self.prev_filtered[ch]
                f=a*disp+(1-a)*p
                self.prev_filtered[ch]=f
                final[ch]=f
            else:
                final[ch]=disp
            self.buffers[ch].append(final[ch])
            if self.is_recording and self.record_count<500:
                self.record_buffers[ch].append(final[ch])

        if self.is_recording:
            self.record_count+=1
            if self.record_count==500:
                self.finish_record()

    # Plot refresh
    def redraw_plots(self):
        for ch,buf in self.buffers.items():
            data=list(buf)
            if not data: continue
            self.curves[ch].setData(data)
            m=sum(data)/len(data)
            r=self.yrange[ch]
            self.plots[ch].setYRange(m-r,m+r,padding=0)
            latest=data[-1]
            lbl=["Fx","Fy","Fz","Mx","My","Mz"][ch]
            sfx = ""
            if self.apply_calib: sfx+=" (calib)"
            if self.enable_filter: sfx+=" (filt)"
            self.plots[ch].setTitle(f"{lbl}{sfx}: {latest:.2f}")

    # Show calibration plots
    def show_calibration_plots(self):
        if self.cal_X_all is None or self.slopes is None:
            QtWidgets.QMessageBox.information(self,"No Data",
                "Please run calibration first.")
            return
        dlg = QtWidgets.QWidget()
        dlg.setWindowTitle("Calibration Curves")
        layout = QtWidgets.QGridLayout(dlg)
        for ch in range(6):
            pw = pg.PlotWidget()
            pw.setLabel('left',"True Load")
            pw.setLabel('bottom',"Raw Reading")
            pw.showGrid(x=True,y=True,alpha=0.3)
            x = self.cal_X_all[:,ch]
            y = self.cal_Y_all[:,ch]
            pw.plot(x, y, pen=None, symbol='o', symbolSize=5)
            a,b = self.slopes[ch], self.intercepts[ch]
            xx = np.array([x.min(), x.max()])
            yy = a*xx + b
            pw.plot(xx, yy, pen='r', width=2)
            row,col = divmod(ch,3)
            layout.addWidget(pw, row, col)
        dlg.resize(800,600)
        dlg.show()
        self.calib_win = dlg  # keep reference

if __name__=="__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    win = SixChannelMonitor(port="COM6", baudrate=38400)
    win.show()
    sys.exit(app.exec_())
