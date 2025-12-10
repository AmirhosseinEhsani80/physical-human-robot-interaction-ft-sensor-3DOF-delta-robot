#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
delta_phri_admittance_sim_zwidth_logger_gui.py

Δ-robot PHRI — Admittance + Z-Width + GUI Logger (+6-axis, forces-from-moments)
- Read Fx,Fy,Fz,Mx,My,Mz (6-axis)
- Derive handle forces from torques with handle offset h=70 mm (z-offset only)
- Log/plot measured forces AND derived-from-moments forces
- Draw user and wall force vectors in 3D
"""

import threading, time, math, os, csv, datetime
import numpy as np
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, TextBox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ——— Sensor & Admittance Config ———
PORT         = "COM6"
BAUDRATE     = 38400
TIMEOUT      = 0.05
TARE_SAMPLES = 20

ALPHA        = 0.28           # low-pass on sensor
THRESHOLD_N  = 1.0            # deadband for admittance input only (not for display)

# Handle offset (force application point) — pure +Z offset
HANDLE_HEIGHT_M = 0.070       # 70 mm above the sensor origin
# If your torque calibration outputs N·mm, set MOMENT_TO_NM = 0.001
MOMENT_TO_NM    = 1.0         # 1.0 if Mx,My,Mz already in N·m; 0.001 if N·mm

# ——— Admittance (lighter feel) ———
M        = 0.02
B_COEFF  = 0.08
K_STIFF  = 0.00002

# ——— Z-Width / Virtual-wall ———
K_WALL          = 0.06
K_WALL_MIN      = 0.0005
K_WALL_MAX_CAP  = 0.1
K_UP_RATE       = 0.010
K_DOWN_FACTOR   = 0.05
D_WALL          = 0.5
GAMMA_PASS      = 1.2
E_ENV_THRESH    = 1.8
E_ENV_LEAK      = 0.985
CONTACT_MIN_DT  = 0.08
AUTO_PROBE      = True

# Hard clamps (far from visible walls)

X_HARD_MAX, X_HARD_MIN =  380.0, -380.0
Y_HARD_MAX, Y_HARD_MIN =  380.0, -380.0
Z_HARD_MAX, Z_HARD_MIN =  280.0, -580.0

# Visible walls (what the user "feels")
X_MAX, X_MIN =  100.0, -100.0
Y_MAX, Y_MIN =  100.0, -100.0
Z_MAX, Z_MIN =   -150.0, -450.0

# Scale for drawing force vectors (mm per Newton) + hard cap on arrow length
FORCE_ARROW_SCALE = 18.0
FORCE_ARROW_MAX   = 140.0

# Load calibration arrays for 6 channels: [Fx, Fy, Fz, Mx, My, Mz]
# (ensure slopes.npy / intercepts.npy contain 6 entries in this order)
slopes     = np.load("slopes.npy")[:6]
intercepts = np.load("intercepts.npy")[:6]

# Shared state (forces + moments)
filtered_force   = np.zeros(3)  # [Fx,Fy,Fz] N
filtered_moment  = np.zeros(3)  # [Mx,My,Mz] (N·m after MOMENT_TO_NM)
# For thread-internal tare
display_offset   = np.zeros(6)
tare_event       = threading.Event()
exit_event       = threading.Event()

# ——— Δ-Robot Geometry ———
f, e, rf, re = 130.0, 100.0, 305.0, 595.0
sqrt3 = math.sqrt(3.0); sin60 = sqrt3/2.0; cos60 = 0.5
r_base = f
BASE_PTS = [
    np.array([ 0.0,               -r_base, 0.0]),
    np.array([ r_base * sin60,    r_base * cos60, 0.0]),
    np.array([-r_base * sin60,    r_base * cos60, 0.0])
]
r_eff = e
EFF_OFFS = [
    np.array([ 0.0,              -r_eff, 0.0]),
    np.array([ r_eff * sin60,    r_eff * cos60, 0.0]),
    np.array([-r_eff * sin60,    r_eff * cos60, 0.0])
]
RZ = [
    np.eye(3),
    np.array([[ cos60, -sin60, 0],[ sin60,  cos60, 0],[0,0,1]]),
    np.array([[ cos60,  sin60, 0],[-sin60,  cos60, 0],[0,0,1]])
]

# ——— IK ———
def delta_calcAngleYZ(x0, y0, z0):
    y1 = -0.5 * 0.57735 * f
    a  = (x0*x0 + y0*y0 + z0*z0 + rf*rf - re*re - y1*y1) / (2 * z0)
    b  = (y1 - y0) / z0
    D  = -(a + b * y1)**2 + rf * (b*b * rf + rf)
    if D < 0: raise ValueError("Point out of reach")
    yj = (y1 - a*b - math.sqrt(D)) / (b*b + 1)
    zj = a + b*yj
    return math.degrees(math.atan2(-zj, y1 - yj))

def inverse_kinematics(x, y, z):
    t1 = delta_calcAngleYZ(x, y, z)
    t2 = delta_calcAngleYZ(x*cos60 + y*sin60, y*cos60 - x*sin60, z)
    t3 = delta_calcAngleYZ(x*cos60 - y*sin60, y*cos60 + x*sin60, z)
    return np.array([t1, t2, t3])

# ——— Sensor thread ———
def sensor_thread():
    global filtered_force, filtered_moment
    ser = serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT)
    time.sleep(0.1)
    prev = np.zeros(6)
    collecting = False
    accum = np.zeros(6); count = 0
    while not exit_event.is_set():
        if tare_event.is_set() and not collecting:
            collecting = True; accum[:] = 0.0; count = 0; tare_event.clear()
        raw_line = ser.readline()
        if not raw_line: continue
        line = raw_line.decode("utf-8", "ignore").strip()
        raw = np.zeros(6)
        for tok in line.split():
            if tok.startswith("Ch"):
                try:
                    idx = int(tok[2])
                except:
                    continue
                if 0 <= idx < 6:
                    # keep same scaling as your original code (integer // 1000)
                    raw[idx] = int(tok.split("=")[1]) // 1000
        # Calibrate 6 channels
        calib = slopes * raw + intercepts  # [Fx,Fy,Fz,Mx,My,Mz] (Mx..Mz in your calibration units)
        # Convert torques to N·m if needed
        calib[3:6] = calib[3:6] * MOMENT_TO_NM

        if collecting:
            accum += calib; count += 1
            if count >= TARE_SAMPLES:
                display_offset[:] = accum / count; collecting = False
        else:
            if np.all(display_offset == 0): continue
            disp = calib - display_offset
            prev[:] = ALPHA * disp + (1 - ALPHA) * prev
            filtered_force[:]  = prev[:3]
            filtered_moment[:] = prev[3:6]
    ser.close()

# ——— Z-Width helpers ———
def wall_contact_force(pos, vel, k_wall, d_wall):
    F = np.zeros(3); contact = False; face = "-"
    # X+
    if pos[0] > X_MAX:
        pen = pos[0] - X_MAX; n = np.array([1.0, 0.0, 0.0]); vn = float(np.dot(vel, n))
        F -= (k_wall * pen + d_wall * vn) * n; contact = True; face = "X+"
    # X-
    if pos[0] < X_MIN:
        pen = X_MIN - pos[0]; n = np.array([-1.0, 0.0, 0.0]); vn = float(np.dot(vel, n))
        F -= (k_wall * pen + d_wall * vn) * n; contact = True; face = "X-"
    # Y+
    if pos[1] > Y_MAX:
        pen = pos[1] - Y_MAX; n = np.array([0.0, 1.0, 0.0]); vn = float(np.dot(vel, n))
        F -= (k_wall * pen + d_wall * vn) * n; contact = True; face = "Y+"
    # Y-
    if pos[1] < Y_MIN:
        pen = Y_MIN - pos[1]; n = np.array([0.0, -1.0, 0.0]); vn = float(np.dot(vel, n))
        F -= (k_wall * pen + d_wall * vn) * n; contact = True; face = "Y-"
    # Z+
    if pos[2] > Z_MAX:
        pen = pos[2] - Z_MAX; n = np.array([0.0, 0.0, 1.0]); vn = float(np.dot(vel, n))
        F -= (k_wall * pen + d_wall * vn) * n; contact = True; face = "Z+"
    # Z-
    if pos[2] < Z_MIN:
        pen = Z_MIN - pos[2]; n = np.array([0.0, 0.0, -1.0]); vn = float(np.dot(vel, n))
        F -= (k_wall * pen + d_wall * vn) * n; contact = True; face = "Z-"
    return F, contact, face

def print_snapshot():
    print(f"[Z-Width] K_wall={K_WALL:.4f} N/mm, D_wall={D_WALL:.4f} N·s/mm, "
          f"K_peak={K_PEAK:.4f} N/mm, E_env={E_env:.3f} N·mm, face={LAST_FACE}")

# ——— Z-Width observer/adaptation state ———
E_env        = 0.0
CONTACT_T0   = None
LAST_FACE    = "-"
K_PEAK       = 0.0
STABLE_T     = 0.0

# ——— Logger state ———
LOG_ACTIVE     = False
LOG_TYPE       = None      # 'zwidth','forces','passivity','state','all'
LOG_T0         = 0.0
LOG_DUR        = 10.0
LOG_DATA       = {}
LOG_QUEUE      = []
LOG_DIR        = "zwidth_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def _get_log_dur():
    """Read seconds from TextBox; clamp to [1, 300]."""
    global LOG_DUR
    try:
        val = float(tb_dur.text)
        if not np.isfinite(val): raise ValueError
        LOG_DUR = float(np.clip(val, 1.0, 300.0))
    except Exception:
        LOG_DUR = 10.0
        tb_dur.set_val("10")  # reflect fallback

def start_logging(typ):
    """Start a logging session for given type, if none active."""
    global LOG_ACTIVE, LOG_TYPE, LOG_T0, LOG_DATA
    if LOG_ACTIVE: return
    _get_log_dur()
    LOG_ACTIVE = True; LOG_TYPE = typ; LOG_T0 = time.time()
    if typ == 'zwidth':
        LOG_DATA = {'t':[], 'K_wall':[], 'D_wall':[], 'E_env':[], 'Fwall_mag':[], 'contact':[]}
    elif typ == 'forces':
        LOG_DATA = {'t':[],
                    # measured forces (sensor)
                    'Fx':[], 'Fy':[], 'Fz':[], 'Fuser_mag':[],
                    # derived forces from moments (at handle)
                    'Fx_fromM':[], 'Fy_fromM':[], 'Fz_fromM':[], 'F_fromM_mag':[],
                    # wall forces
                    'Fwall_x':[], 'Fwall_y':[], 'Fwall_z':[], 'Fwall_mag':[],
                    # torques (sensor)
                    'Mx':[], 'My':[], 'Mz':[],
                    # motion
                    'v_mag':[]}
    elif typ == 'passivity':
        LOG_DATA = {'t':[], 'P_env':[], 'E_env':[]}
    elif typ == 'state':
        LOG_DATA = {'t':[], 'x':[], 'y':[], 'z':[]}
    elif typ == 'all':
        LOG_DATA = {'t':[], 'K_wall':[], 'D_wall':[], 'E_env':[], 'Fwall_mag':[], 'contact':[],
                    'Fx':[], 'Fy':[], 'Fz':[], 'Fuser_mag':[],
                    'Fx_fromM':[], 'Fy_fromM':[], 'Fz_fromM':[], 'F_fromM_mag':[],
                    'Fwall_x':[], 'Fwall_y':[], 'Fwall_z':[],
                    'Mx':[], 'My':[], 'Mz':[],
                    'v_mag':[], 'P_env':[],
                    'x':[], 'y':[], 'z':[], 'face':[]}

def cancel_logging(_=None):
    global LOG_ACTIVE, LOG_TYPE, LOG_DATA
    LOG_ACTIVE = False; LOG_TYPE = None; LOG_DATA = {}
    for k in timer_texts: timer_texts[k].set_text("")

def maybe_finish_logging():
    """If session completed, dump CSV and queue plotting."""
    global LOG_ACTIVE, LOG_TYPE, LOG_DATA
    if not LOG_ACTIVE: return
    if (time.time() - LOG_T0) < LOG_DUR: return
    LOG_ACTIVE = False
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(LOG_DIR, f"{ts}_{LOG_TYPE}.csv")
    keys = list(LOG_DATA.keys())
    try:
        with open(fname, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(keys)
            for i in range(len(LOG_DATA[keys[0]])):
                w.writerow([LOG_DATA[k][i] for k in keys])
        print(f"[LOG] Saved {LOG_TYPE} CSV: {fname}")
    except Exception as e:
        print(f"[LOG] Error saving CSV: {e}")
    LOG_QUEUE.append({'type': LOG_TYPE, 'data': LOG_DATA, 'file': fname})
    for k in timer_texts: timer_texts[k].set_text("")
    LOG_TYPE = None; LOG_DATA = {}

def plot_completed_logs():
    """Pop one completed log (if any) and show plots in a new figure."""
    if not LOG_QUEUE: return
    job = LOG_QUEUE.pop(0)
    typ = job['type']; data = job['data']; fname = job['file']
    t = np.array(data['t'])

    if typ == 'state':
        fig = plt.figure(figsize=(9, 7)); fig.canvas.manager.set_window_title(f"Log: {typ} ({os.path.basename(fname)})")
        ax1 = fig.add_subplot(3,1,1); ax1.plot(t, data['x']); ax1.set_ylabel("x (mm)"); ax1.grid(True, alpha=0.3)
        ax1.axhline(X_MIN, ls='--', c='k', alpha=0.35); ax1.axhline(X_MAX, ls='--', c='k', alpha=0.35)
        ax2 = fig.add_subplot(3,1,2); ax2.plot(t, data['y']); ax2.set_ylabel("y (mm)"); ax2.grid(True, alpha=0.3)
        ax2.axhline(Y_MIN, ls='--', c='k', alpha=0.35); ax2.axhline(Y_MAX, ls='--', c='k', alpha=0.35)
        ax3 = fig.add_subplot(3,1,3); ax3.plot(t, data['z']); ax3.set_ylabel("z (mm)"); ax3.set_xlabel("Time (s)"); ax3.grid(True, alpha=0.3)
        ax3.axhline(Z_MIN, ls='--', c='k', alpha=0.35); ax3.axhline(Z_MAX, ls='--', c='k', alpha=0.35)
        fig.suptitle("State — dotted wall limits")
        fig.tight_layout(); fig.show(); return

    if typ == 'all':
        fig = plt.figure(figsize=(13, 9)); fig.canvas.manager.set_window_title(f"Log: {typ} ({os.path.basename(fname)})")
        gs = fig.add_gridspec(3, 5, wspace=0.28, hspace=0.32)
        ax1 = fig.add_subplot(gs[0,0]); ax1.plot(t, data['K_wall']); ax1.set_title("K_wall (N/mm)"); ax1.grid(True, alpha=0.3)
        ax2 = fig.add_subplot(gs[0,1]); ax2.plot(t, data['D_wall']); ax2.set_title("D_wall (N·s/mm)"); ax2.grid(True, alpha=0.3)
        ax3 = fig.add_subplot(gs[0,2]); ax3.plot(t, data['E_env']); ax3.set_title("E_env (N·mm)"); ax3.grid(True, alpha=0.3)
        ax4 = fig.add_subplot(gs[0,3]); ax4.plot(t, data['P_env']);  ax4.set_title("P_env (N·mm/s)"); ax4.grid(True, alpha=0.3)
        ax5 = fig.add_subplot(gs[0,4]); ax5.plot(t, data['v_mag']); ax5.set_title("|v| (mm/s)"); ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[1,0]); ax6.plot(t, data['Fx'], label='Fx'); ax6.plot(t, data['Fx_fromM'],'--',label='Fx_fromM'); ax6.legend(fontsize=8); ax6.set_title("Fx (meas vs from M)"); ax6.grid(True, alpha=0.3)
        ax7 = fig.add_subplot(gs[1,1]); ax7.plot(t, data['Fy'], label='Fy'); ax7.plot(t, data['Fy_fromM'],'--',label='Fy_fromM'); ax7.legend(fontsize=8); ax7.set_title("Fy (meas vs from M)"); ax7.grid(True, alpha=0.3)
        ax8 = fig.add_subplot(gs[1,2]); ax8.plot(t, data['Fz'], label='Fz'); ax8.plot(t, data['Fz_fromM'],'--',label='Fz_fromM'); ax8.legend(fontsize=8); ax8.set_title("Fz (meas vs from M)"); ax8.grid(True, alpha=0.3)
        ax9 = fig.add_subplot(gs[1,3]); ax9.plot(t, data['Fuser_mag'], label='|F_user|'); ax9.plot(t, data['F_fromM_mag'],'--',label='|F_fromM|'); ax9.plot(t, data['Fwall_mag'], label='|F_wall|'); ax9.legend(fontsize=8); ax9.set_title("Magnitudes (N)"); ax9.grid(True, alpha=0.3)
        ax10 = fig.add_subplot(gs[1,4]); ax10.plot(t, data['Mx'], label='Mx'); ax10.plot(t, data['My'], label='My'); ax10.plot(t, data['Mz'], label='Mz'); ax10.legend(fontsize=8); ax10.set_title("Torques (N·m)"); ax10.grid(True, alpha=0.3)

        ax11 = fig.add_subplot(gs[2,0]); ax11.plot(t, data['Fwall_x'], label='Fx_wall'); ax11.plot(t, data['Fwall_y'], label='Fy_wall'); ax11.plot(t, data['Fwall_z'], label='Fz_wall'); ax11.legend(fontsize=8); ax11.set_title("Wall Force (N)"); ax11.grid(True, alpha=0.3)
        ax12 = fig.add_subplot(gs[2,1]); ax12.plot(t, data['x']); ax12.set_title("x (mm)"); ax12.grid(True, alpha=0.3); ax12.axhline(X_MIN, ls='--', c='k', alpha=0.35); ax12.axhline(X_MAX, ls='--', c='k', alpha=0.35)
        ax13 = fig.add_subplot(gs[2,2]); ax13.plot(t, data['y']); ax13.set_title("y (mm)"); ax13.grid(True, alpha=0.3); ax13.axhline(Y_MIN, ls='--', c='k', alpha=0.35); ax13.axhline(Y_MAX, ls='--', c='k', alpha=0.35)
        ax14 = fig.add_subplot(gs[2,3]); ax14.plot(t, data['z']); ax14.set_title("z (mm)"); ax14.grid(True, alpha=0.3); ax14.axhline(Z_MIN, ls='--', c='k', alpha=0.35); ax14.axhline(Z_MAX, ls='--', c='k', alpha=0.35)
        ax15 = fig.add_subplot(gs[2,4]); ax15.axis('off'); ax15.text(0.02,0.7,f"face: {data['face'][-1] if data['face'] else '-'}", fontsize=10)
        fig.suptitle("ALL Signals (incl. 6-axis + forces-from-moments)")
        fig.tight_layout(); fig.show(); return

    # Other single-type figures
    fig = plt.figure(figsize=(10, 10)); fig.canvas.manager.set_window_title(f"Log: {typ} ({os.path.basename(fname)})")
    if typ == 'zwidth':
        ax1 = fig.add_subplot(2,2,1); ax1.plot(t, data['K_wall']); ax1.set_title("K_wall (N/mm)"); ax1.grid(True, alpha=0.3)
        ax2 = fig.add_subplot(2,2,2); ax2.plot(t, data['D_wall']); ax2.set_title("D_wall (N·s/mm)"); ax2.grid(True, alpha=0.3)
        ax3 = fig.add_subplot(2,2,3); ax3.plot(t, data['E_env']); ax3.set_title("E_env (N·mm)"); ax3.grid(True, alpha=0.3)
        ax4 = fig.add_subplot(2,2,4); ax4.plot(t, data['Fwall_mag']); ax4.set_title("|F_wall| (N)"); ax4.grid(True, alpha=0.3)
        fig.suptitle("Z-Width Logging")
    elif typ == 'forces':
        # overlay measured vs derived-from-moments
        ax1 = fig.add_subplot(5,1,1); ax1.plot(t, data['Fx'], label='Fx'); ax1.plot(t, data['Fx_fromM'],'--', label='Fx_fromM'); ax1.set_ylabel("Fx (N)"); ax1.grid(True, alpha=0.3); ax1.legend(ncol=2, fontsize=8)
        ax2 = fig.add_subplot(5,1,2); ax2.plot(t, data['Fy'], label='Fy'); ax2.plot(t, data['Fy_fromM'],'--', label='Fy_fromM'); ax2.set_ylabel("Fy (N)"); ax2.grid(True, alpha=0.3); ax2.legend(ncol=2, fontsize=8)
        ax3 = fig.add_subplot(5,1,3); ax3.plot(t, data['Fz'], label='Fz'); ax3.plot(t, data['Fz_fromM'],'--', label='Fz_fromM'); ax3.set_ylabel("Fz (N)"); ax3.grid(True, alpha=0.3); ax3.legend(ncol=2, fontsize=8)
        ax4 = fig.add_subplot(5,1,4); ax4.plot(t, data['Fuser_mag'], label='|F_user|'); ax4.plot(t, data['F_fromM_mag'],'--', label='|F_fromM|'); ax4.plot(t, data['Fwall_mag'], label='|F_wall|'); ax4.set_ylabel("N"); ax4.grid(True, alpha=0.3); ax4.legend(ncol=3, fontsize=8)
        ax5 = fig.add_subplot(5,1,5); ax5.plot(t, data['Mx'], label='Mx'); ax5.plot(t, data['My'], label='My'); ax5.plot(t, data['Mz'], label='Mz'); ax5.set_ylabel("N·m"); ax5.set_xlabel("Time (s)"); ax5.grid(True, alpha=0.3); ax5.legend(ncol=3, fontsize=8)
        fig.suptitle("Forces (measured vs derived-from-moments) + torques")
    elif typ == 'passivity':
        ax1 = fig.add_subplot(2,1,1); ax1.plot(t, data['P_env']); ax1.set_ylabel("P_env (N·mm/s)"); ax1.grid(True, alpha=0.3)
        ax2 = fig.add_subplot(2,1,2); ax2.plot(t, data['E_env']); ax2.set_ylabel("E_env (N·mm)"); ax2.set_xlabel("Time (s)"); ax2.grid(True, alpha=0.3)
        fig.suptitle("Passivity")
    fig.tight_layout(); fig.show()

# ——— Start sensor thread ———
threading.Thread(target=sensor_thread, daemon=True).start()

# ——— Matplotlib 3D + GUI layout ———
plt.style.use('default')
fig = plt.figure(figsize=(12.4, 9.4))
# Leave space on the right for GUI
plt.subplots_adjust(left=0.05, right=0.73, bottom=0.05, top=0.95)

ax3d  = fig.add_subplot(111, projection='3d')
ax3d.set_xlim(-600, 600); ax3d.set_ylim(-600, 600); ax3d.set_zlim(-650, 300)
ax3d.set_xlabel('X (mm)'); ax3d.set_ylabel('Y (mm)'); ax3d.set_zlabel('Z (mm)')
ax3d.set_title("Δ-Robot PHRI: Admittance + Z-Width + GUI Logger", pad=16)

# ——— Fancy workspace: box + faces + floor grid ———
def draw_workspace(ax):
    faces = []
    for xw in [X_MIN, X_MAX]:
        faces.append([(xw, Y_MIN, Z_MIN),(xw, Y_MIN, Z_MAX),(xw, Y_MAX, Z_MAX),(xw, Y_MAX, Z_MIN)])
    for yw in [Y_MIN, Y_MAX]:
        faces.append([(X_MIN, yw, Z_MIN),(X_MIN, yw, Z_MAX),(X_MAX, yw, Z_MAX),(X_MAX, yw, Z_MIN)])
    for zw in [Z_MIN, Z_MAX]:
        faces.append([(X_MIN, Y_MIN, zw),(X_MAX, Y_MIN, zw),(X_MAX, Y_MAX, zw),(X_MIN, Y_MAX, zw)])
    coll = Poly3DCollection(faces, facecolors=['#9bbcf2']*6, alpha=0.08, edgecolors='#5f6a7a', linewidths=0.8)
    ax.add_collection3d(coll)
    # box edges
    edges = [
        (X_MIN,Y_MIN,Z_MIN),(X_MIN,Y_MIN,Z_MAX),(X_MIN,Y_MAX,Z_MIN),(X_MIN,Y_MAX,Z_MAX),
        (X_MAX,Y_MIN,Z_MIN),(X_MAX,Y_MIN,Z_MAX),(X_MAX,Y_MAX,Z_MIN),(X_MAX,Y_MAX,Z_MAX)
    ]
    e = edges
    def L(p,q): ax.plot([p[0],q[0]],[p[1],q[1]],[p[2],q[2]], color='#4a4a4a', lw=2.0, alpha=0.9)
    L(e[0],e[1]); L(e[0],e[2]); L(e[0],e[4])
    L(e[3],e[1]); L(e[3],e[2]); L(e[3],e[7])
    L(e[5],e[1]); L(e[5],e[7]); L(e[5],e[4])
    L(e[6],e[2]); L(e[6],e[4]); L(e[6],e[7])
    # floor grid
    step = 30.0
    xs = np.arange(X_MIN, X_MAX+1e-6, step)
    ys = np.arange(Y_MIN, Y_MAX+1e-6, step)
    for x in xs:
        ax.plot([x, x], [Y_MIN, Y_MAX], [Z_MIN, Z_MIN], color='#888', lw=0.6, alpha=0.35)
    for y in ys:
        ax.plot([X_MIN, X_MAX], [y, y], [Z_MIN, Z_MIN], color='#888', lw=0.6, alpha=0.35)

draw_workspace(ax3d)

# Base triangle (filled, more solid)
base_fill = Poly3DCollection([[(BASE_PTS[i][0], BASE_PTS[i][1], BASE_PTS[i][2]) for i in [0,1,2]]],
                             facecolors='#406d9b', alpha=0.35, edgecolor='#2a5578', linewidths=1.8)
ax3d.add_collection3d(base_fill)

# Effector polygon (filled + dashed outline)
eff_fill = Poly3DCollection([[(0,0,0),(0,0,0),(0,0,0)]],
                            facecolors='#e34a33', alpha=0.25, edgecolor='#c43722', linewidths=1.2)
ax3d.add_collection3d(eff_fill)
eff_outline = ax3d.plot([], [], [], 'r--', lw=1.6, alpha=0.85)[0]

# Robot arms (no moving scatters ⇒ no stuck dots)
upper_arms = [ax3d.plot([], [], [], color='#1f77b4', lw=3.0)[0] for _ in range(3)]  # base→elbow
fore_arms  = [ax3d.plot([], [], [], color='#ff7f0e', lw=3.0)[0] for _ in range(3)]  # elbow→effector

# Static base joint markers are OK (do not move)
ax3d.scatter([p[0] for p in BASE_PTS],[p[1] for p in BASE_PTS],[p[2] for p in BASE_PTS],
             s=18, c='#1f6fa4', depthshade=True)

# Force vectors (as simple line segments with scaling)
force_user_line = ax3d.plot([], [], [], color='tab:green', lw=3.0, label='F_user')[0]
force_wall_line = ax3d.plot([], [], [], color='tab:purple', lw=3.0, label='F_wall')[0]

# HUD on the 3D view
hud_text = ax3d.text2D(0.02, 0.95, "", transform=ax3d.transAxes, color='dimgray')

# Admittance state
current_pos = np.array([0.0, 0.0, -475.0])
current_vel = np.zeros(3)
equilibrium = current_pos.copy()
dt = 1/30  # ~30 Hz

# Z-width state
E_env      = 0.0
CONTACT_T0 = None
LAST_FACE  = "-"
K_PEAK     = 0.0
STABLE_T   = 0.0

t_start = time.time()

# ——— GUI Buttons (right panel) ———
_btn_refs = []
def _add_button(x, y, w, h, label, callback):
    axb = fig.add_axes([x, y, w, h]); btn = Button(axb, label); btn.on_clicked(callback)
    _btn_refs.append(btn); return btn

# Layout
X0, W, H, GAP = 0.76, 0.20, 0.045, 0.010

# Sensor readout box in GUI
ax_sensor = fig.add_axes([X0, 0.945, W, 0.04])
ax_sensor.set_facecolor('#f7f7f7'); ax_sensor.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
for spine in ax_sensor.spines.values(): spine.set_alpha(0.25)
sensor_text = ax_sensor.text(0.02, 0.5, "F=[0,0,0]N |F|=0  M=[0,0,0]N·m",
                             transform=ax_sensor.transAxes, va='center', fontsize=9, family='monospace')

# Log duration TextBox
ax_dur = fig.add_axes([X0, 0.900, W*0.65, 0.035])
tb_dur = TextBox(ax_dur, "Dur(s): ", initial=f"{LOG_DUR:.0f}")
tb_dur.cursor.set_visible(True)

# Buttons
y = 0.845
def gui_tare(event): tare_event.set()
def gui_autoprobe(event):
    global AUTO_PROBE
    AUTO_PROBE = not AUTO_PROBE
    btn_auto.label.set_text(f"AutoProbe: {'ON' if AUTO_PROBE else 'OFF'}")
def gui_k_plus(event):
    global K_WALL; K_WALL = min(K_WALL * 1.10, K_WALL_MAX_CAP)
def gui_k_minus(event):
    global K_WALL; K_WALL = max(K_WALL * 0.90, K_WALL_MIN)
def gui_d_plus(event):
    global D_WALL; D_WALL *= 1.10
def gui_d_minus(event):
    global D_WALL; D_WALL = max(0.0, D_WALL * 0.90)
def gui_snapshot(event): print_snapshot()

def gui_log_zwidth(event):   start_logging('zwidth')
def gui_log_forces(event):   start_logging('forces')
def gui_log_pass(event):     start_logging('passivity')
def gui_log_state(event):    start_logging('state')
def gui_log_all(event):      start_logging('all')
def gui_log_cancel(event):   cancel_logging()

btn_tare   = _add_button(X0, y, W, H, "Tare", gui_tare); y -= (H+GAP)
btn_auto   = _add_button(X0, y, W, H, f"AutoProbe: {'ON' if AUTO_PROBE else 'OFF'}", gui_autoprobe); y -= (H+GAP)
btn_kp     = _add_button(X0, y, W/2-0.005, H, "K+", gui_k_plus)
btn_km     = _add_button(X0+W/2+0.005, y, W/2-0.005, H, "K-", gui_k_minus); y -= (H+GAP)
btn_dp     = _add_button(X0, y, W/2-0.005, H, "D+", gui_d_plus)
btn_dm     = _add_button(X0+W/2+0.005, y, W/2-0.005, H, "D-", gui_d_minus); y -= (H+GAP)
btn_snap   = _add_button(X0, y, W, H, "Snapshot", gui_snapshot); y -= (H+GAP*1.8)

btn_lzw    = _add_button(X0, y, W, H, "Log Z-Width",   gui_log_zwidth); y_z = y; y -= (H+GAP)
btn_lfor   = _add_button(X0, y, W, H, "Log Forces",    gui_log_forces); y_f = y; y -= (H+GAP)
btn_lpas   = _add_button(X0, y, W, H, "Log Passivity", gui_log_pass);  y_p = y; y -= (H+GAP)
btn_lsta   = _add_button(X0, y, W, H, "Log State",     gui_log_state); y_s = y; y -= (H+GAP)
btn_lall   = _add_button(X0, y, W, H, "Log ALL",       gui_log_all);   y_a = y; y -= (H+GAP)
btn_cancel = _add_button(X0, y, W, H, "Cancel Log",    gui_log_cancel); y -= (H+GAP)

# Per-button countdown timers (right of each button)
def _timer_axis(ypos):
    ax_t = fig.add_axes([X0 + W + 0.01, ypos, 0.075, H])
    ax_t.axis('off')
    return ax_t.text(0.0, 0.5, "", transform=ax_t.transAxes,
                     va='center', color='tab:red', fontsize=9, fontweight='bold')
timer_texts = {
    'zwidth':   _timer_axis(y_z),
    'forces':   _timer_axis(y_f),
    'passivity':_timer_axis(y_p),
    'state':    _timer_axis(y_s),
    'all':      _timer_axis(y_a),
}

# ——— Animation update ———
def update(frame):
    global current_pos, current_vel
    global E_env, CONTACT_T0, LAST_FACE, K_WALL, D_WALL, K_PEAK, STABLE_T

    # Read user wrench (signed)
    F_user = filtered_force.copy()    # N
    M_user = filtered_moment.copy()   # N·m
    # derive forces from moments (pure z-offset handle)
    Fx_fromM =  (M_user[1] / HANDLE_HEIGHT_M)    # +My / h
    Fy_fromM = -(M_user[0] / HANDLE_HEIGHT_M)    # -Mx / h
    Fz_fromM =  0.0
    F_fromM_vec = np.array([Fx_fromM, Fy_fromM, Fz_fromM])

    F_mag  = float(np.linalg.norm(F_user))
    F_u    = F_user if F_mag >= THRESHOLD_N else np.zeros(3)

    # Update sensor readout in GUI
    sensor_text.set_text(
        f"F=[{F_user[0]:5.2f},{F_user[1]:5.2f},{F_user[2]:5.2f}] N  |F|={F_mag:5.2f}   "
        f"M=[{M_user[0]:5.3f},{M_user[1]:5.3f},{M_user[2]:5.3f}] N·m"
    )

    # Damping floor for passivity (discrete-time rule)
    D_floor = GAMMA_PASS * K_WALL * dt
    if D_WALL < D_floor: D_WALL = D_floor

    # Wall force + contact
    F_wall, in_contact, face = wall_contact_force(current_pos, current_vel, K_WALL, D_WALL)
    LAST_FACE = face
    F_wall_mag = float(np.linalg.norm(F_wall))

    # Center spring
    F_center = -K_STIFF * (current_pos - equilibrium)

    # Dynamics
    accel = (F_u + F_wall + F_center - B_COEFF * current_vel) / M
    current_vel += accel * dt
    new_pos = current_pos + current_vel * dt

    # Hard safety clamps
    new_pos[0] = np.clip(new_pos[0], X_HARD_MIN, X_HARD_MAX)
    new_pos[1] = np.clip(new_pos[1], Y_HARD_MIN, Y_HARD_MAX)
    new_pos[2] = np.clip(new_pos[2], Z_HARD_MIN, Z_HARD_MAX)
    current_pos[:] = new_pos

    # Passivity observer & K adaptation
    if in_contact:
        P_env = -float(np.dot(F_wall, current_vel))   # N·mm/s
        dE = max(0.0, P_env) * dt
        E_env = E_ENV_LEAK * E_env + dE

        if CONTACT_T0 is None:
            CONTACT_T0 = time.time(); STABLE_T = 0.0
        contact_dt = time.time() - CONTACT_T0

        if AUTO_PROBE and contact_dt > CONTACT_MIN_DT:
            if E_env > E_ENV_THRESH:
                K_WALL = max(K_WALL_MIN, K_WALL * K_DOWN_FACTOR)
                E_env = 0.0; STABLE_T = 0.0
            else:
                K_WALL = min(K_WALL_MAX_CAP, K_WALL + K_UP_RATE * dt)
                STABLE_T += dt
                if STABLE_T > 0.4:
                    if K_WALL > K_PEAK: K_PEAK = K_WALL
                    STABLE_T = 0.0
    else:
        P_env = 0.0
        E_env *= E_ENV_LEAK
        CONTACT_T0 = None; STABLE_T = 0.0

    # ——— Logging capture & countdown ———
    if LOG_ACTIVE:
        t = time.time() - t_start
        if LOG_TYPE == 'zwidth':
            LOG_DATA['t'].append(t); LOG_DATA['K_wall'].append(float(K_WALL))
            LOG_DATA['D_wall'].append(float(D_WALL)); LOG_DATA['E_env'].append(float(E_env))
            LOG_DATA['Fwall_mag'].append(float(F_wall_mag)); LOG_DATA['contact'].append(int(1 if in_contact else 0))
        elif LOG_TYPE == 'forces':
            LOG_DATA['t'].append(t)
            # measured forces + magnitude
            LOG_DATA['Fx'].append(float(F_user[0])); LOG_DATA['Fy'].append(float(F_user[1])); LOG_DATA['Fz'].append(float(F_user[2]))
            LOG_DATA['Fuser_mag'].append(float(F_mag))
            # derived from moments + magnitude
            LOG_DATA['Fx_fromM'].append(float(Fx_fromM)); LOG_DATA['Fy_fromM'].append(float(Fy_fromM)); LOG_DATA['Fz_fromM'].append(float(Fz_fromM))
            LOG_DATA['F_fromM_mag'].append(float(np.linalg.norm(F_fromM_vec)))
            # wall force + torques
            LOG_DATA['Fwall_x'].append(float(F_wall[0])); LOG_DATA['Fwall_y'].append(float(F_wall[1])); LOG_DATA['Fwall_z'].append(float(F_wall[2]))
            LOG_DATA['Fwall_mag'].append(float(F_wall_mag))
            LOG_DATA['Mx'].append(float(M_user[0])); LOG_DATA['My'].append(float(M_user[1])); LOG_DATA['Mz'].append(float(M_user[2]))
            # speed
            LOG_DATA['v_mag'].append(float(np.linalg.norm(current_vel)))
        elif LOG_TYPE == 'passivity':
            LOG_DATA['t'].append(t); LOG_DATA['P_env'].append(float(P_env)); LOG_DATA['E_env'].append(float(E_env))
        elif LOG_TYPE == 'state':
            LOG_DATA['t'].append(t); LOG_DATA['x'].append(float(current_pos[0])); LOG_DATA['y'].append(float(current_pos[1])); LOG_DATA['z'].append(float(current_pos[2]))
        elif LOG_TYPE == 'all':
            LOG_DATA['t'].append(t)
            LOG_DATA['K_wall'].append(float(K_WALL)); LOG_DATA['D_wall'].append(float(D_WALL))
            LOG_DATA['E_env'].append(float(E_env)); LOG_DATA['Fwall_mag'].append(float(F_wall_mag)); LOG_DATA['contact'].append(int(1 if in_contact else 0))
            LOG_DATA['Fx'].append(float(F_user[0])); LOG_DATA['Fy'].append(float(F_user[1])); LOG_DATA['Fz'].append(float(F_user[2])); LOG_DATA['Fuser_mag'].append(float(F_mag))
            LOG_DATA['Fx_fromM'].append(float(Fx_fromM)); LOG_DATA['Fy_fromM'].append(float(Fy_fromM)); LOG_DATA['Fz_fromM'].append(float(Fz_fromM)); LOG_DATA['F_fromM_mag'].append(float(np.linalg.norm(F_fromM_vec)))
            LOG_DATA['Fwall_x'].append(float(F_wall[0])); LOG_DATA['Fwall_y'].append(float(F_wall[1])); LOG_DATA['Fwall_z'].append(float(F_wall[2]))
            LOG_DATA['Mx'].append(float(M_user[0])); LOG_DATA['My'].append(float(M_user[1])); LOG_DATA['Mz'].append(float(M_user[2]))
            LOG_DATA['v_mag'].append(float(np.linalg.norm(current_vel)))
            LOG_DATA['P_env'].append(float(P_env)); LOG_DATA['x'].append(float(current_pos[0])); LOG_DATA['y'].append(float(current_pos[1])); LOG_DATA['z'].append(float(current_pos[2]))
            LOG_DATA['face'].append(LAST_FACE)

        remaining = max(0.0, LOG_DUR - (time.time() - LOG_T0))
        for k in timer_texts: timer_texts[k].set_text("")
        if LOG_TYPE in timer_texts:
            timer_texts[LOG_TYPE].set_text(f"{remaining:4.1f}s")

        maybe_finish_logging()
    else:
        for k in timer_texts: timer_texts[k].set_text("")

    if LOG_QUEUE:
        plot_completed_logs()

    # ——— Kinematics & Draw ———
    try:
        thetas = inverse_kinematics(*current_pos)
    except ValueError:
        # out of reach; skip drawing update
        return upper_arms + fore_arms + [eff_outline, eff_fill, hud_text, force_user_line, force_wall_line]

    elbows, conns = [], []
    for i in range(3):
        Bi = BASE_PTS[i]
        t = math.radians(thetas[i])
        local_elbow = np.array([0, -rf*math.cos(t), -rf*math.sin(t)])
        Ei = Bi + (RZ[i] @ local_elbow); elbows.append(Ei)
        Ci = current_pos + EFF_OFFS[i]; conns.append(Ci)

    for i in range(3):
        upper_arms[i].set_data([BASE_PTS[i][0], elbows[i][0]],
                               [BASE_PTS[i][1], elbows[i][1]])
        upper_arms[i].set_3d_properties([BASE_PTS[i][2], elbows[i][2]])
        fore_arms[i].set_data([elbows[i][0], conns[i][0]],
                              [elbows[i][1], conns[i][1]])
        fore_arms[i].set_3d_properties([elbows[i][2], conns[i][2]])

    # effector polygon (filled) + dashed outline
    tri = [(conns[i][0], conns[i][1], conns[i][2]) for i in [0,1,2]]
    eff_fill.set_verts([tri])
    ex = [p[0] for p in conns] + [conns[0][0]]
    ey = [p[1] for p in conns] + [conns[0][1]]
    ez = [p[2] for p in conns] + [conns[0][2]]
    eff_outline.set_data(ex, ey); eff_outline.set_3d_properties(ez)

    # Force vectors (scale + cap)
    def _scaled_segment(Fvec):
        seg = Fvec * FORCE_ARROW_SCALE
        n = float(np.linalg.norm(seg))
        if n > FORCE_ARROW_MAX and n > 1e-9:
            seg *= (FORCE_ARROW_MAX / n)
        return seg

    seg_user = _scaled_segment(F_user)
    seg_wall = _scaled_segment(F_wall)

    # user force from end-effector center
    p0 = current_pos
    pu = p0 + seg_user
    pw = p0 + seg_wall
    force_user_line.set_data([p0[0], pu[0]], [p0[1], pu[1]])
    force_user_line.set_3d_properties([p0[2], pu[2]])
    # wall force (if any)
    force_wall_line.set_data([p0[0], pw[0]], [p0[1], pw[1]])
    force_wall_line.set_3d_properties([p0[2], pw[2]])

    # HUD
    fx, fy, fz = F_user; vnorm = float(np.linalg.norm(current_vel))
    hud_text.set_text(
        f"F_sensor=[{fx:6.2f},{fy:6.2f},{fz:6.2f}] N  |F|={F_mag:6.2f}  "
        f"M=[{M_user[0]:6.3f},{M_user[1]:6.3f},{M_user[2]:6.3f}] N·m  "
        f"F_fromM=[{Fx_fromM:6.2f},{Fy_fromM:6.2f},{Fz_fromM:6.2f}] N\n"
        f"Z-Width  K_wall={K_WALL:.4f} N/mm  D_wall={D_WALL:.4f} N·s/mm  "
        f"K_peak={K_PEAK:.4f}  E_env={E_env:.3f}  face={LAST_FACE}  AUTO={'ON' if AUTO_PROBE else 'OFF'}   "
        f"(vector scale {FORCE_ARROW_SCALE:.0f} mm/N)"
    )

    return upper_arms + fore_arms + [eff_outline, eff_fill, hud_text, force_user_line, force_wall_line]

ani = animation.FuncAnimation(fig, update, interval=int((1/30)*1000), blit=True)
plt.show()
exit_event.set()
