"""
Urban PE Scenario Configuration
================================
Easily configure different urban propagation scenarios.
Modify the parameters below to generate different test cases.
"""

# ==========================================
# Basic Parameters
# ==========================================
f = 900e6  # frequency (Hz)
c = 3e8
lam = c / f
k0 = 2 * np.pi / lam

# Grid dimensions
Nx = 100
Ny = 128
Nz = 80

# Domain size (m)
Lx = 1000.0
Ly = 160.0
Lz = 80.0

dx = Lx / Nx
dy = Ly / Ny
dz = Lz / Nz

# ==========================================
# Antenna Configuration
# ==========================================
tx_y = 0.0      # antenna y-position (m)
tx_z = 15.0     # antenna height (m)
sigma_y = 3.0   # antenna beamwidth in y (m)
sigma_z = 2.0   # antenna beamwidth in z (m)

# ==========================================
# Building Configuration
# ==========================================
# Define buildings as a list of dictionaries
# Each building:
#   y0, y1: y-range (m) - [start, end]
#   height: building height (m)
#   x_start: x-position where building starts (m)
#   x_end: x-position where building ends (m)
#   eps_r: relative permittivity
#   sigma: conductivity (S/m)
#   name: label

# Scenario 1: Street canyon (2 buildings)
scenario_street_canyon = [
    {
        'y0': -50.0, 'y1': -10.0,
        'height': 30.0,
        'x_start': 100.0, 'x_end': 900.0,
        'eps_r': 5.0, 'sigma': 0.01,
        'name': 'Building 1 (concrete)'
    },
    {
        'y0': 10.0, 'y1': 50.0,
        'height': 25.0,
        'x_start': 100.0, 'x_end': 900.0,
        'eps_r': 4.0, 'sigma': 0.005,
        'name': 'Building 2 (brick)'
    },
]

# Scenario 2: Asymmetric street (different widths)
scenario_asymmetric = [
    {
        'y0': -60.0, 'y1': -15.0,
        'height': 35.0,
        'x_start': 50.0, 'x_end': 950.0,
        'eps_r': 6.0, 'sigma': 0.01,
        'name': 'Building 1 (tall)'
    },
    {
        'y0': 10.0, 'y1': 35.0,
        'height': 20.0,
        'x_start': 100.0, 'x_end': 900.0,
        'eps_r': 4.0, 'sigma': 0.005,
        'name': 'Building 2 (short)'
    },
]

# Scenario 3: Gap in building (LOS window)
scenario_gap = [
    {
        'y0': -50.0, 'y1': -10.0,
        'height': 30.0,
        'x_start': 100.0, 'x_end': 400.0,
        'eps_r': 5.0, 'sigma': 0.01,
        'name': 'Building 1a'
    },
    {
        'y0': -50.0, 'y1': -10.0,
        'height': 30.0,
        'x_start': 600.0, 'x_end': 900.0,
        'eps_r': 5.0, 'sigma': 0.01,
        'name': 'Building 1b'
    },
    {
        'y0': 10.0, 'y1': 50.0,
        'height': 25.0,
        'x_start': 100.0, 'x_end': 900.0,
        'eps_r': 4.0, 'sigma': 0.005,
        'name': 'Building 2'
    },
]

# Scenario 4: Dense urban (3 buildings on one side)
scenario_dense = [
    {
        'y0': -50.0, 'y1': -10.0,
        'height': 30.0,
        'x_start': 100.0, 'x_end': 900.0,
        'eps_r': 5.0, 'sigma': 0.01,
        'name': 'Building 1'
    },
    {
        'y0': 10.0, 'y1': 25.0,
        'height': 20.0,
        'x_start': 100.0, 'x_end': 500.0,
        'eps_r': 4.0, 'sigma': 0.005,
        'name': 'Building 2a'
    },
    {
        'y0': 10.0, 'y1': 30.0,
        'height': 35.0,
        'x_start': 500.0, 'x_end': 900.0,
        'eps_r': 6.0, 'sigma': 0.015,
        'name': 'Building 2b'
    },
]

# ==========================================
# Ground Material
# ==========================================
ground_eps_r = 15.0
ground_sigma = 0.005  # S/m

# ==========================================
# Select Scenario
# ==========================================
# Choose which scenario to run:
#   scenario_street_canyon   - 2 buildings, symmetric street
#   scenario_asymmetric      - 2 buildings, asymmetric heights
#   scenario_gap             - 3 buildings, gap in left building
#   scenario_dense           - 3 buildings, dense urban

SELECTED_SCENARIO = scenario_street_canyon

print(f"Selected scenario: {len(SELECTED_SCENARIO)} buildings")
for i, b in enumerate(SELECTED_SCENARIO):
    print(f"  {i+1}. {b['name']}: y=[{b['y0']:.0f},{b['y1']:.0f}]m, "
          f"height={b['height']:.0f}m, x=[{b['x_start']:.0f},{b['x_end']:.0f}]m")
