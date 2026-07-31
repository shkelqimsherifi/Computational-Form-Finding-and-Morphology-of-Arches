import Rhino
import Grasshopper as gh
import scriptcontext as sc
import System
import os
import csv

GUID_N       = "ea491f79-e3e4-44bf-ab5b-71517f8a1cfe"
GUID_G       = "4340db3b-6335-4ca5-b38f-3466c141cf62"
GUID_K       = "34612116-39ac-43b7-a468-2bacc248ec15"
GUID_LOAD    = "2e7cf682-43f0-4d9a-8616-2bd0bf5d61aa"
GUID_RESET   = "3a6bae32-a0ef-47f3-b862-c21874af1a03"
GUID_CONV    = "2e810dd0-d591-49e4-99c8-5931bcd40a1e"
GUID_GEOM    = "f87d8e1a-b58e-432d-9ae1-8cf653c35c15"

# Optional: GUID of the boolean toggle wired into this component's "run" input.
# If you set this, the script will auto-flip it off when the batch finishes,
# preventing an accidental full re-run. Leave as None if you don't have it /
# don't want that behavior (see SAFETY NOTE at bottom of file).
GUID_RUN     = None   # e.g. "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

OUT_FOLDER = r"D:\UT_IT\Conferenca\2026\Architecture\ProgramaticDesign\Computational_Form-Finding_of_Arches\testImages"

loadtypes = [0, 1, 2]
g_values  = [-0.1, -0.3, -0.5, -0.7]
k_values  = [10.0, 25.0, 50.0]
n_values  = [12, 24, 48]

MAX_WAIT_TICKS = 100     # per-combo convergence timeout = MAX_WAIT_TICKS * TICK_MS
SETTLE_TICKS   = 2       # forced idle ticks after reset before trusting convergence
TICK_MS = 150

doc = ghenv.Component.OnPingDocument()
STICKY_KEY = "arch_batch_state"


def find_obj(guid_str):
    target = System.Guid(guid_str)
    for obj in doc.Objects:
        if obj.InstanceGuid == target:
            return obj
    raise Exception("Component not found: " + guid_str)


def set_slider(slider_obj, value):
    slider_obj.SetSliderValue(System.Decimal(value))
    # Mark dirty only -- don't force a solve here. The reset toggle below
    # triggers the single consolidated recompute so all four inputs
    # (loadtype, g, k, n) land in the same solution.
    slider_obj.ExpireSolution(False)

def set_dropdown(list_obj, index):
    # GH_ValueList: selection is per-item (.Selected), there is no SelectedIndex.
    if hasattr(list_obj, 'ListItems'):
        for i, item in enumerate(list_obj.ListItems):
            item.Selected = (i == index)
        list_obj.ExpireSolution(False)
    else:
        # Fallback for a genuine dropdown-style object that does expose SelectedIndex
        list_obj.SelectedIndex = index
        list_obj.ExpireSolution(False)


def get_data_param(obj, output_index=0):
    if hasattr(obj, 'Params') and hasattr(obj.Params, 'Output'):
        return obj.Params.Output[output_index]
    else:
        return obj  # already a param (e.g. Boolean Toggle, Number Slider)


def is_converged():
    conv_comp = find_obj(GUID_CONV)
    conv_output = get_data_param(conv_comp)
    vals = [v.Value for v in conv_output.VolatileData.AllData(True)]
    return bool(vals) and all(vals)


def export_csv(filename):
    geom_comp = find_obj(GUID_GEOM)
    output_param = get_data_param(geom_comp)
    pts = []
    for branch in output_param.VolatileData.Branches:
        for item in branch:
            pt = item.Value
            pts.append((pt.X, pt.Y, pt.Z))
    path = os.path.join(OUT_FOLDER, filename)
    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        for p in pts:
            writer.writerow(p)


def build_combos():
    combos = []
    for lt in loadtypes:
        for g_val in g_values:
            for k_val in k_values:
                for n_val in n_values:
                    combos.append((lt, g_val, k_val, n_val))
    return combos


def schedule_next_tick():
    def callback(d):
        ghenv.Component.ExpireSolution(False)
    doc.ScheduleSolution(TICK_MS, callback)


def stop_run_toggle():
    """Best-effort: flip the external run toggle off so the batch can't
    silently restart on the next unrelated recompute."""
    if GUID_RUN is None:
        return
    try:
        run_toggle = find_obj(GUID_RUN)
        run_toggle.Value = False
        run_toggle.ExpireSolution(True)
    except Exception:
        pass


# --- Start a new batch on the rising edge of 'run' ---
# 'batch_done' guards against re-triggering while `run` stays True with no
# GUID_RUN available to flip it off programmatically (see SAFETY NOTE below).
if run and STICKY_KEY not in sc.sticky and not sc.sticky.get(STICKY_KEY + "_done", False):
    sc.sticky[STICKY_KEY] = {
        'combos': build_combos(),
        'idx': 0,
        'phase': 'reset_on',
        'wait_ticks': 0,
        'settle_ticks': 0,
        'log': []
    }
    sc.sticky[STICKY_KEY + "_done"] = False

if not run:
    # Rising edge required again next time -- clear the done-latch once the
    # user manually drops `run` back to False.
    sc.sticky[STICKY_KEY + "_done"] = False

state = sc.sticky.get(STICKY_KEY, None)
a = "idle"

if state is not None:
    combos = state['combos']
    idx = state['idx']

    if idx >= len(combos):
        a = "BATCH COMPLETE -- {0} runs".format(len(combos))
        for r in state['log']:
            print(r)
        if STICKY_KEY in sc.sticky:
            del sc.sticky[STICKY_KEY]
        sc.sticky[STICKY_KEY + "_done"] = True
        stop_run_toggle()
    else:
        lt, g_val, k_val, n_val = combos[idx]
        phase = state['phase']
        sc.sticky[STICKY_KEY] = state

        if phase == 'reset_on':
            set_dropdown(find_obj(GUID_LOAD), lt)
            set_slider(find_obj(GUID_G), g_val)
            set_slider(find_obj(GUID_K), k_val)
            set_slider(find_obj(GUID_N), n_val)
            reset_toggle = find_obj(GUID_RESET)
            reset_toggle.Value = True
            reset_toggle.ExpireSolution(True)   # single consolidated solve
            state['phase'] = 'reset_off'
            a = "combo {0}/{1}: reset ON".format(idx + 1, len(combos))
            schedule_next_tick()

        elif phase == 'reset_off':
            reset_toggle = find_obj(GUID_RESET)
            reset_toggle.Value = False
            reset_toggle.ExpireSolution(True)
            state['phase'] = 'settle'
            state['settle_ticks'] = 0
            a = "combo {0}/{1}: settling...".format(idx + 1, len(combos))
            schedule_next_tick()

        elif phase == 'settle':
            # Force a few idle ticks so the solver has actually run at least
            # once on the *new* inputs before we start trusting is_converged().
            # Prevents reading a stale True carried over from the previous combo.
            state['settle_ticks'] += 1
            if state['settle_ticks'] >= SETTLE_TICKS:
                state['phase'] = 'check_converge'
                state['wait_ticks'] = 0
            a = "combo {0}/{1}: settling ({2}/{3})".format(
                idx + 1, len(combos), state['settle_ticks'], SETTLE_TICKS)
            schedule_next_tick()

        elif phase == 'check_converge':
            if is_converged():
                fname = "arch_run{0}_{1}_g{2}_k{3}.csv".format(
                    n_val, lt, round(g_val, 3), round(k_val, 3))
                export_csv(fname)
                state['log'].append((fname, "OK"))
                state['idx'] += 1
                state['phase'] = 'reset_on'
                a = "combo {0}/{1}: CONVERGED, exported".format(idx + 1, len(combos))
            else:
                state['wait_ticks'] += 1
                if state['wait_ticks'] > MAX_WAIT_TICKS:
                    fname = "arch_run{0}_{1}_g{2}_k{3}.csv".format(
                        n_val, lt, round(g_val, 3), round(k_val, 3))
                    state['log'].append((fname, "DID NOT CONVERGE"))
                    state['idx'] += 1
                    state['phase'] = 'reset_on'
                    a = "combo {0}/{1}: TIMED OUT".format(idx + 1, len(combos))
                else:
                    a = "combo {0}/{1}: waiting ({2}/{3})".format(
                        idx + 1, len(combos), state['wait_ticks'], MAX_WAIT_TICKS)
            schedule_next_tick()