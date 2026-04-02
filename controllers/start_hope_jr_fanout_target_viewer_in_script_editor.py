import importlib.util
import sys

module_name = "hope_jr_fanout_target_viewer"
path = "/home/viaan/vivy_hopejr_sim/controllers/hope_jr_fanout_target_viewer.py"

old_module = sys.modules.get(module_name)
if old_module is not None and hasattr(old_module, "stop_script_editor_loop"):
    try:
        old_module.stop_script_editor_loop()
    except Exception as exc:
        print(f"old stop error: {exc}")

if module_name in sys.modules:
    del sys.modules[module_name]

spec = importlib.util.spec_from_file_location(module_name, path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

module.start_script_editor_loop()
