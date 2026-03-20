from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from pathlib import Path

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg


ASSET_ROOT = Path('/home/viaan/vivy_hopejr_sim/assets')
SRC_DIR = ASSET_ROOT / 'src'
USD_DIR = ASSET_ROOT / 'usd'

MOVING_PARTS = [
    'pitch_roller_clip_bottom',
    'pitch_roller_clip_left_side',
    'pitch_roller_clip_right_side',
    'pitch_roller_clip_up',
    'shoulder_yaw_servo_sts3215',
    'shoulder_yaw',
    'humreal_up_fixed',
    'humreal_down',
]


def main() -> None:
    for part in MOVING_PARTS:
        cfg = MeshConverterCfg(
            asset_path=str(SRC_DIR / f'{part}.stl'),
            usd_dir=str(USD_DIR),
            usd_file_name=f'{part}_visual.usd',
            force_usd_conversion=True,
            make_instanceable=False,
            collision_props=None,
        )
        converter = MeshConverter(cfg)
        print(f'generated {converter.usd_path}')

    simulation_app.close()


if __name__ == '__main__':
    main()
