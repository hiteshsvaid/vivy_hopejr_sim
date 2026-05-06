from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from pathlib import Path

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg


ASSET_ROOT = Path('/home/viaan/vivy_hopejr_sim/assets')
SRC_DIR = ASSET_ROOT / 'src'
USD_DIR = ASSET_ROOT / 'usd'

VISUAL_ONLY_PARTS = [
    'chest',
    'arm_v2-chest_clone',
    'arm_v2-body_torso_clone',
    'shoulder_support_bottom',
    'shoulder_support_top',
    'arm_v2-shoulder_support_bottom',
    'arm_v2-shoulder_support_top',
    'arm_v2-new_shoulder_bottom_right',
    'arm_v2-new_shoulder_top_right',
    'arm_v2-new_shoulder_bottom_left',
    'arm_v2-new_shoulder_top_left',
    'pitch_holder',
    'shoulder_pitch_servo_s85',
    'pitch_roller_clip_bottom',
    'pitch_roller_clip_left_side',
    'pitch_roller_clip_right_side',
    'pitch_roller_clip_up',
    'arm_v2-pitch_roller_clip_bottom',
    'arm_v2-pitch_roller_clip_left_side',
    'arm_v2-pitch_roller_clip_right_side',
    'arm_v2-pitch_roller_clip_up',
    'arm_v2-pitch_roller_clip_bottom_left',
    'arm_v2-pitch_roller_clip_left_side_left',
    'arm_v2-pitch_roller_clip_right_side_left',
    'arm_v2-pitch_roller_clip_up_left',
    'shoulder_yaw_servo_sts3215',
    'shoulder_yaw',
    'humreal_up_fixed',
    'humreal_down',
    'arm_v2-humreal_up',
    'arm_v2-humreal_down',
    'arm_v2-humreal_up_left',
    'arm_v2-humreal_down_left',
    'elbow_servo_sts3215',
    'upper_elbow_roll_servo_sts3215',
    'elbow_up',
    'elbow_down',
    'arm_v2-elbow_up',
    'arm_v2-elbow_down',
    'arm_v2-elbow_up_left',
    'arm_v2-elbow_down_left',
    'lower_elbow_inner',
    'lower_elbow_outer',
    'lower_elbow_side_l',
    'lower_elbow_side_r',
    'lower_elbow_roll_servo_sts3215',
    'arm_v2-lower_elbow_inner_body',
    'arm_v2-lower_elbow_outer_body',
    'arm_v2-lower_elbow_side_l_body',
    'arm_v2-lower_elbow_side_r_body',
    'arm_v2-lower_elbow_inner_modified_left',
    'arm_v2-lower_elbow_outer_body_left',
    'arm_v2-lower_elbow_side_l_body_left',
    'arm_v2-lower_elbow_side_r_body_left',
    'arm_v2-forearm_l_clone',
    'arm_v2-forearm_r_clone',
    'arm_v2-forearm_l_left',
    'arm_v2-forearm_r_left',
    'forearm_to_wrist_servo_sts3215',
    'wrist_servo_sts3215',
    'wrist_yaw_pitch',
    'arm_v2-wrist_yaw_pitch',
    'palm_base',
    'arm_v2-palm_base_clone',
    'arm_v2-wrist_servo_sts3215',
    'arm_v2-thumb_servo_sts3215',
    'arm_v2-index_servo_sts3215',
    'arm_v2-thumb_clone',
    'arm_v2-index_clone',
    'thumb_servo_sts3215',
    'thumb',
    'index_servo_sts3215',
    'index',
]


def main() -> None:
    for part in VISUAL_ONLY_PARTS:
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
