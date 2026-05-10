# Vivy Left Target Marker Investigation

This note captures the current state of the left/right teleop split so the next session can resume without rediscovering the same issues.

## Current status

- Right-arm teleop is working.
- Left Quest packets are present and coming from Quest.
- The left path is now split from the right path in the viewer and side panel.
- Grip is no longer used as a gate for teleop activation.
- The left and right event streams are meant to be labeled by source in the UI.

## What we verified

- `/tmp/hope_jr_quest_packets.ndjson` showed left packets with:
  - `left.is_tracked = true`
  - `left.thumbstick_click = true` during the press window
- `/tmp/hope_jr_fanout_target.json` contains both:
  - `normalized.right_hand`
  - `normalized.left_hand`
- The sim viewer reads left state from:
  - `normalized.left_hand`
  - fallback raw Quest fields under `parsed_message.left`

## Relevant files

- `controllers/quest_teleop_mapper.py`
- `controllers/vivy/vivy_target_viewer.py`
- `ui/vivy/vivy_side_panel.py`
- `ui/teleop_debug_visuals.py`
- `src/lerobot/robots/vivy/quest_teleop_bridge/server.py`

## Recent code changes

- Left mapper was added alongside the right mapper in the sim viewer.
- Left target marker support exists in `TeleopDebugVisuals`:
  - `LeftQuestMapped`
  - `LeftSimTarget`
  - `LeftSimTargetCross`
- The side panel event history now prefixes messages with hand source:
  - `Right: ...`
  - `Left: ...`
- `QuestTeleopMapper` no longer checks `grip_threshold` for activation.

## Important findings

- Grip should stay reserved for finger behavior, not for teleop gating.
- The left Quest data is not the blocker; it is present in the packet trace.
- If the left marker is missing, the next thing to verify is that the sim viewer is running with the latest patched code loaded.

## Next session checklist

1. Start a clean Vivy run.
2. Confirm the viewer reloads the patched `vivy_target_viewer.py`.
3. Confirm `LeftQuestMapped` and `LeftSimTarget` appear in `/World/JointTest/TeleopDebug`.
4. Confirm the EVENTS panel shows source-labeled entries for both hands.
5. If the left marker is still absent, inspect the live `/tmp/vivy_sim_write_debug.json` and `/tmp/hope_jr_fanout_target.json` state first.

