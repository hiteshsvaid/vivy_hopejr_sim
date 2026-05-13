# Repository Guidelines

## Layout
- Core runtime: `controllers/`, `ui/`, `utils/`
- Robot asset: `joint_test.usda`
- Launch stage: `vivy_stage.usda`
- Temporary probes: `probes/`

## Teleop Source Of Truth
- The live architecture diagram in `/home/viaan/huggingface/lerobot/docs/vivy_teleop_target_flow.md` is the source of truth for Quest input, right arm, left arm, and head mapping.
- When changing controller behavior, keep that Mermaid diagram updated in the same change.
- Keep naming consistent with the diagram: `Quest input`, `right arm mapper`, `left arm mapper`, `HeadTeleopMapper`, `Right arm command`, `Left arm command`, and `Head command`.

## Style
- Python 3.10+, 4-space indentation, type hints preferred
- Ruff-friendly formatting, double quotes
- `snake_case` for functions/variables, `PascalCase` for classes
