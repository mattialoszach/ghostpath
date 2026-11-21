# GhostPath

- Keep robot control, capture, and dataset writing in `src/ghostpath`. The React workspace lives in `frontend`.
- Set up with `uv sync --locked --extra hardware --extra dev` and `npm --prefix frontend ci`.
- Verify changes with `uv run --no-sync pytest -q`, `uv run --no-sync ruff check src tests`, and `npm --prefix frontend run build`.
- Use the simulator first. Confirm the arm is clear and supervised before any physical motion test. Preserve torque, tracking, and stop safeguards.
- Preserve raw teaching data and keep measured observations separate from commanded actions.
- Make a focused commit after each coherent, verified change. Keep local datasets, calibration, secrets, and build artifacts out of Git.
