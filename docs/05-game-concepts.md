# Game Concepts & LockSmith Rules

## The Game Model

Every ARC-AGI-3 game follows the same interface:

1. Agent sends `RESET` → gets initial frame.
2. Agent sends an action → gets a new frame.
3. Repeat until `GameState.WIN`, `GameState.GAME_OVER`, or max actions hit.

### Frame Structure

A frame's `frame` field is `list[list[list[int]]]`:
- Outermost list: multiple grids (usually 1, but can be animation frames).
- Middle list: rows (64 rows).
- Inner list: columns (64 columns).
- Each value: integer 0–15 (a color index).

### GameState Lifecycle

```
NOT_PLAYED → (RESET) → NOT_FINISHED → ... → WIN or GAME_OVER
                              ↑                      │
                              └──────── (RESET) ─────┘
```

### Color Palette

The 16 colors (used across most agents):

| Index | Color | Hex | Common Meaning (LockSmith) |
|-------|-------|-----|----------------------------|
| 0 | White | #FFFFFF | Transparent / empty |
| 1 | Off-white | #CCCCCC | |
| 2 | Neutral light | #999999 | Color rotator indicator |
| 3 | Neutral | #666666 | |
| 4 | Off-black | #333333 | Player body |
| 5 | Black | #000000 | |
| 6 | Magenta/Blue | #E53AA3 / #0000FF | Energy pills / refiller |
| 7 | Pink/Yellow | #FF7BCC / #FFFF00 | Rotator interior |
| 8 | Red/Orange | #F93C31 / #FFA500 | Walkable floor / used energy |
| 9 | Blue/Purple | #1E93FF / #800080 | Rotator border |
| 10 | Blue light/White | #88D8F1 / #FFFFFF | Walls |
| 11 | Yellow/Gray | #FFDC00 / #808080 | Door border |
| 12 | Orange | #FF851B | Player head (top) |
| 13 | Maroon | #921231 | |
| 14 | Green | #4FCC30 | |
| 15 | Purple | #A356D6 | |

Note: Different agents use slightly different palettes. The official ARC palette and the various agent rendering palettes don't fully agree — be aware of this when interpreting agent output.

## LockSmith Game Rules (from GuidedLLM + LangGraphThinking prompts)

**Objective:** Navigate a maze, find the correct key, and exit through the door. 6 levels total.

**Actions in LockSmith:**
- ACTION1 = Move Up (W)
- ACTION2 = Move Down (S)
- ACTION3 = Move Left (A)
- ACTION4 = Move Right (D)
- ACTION5 = Enter/Spacebar/Delete (does nothing in this game)
- ACTION6 = Click (does nothing in this game)

**Game Objects:**
- **Player:** 4×4 sprite (blue body + orange head). Moves 4 grid cells per action.
- **Walls:** INT<10>, impassable.
- **Floor:** INT<8>, walkable.
- **Door:** 4×4 with INT<11> border, contains a half-size key pattern inside.
- **Key:** Shown in bottom-left corner (6×6). Must match the door pattern.
- **Shape Rotator:** 4×4 with INT<9> border and INT<4> in top-left. Touch to rotate key shape.
- **Color Rotator:** 4×4 with INT<9> border and INT<2> in bottom-left. Touch to rotate key color.
- **Energy Pills:** 2×2 of INT<6>. Touch to refill energy.
- **Energy Bar:** Row 3 of the frame. INT<6> = unused energy, INT<8> = used energy. 25 total per life.
- **Lives:** 3 lives, shown as 2×2 INT<2> squares in top-right area.

**Strategy:**
1. Navigate to rotator(s).
2. Rotate key shape and color until it matches the pattern inside the door (to rotate more than once, move away 1 space then back).
3. Navigate to the door and touch it.
4. Watch energy — touch energy pills to refill. Running out = lose a life + level restart.
5. Complete all 6 levels to WIN. Level indicator in row 62 of the grid.

## Local vs. Online Environments

The `arc-agi` SDK supports two modes:
- **Local:** Game environments run on your machine (stored in `environment_files/`). Default mode.
- **Online:** Game environments run on ARC servers. Set `ONLINE_ONLY=True` in `.env`.

Local is faster for development. Online is required for official scoring.
