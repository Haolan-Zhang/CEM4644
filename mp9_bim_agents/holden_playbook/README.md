# Holden Hall South Wing: Revit MCP Playbook

This playbook builds a simplified BIM model of Holden Hall, South Wing, Levels 1 and 2 (Virginia Tech), from the PDF `Holden_Hall_South_Wing_L1-L2_for_Agent.pdf`, using the Revit MCP server ([mcp-servers-for-revit](https://github.com/mcp-servers-for-revit/mcp-servers-for-revit)).

Every number in it has already been measured from the PDF. The AI's job is only to run the steps in order and check each result.

Result when finished:

| | Level 1 | Level 2 | Total |
|---|---|---|---|
| Walls | 42 | 50 | 92 |
| Doors | 26 | 29 | 55 |
| Rooms (named and numbered) | 24 | 37 | 61 |
| Plumbing fixtures | 3 | 2 | 5 |
| Floors | 1 | 1 | 2 |

The model also has 2 levels, 10 grids (X2 to X10 and X5.5), 16 wall types named after the partition marks, room tags on both plans, and a 3D view.

---

## Part A: for the human (do this before starting the AI)

1. Open Revit 2022 or newer, start a **new, empty project** from any architectural template, and open a floor plan view.
2. Make sure the mcp-servers-for-revit add-in is installed. On its ribbon tab, click **Settings**, enable at least `send_code_to_revit`, and click **Save**.
3. In your AI client, add the MCP server. On Windows, the command is `cmd /c npx -y mcp-server-for-revit`; see the project README for each client.
4. Give the AI this folder and tell it: **"Follow holden_playbook/README.md exactly."**
5. Stay near Revit. If Revit shows a warning dialog, click **OK**. The AI cannot click it, and Revit ignores the AI until the dialog is closed.

---

## Part B: rules for the AI (read all of them before step 1)

1. **Do the steps in order, one at a time.** Open one step file, run it, read the result, then open the next file. Do not read ahead and do not combine steps.
2. **Every step is exactly one call to the tool `send_code_to_revit`.** Put the code block from the step file, everything between the ` ```csharp ` line and the closing ` ``` ` line, into the `code` argument.
3. **Copy the code exactly, character for character.** Do not shorten it, reformat it, rename anything, translate it, add `using` lines, wrap it in a method, or "improve" it. Do not pass the `parameters` argument.
4. **Never change or compute numbers.** All coordinates are already correct. Do not read the PDF to adjust them.
5. **Do not use the other Revit tools** (`create_line_based_element`, `create_room`, `create_level`, ...) to build this model. They use different units and cannot do several of these steps.
6. **Read the result of every step:**
   - Starts with `OK`: go to the next step.
   - Starts with `SKIPPED`: that step was already done. Go to the next step.
   - Contains `ERROR`: stop, tell the human the exact message, and do what it says. Nothing from that step was saved.
   - A compile error (for example `代码编译错误` or `Line 12: ...`): you changed the code. Copy it again exactly and resend it.
   - A timeout (`超时`, `timeout`): **do not send the next step.** Ask the human: "Is a dialog open in Revit? Please click OK." Wait until they answer, then send **the same step again**. This is safe because every step skips work that already exists.
7. When step 12 is done, show its full result to the human.

Chinese words you may see in tool results: `执行失败` = failed, `超时` = timeout, `代码编译错误` = code compile error.

---

## Steps

| Step | File | What it does | Expected result |
|---|---|---|---|
| 1 | [step01_check_connection.md](step01_check_connection.md) | Confirms Revit is connected and the model is empty | `OK: connected ...` |
| 2 | [step02_levels_units_walltypes.md](step02_levels_units_walltypes.md) | Levels 1 and 2, feet-inch units, 16 wall types | `OK: levels, units and 16 wall types ready` |
| 3 | [step03_grids.md](step03_grids.md) | Grids X2 to X10 and X5.5 | `OK: created 10 grids` |
| 4 | [step04_door_types_and_fixtures.md](step04_door_types_and_fixtures.md) | 4 door types, toilet and sink families | `OK: door types and plumbing families ready` |
| 5 | [step05_level1_walls.md](step05_level1_walls.md) | Level 1 walls | `OK: created 42 walls on Level 1` |
| 6 | [step06_level1_doors.md](step06_level1_doors.md) | Level 1 doors | `OK: created 26 doors on Level 1` |
| 7 | [step07_level1_rooms_fixtures_floor.md](step07_level1_rooms_fixtures_floor.md) | Level 1 rooms, toilet fixtures, floor | `OK: ... 24 rooms ...` |
| 8 | [step08_level2_walls.md](step08_level2_walls.md) | Level 2 walls | `OK: created 50 walls on Level 2` |
| 9 | [step09_level2_doors.md](step09_level2_doors.md) | Level 2 doors | `OK: created 29 doors on Level 2` |
| 10 | [step10_level2_rooms_fixtures_floor.md](step10_level2_rooms_fixtures_floor.md) | Level 2 rooms, toilet fixtures, floor | `OK: ... 37 rooms ...` |
| 11 | [step11_tags_and_views.md](step11_tags_and_views.md) | Room tags, cropped plans, 3D view | `OK: ...` |
| 12 | [step12_verify.md](step12_verify.md) | Counts everything and compares with the expected numbers | Every line starts with `OK` |

---

## Part C: background for the instructor

### Coordinate system used in all steps

- Units are **decimal feet**, the Revit API's internal unit (`11.8333` = 11'-10").
- **x = 0 is grid X2**, increasing east. Grids: X2 0, X3 14, X4 28, X5 42, X5.5 48.833, X6 56, X7 70, X8 84, X9 96, X10 108.
- **y = 0 is the outside face of the south exterior wall** (west part of the wing), increasing north. The north outside face is at y = 46.
- Exterior walls are 1'-6" thick and modeled on their centerline (0.75 ft inside the outside face). Interior walls are on their centerlines.
- Level 1 walls run from Level 1 to Level 2 (11'-10"). Level 2 walls are 11'-0" high.

### How the numbers were obtained

The PDF crops are vector drawings, so the measurements were read from the drawing data, not estimated from pixels:

1. **Scale.** The grid bubble labels give the scale: Level 1 is 8.194 PDF points per foot (X2 at x = 181.2 pt), and Level 2 is 8.256 pt/ft (X2 at x = 181.7 pt). X2 to X8 (84 ft) and X2 to X10 (108 ft) agree.
2. **Walls.** New partitions are drawn with a gray fill (the stud core). The gray fills were rasterized into a mask, and straight runs were extracted as wall centerlines with their thicknesses. The P13 furring line along the perimeter is the inside face of the exterior wall: it falls at x = −14.7 and x = 120.0 ft, which matches the PDF's outline (west face 16'-2" west of X2, east face 14'-0" east of X10).
3. **Doors and rooms.** The positions of the door tags (ovals) and room number boxes were read from the PDF text and converted to feet. Each door was placed on the wall under its tag; each room was placed at its number box.
4. **Wall types** come from the partition mark next to each wall, with the thicknesses given on page 5.
5. **Fixtures** come from the enlarged toilet plans on page 4: 2 water closets and 1 lavatory in W TLT A72, and 1 of each in UNIV TLT B72.

### Simplifications (on purpose)

- The north side between X7 and X8 has no exterior wall. Instead, a short partition holds the door pairs A70A and B70A.
- The south face steps 4 ft south at rooms 160A–160C (Level 1) and from FAC 8 eastward (Level 2), as drawn. The bay at 160B / 268 and the canopy at A78 are ignored.
- Open-plan areas (collab rooms, reception, break room, BSA, visitor room) are separated with room separation lines instead of walls. The three MSE COLLAB rooms have storefront glazing on grids X3, X4, X5.
- Revit 2026 ships without local plumbing content, so step 4 builds simple box-shaped water closet and lavatory families from the Generic Model template.
- Door 293 is placed 2 ft west of its drawn position. At the drawn position it overlaps a wall junction, and Revit shows a warning dialog that blocks the MCP connection.
- Left out: windows, ceilings, stair geometry, casework, columns, toilet partitions, door 260A (host wall unclear), and door 250B (outside the wing).

### Known problems and fixes

| Symptom | Cause | Fix |
|---|---|---|
| Every tool call times out | A Revit dialog is open | Click OK (or Close) in Revit, then rerun the same step |
| Step 4: `no door family is loaded` | The template has no doors | In Revit: Insert > Load Autodesk Family > Doors, load one single and one double door, then rerun step 4 |
| Step 4: `Generic Model family template not found` | Family templates not installed | Install the Revit content, or skip fixtures: steps 7 and 10 report the fixtures as missing but still work |
| `room not enclosed: ...` in step 7 or 10 | A wall is missing | Run step 12, find which wall count is off, and rerun the walls step for that level on an empty level |
| Wrong results after an AI "fixed" the code | The model edited the code | Undo in Revit (or start a new project), and make the AI copy the code exactly |
