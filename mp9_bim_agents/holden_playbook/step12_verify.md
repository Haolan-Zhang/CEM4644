# Step 12: Verify the model

This step creates nothing. It counts the elements and compares them with the expected numbers.

Tool: `send_code_to_revit`. Send this code exactly, as the `code` argument:

```csharp
var lv = new FilteredElementCollector(document).OfClass(typeof(Level)).Cast<Level>().ToList();
var l1 = lv.FirstOrDefault(l => l.Name == "Level 1");
var l2 = lv.FirstOrDefault(l => l.Name == "Level 2");
if (l1 == null || l2 == null) return "ERROR: Level 1 or Level 2 not found.";
Func<Level, int> walls = L => new FilteredElementCollector(document).OfClass(typeof(Wall)).Cast<Wall>().Count(w => w.LevelId == L.Id);
Func<Level, int> doors = L => new FilteredElementCollector(document).OfCategory(BuiltInCategory.OST_Doors).WhereElementIsNotElementType().Cast<FamilyInstance>().Count(d => d.LevelId == L.Id);
Func<Level, int> rooms = L => new FilteredElementCollector(document).OfCategory(BuiltInCategory.OST_Rooms).Cast<SpatialElement>().Count(r => r.LevelId == L.Id && r.Area > 0);
Func<Level, int> openRooms = L => new FilteredElementCollector(document).OfCategory(BuiltInCategory.OST_Rooms).Cast<SpatialElement>().Count(r => r.LevelId == L.Id && r.Area <= 0);
Func<Level, int> fixtures = L => new FilteredElementCollector(document).OfCategory(BuiltInCategory.OST_PlumbingFixtures).WhereElementIsNotElementType().Cast<FamilyInstance>().Count(f => f.LevelId == L.Id);
Func<Level, int> floors = L => new FilteredElementCollector(document).OfClass(typeof(Floor)).Cast<Floor>().Count(f => f.LevelId == L.Id);
var gridNames = new[] { "X2", "X3", "X4", "X5", "X5.5", "X6", "X7", "X8", "X9", "X10" };
int grids = new FilteredElementCollector(document).OfClass(typeof(Grid)).Cast<Grid>().Count(g => gridNames.Contains(g.Name));
var warnings = document.GetWarnings().Select(w => w.GetDescriptionText()).ToList();
var checks = new List<Tuple<string, int, int>> {
    Tuple.Create("Grids", grids, 10),
    Tuple.Create("Level 1 walls", walls(l1), 42), Tuple.Create("Level 1 doors", doors(l1), 26), Tuple.Create("Level 1 rooms (enclosed)", rooms(l1), 24),
    Tuple.Create("Level 1 rooms NOT enclosed", openRooms(l1), 0), Tuple.Create("Level 1 fixtures", fixtures(l1), 3), Tuple.Create("Level 1 floors", floors(l1), 1),
    Tuple.Create("Level 2 walls", walls(l2), 50), Tuple.Create("Level 2 doors", doors(l2), 29), Tuple.Create("Level 2 rooms (enclosed)", rooms(l2), 37),
    Tuple.Create("Level 2 rooms NOT enclosed", openRooms(l2), 0), Tuple.Create("Level 2 fixtures", fixtures(l2), 2), Tuple.Create("Level 2 floors", floors(l2), 1),
    Tuple.Create("Revit warnings", warnings.Count, 0)
};
var lines = checks.Select(c => (c.Item2 == c.Item3 ? "OK       " : "MISMATCH ") + c.Item1 + ": " + c.Item2 + " (expected " + c.Item3 + ")").ToList();
foreach (var w in warnings.Distinct()) lines.Add("WARNING TEXT: " + w);
bool pass = checks.All(c => c.Item2 == c.Item3);
return (pass ? "OK: ALL CHECKS PASSED" : "CHECK: some numbers differ") + "\n" + string.Join("\n", lines);
```

## Expected result

```
OK: ALL CHECKS PASSED
OK       Grids: 10 (expected 10)
OK       Level 1 walls: 42 (expected 42)
...
OK       Revit warnings: 0 (expected 0)
```

Show the full result to the human. The model is finished.

- If any line says `MISMATCH`, tell the human which lines.
- Fixtures can be 0 if step 4 could not find the family template; that is acceptable.
- Tell the human to save the project in Revit (Ctrl+S).
