# Step 6: Level 1 doors

Places the 26 doors of Level 1 into the walls from step 5. Each data row is `MARK TYPE x y toward_x toward_y`:

- `x y` is the door position on its wall;
- `toward_x toward_y` is a point in the room the door swings into;
- `TYPE` is `S` (3'-0"×7'-0"), `S8` (3'-0"×8'-0"), `D6` (pair, 6'-0"×7'-0"), or `D88` (pair of 3'-8"×7'-6" leaves).

Tool: `send_code_to_revit`. Send this code exactly, as the `code` argument:

```csharp
var D = new Func<string, double>(s => double.Parse(s, System.Globalization.CultureInfo.InvariantCulture));
var L = new FilteredElementCollector(document).OfClass(typeof(Level)).Cast<Level>().FirstOrDefault(l => l.Name == "Level 1");
if (L == null) return "ERROR: Level 1 not found. Run step 2 first.";
int already = new FilteredElementCollector(document).OfCategory(BuiltInCategory.OST_Doors).WhereElementIsNotElementType().Cast<FamilyInstance>().Count(d => d.LevelId == L.Id);
if (already > 0) return "SKIPPED: Level 1 already has " + already + " doors.";
var walls = new FilteredElementCollector(document).OfClass(typeof(Wall)).Cast<Wall>().Where(w => w.LevelId == L.Id && w.WallType.Kind == WallKind.Basic).ToList();
if (walls.Count == 0) return "ERROR: Level 1 has no walls. Run step 5 first.";
var syms = new FilteredElementCollector(document).OfClass(typeof(FamilySymbol)).OfCategory(BuiltInCategory.OST_Doors).Cast<FamilySymbol>().ToList();
var typeNames = new Dictionary<string, string> { { "S", "HH Single 36x84" }, { "S8", "HH Single 36x96" }, { "D6", "HH Pair 72x84" }, { "D88", "HH Pair 88x90" } };

string data = @"
191 S 4.76 26.05 4 30|185 S 39.5 26.05 35 30|195 S 0.46 23.6 -5 23|196 S 0.46 12.4 -5 14|194 S 0.46 7.6 -5 5|190A S 24.45 20.97 20 15|190B S 24.49 6.56 20 12|
184 S 35.49 6.56 33 12|182 S 41.65 6.56 44 12|178 S 52.55 6.56 55 12|185A S 43.77 32.05 48 30|A81 S 58.34 27.24 55 30|A71B S 74.02 36.42 60 38|A71A S8 64.32 45.25 64 40|
A72 S 82.11 42.56 88 41|A76 S 83.65 30.5 78 30|160E S 93.0 20.01 90 15|160F S 83.65 10.3 78 11|160A S 93.72 8.24 90 3|160B S 98.29 8.24 102 3|160C S 109.94 8.24 114 3|
160G S 121.25 23.04 115 23|160D S 108.93 45.25 109 40|A78B D6 75.56 8.24 76 12|A78A D6 78.54 0.75 77 4|A70A D88 78.05 45.25 78 40";

var problems = new List<string>();
int n = 0;
foreach (var row in data.Split(new[] { '|' }, StringSplitOptions.RemoveEmptyEntries))
{
    var p = row.Trim().Split(' ');
    var sym = syms.FirstOrDefault(f => f.Name == typeNames[p[1]]);
    if (sym == null) throw new Exception("ERROR: door type " + typeNames[p[1]] + " is missing. Run step 4 first. Nothing was created.");
    if (!sym.IsActive) { sym.Activate(); document.Regenerate(); }
    var pt = new XYZ(D(p[2]), D(p[3]), L.Elevation);
    Wall host = null; XYZ at = null; double best = 0.4;
    foreach (var w in walls)
    {
        var c = (w.Location as LocationCurve).Curve;
        var r = c.Project(pt);
        if (r == null) continue;
        if (r.Distance < best && r.XYZPoint.DistanceTo(c.GetEndPoint(0)) > 0.5 && r.XYZPoint.DistanceTo(c.GetEndPoint(1)) > 0.5)
        { best = r.Distance; host = w; at = r.XYZPoint; }
    }
    if (host == null) { problems.Add("no wall found for door " + p[0]); continue; }
    var fi = document.Create.NewFamilyInstance(at, sym, host, L, Autodesk.Revit.DB.Structure.StructuralType.NonStructural);
    fi.get_Parameter(BuiltInParameter.ALL_MODEL_MARK).Set(p[0]);
    document.Regenerate();
    var toward = new XYZ(D(p[4]), D(p[5]), L.Elevation) - at;
    if (fi.FacingOrientation.DotProduct(toward) < 0) fi.flipFacing();
    n++;
}
return (problems.Count == 0 ? "OK: created " : "OK with problems: created ") + n + " doors on Level 1. " + string.Join("; ", problems);
```

## Expected result

`OK: created 26 doors on Level 1.` (or `SKIPPED: Level 1 already has 26 doors.`).

If it says `OK with problems`, tell the human which doors are listed, then continue anyway.

Then go to step 7.
