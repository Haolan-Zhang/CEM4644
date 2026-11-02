# Step 9: Level 2 doors

Places the 29 doors of Level 2. The row format is the same as in step 6: `MARK TYPE x y toward_x toward_y`.

Tool: `send_code_to_revit`. Send this code exactly, as the `code` argument:

```csharp
var D = new Func<string, double>(s => double.Parse(s, System.Globalization.CultureInfo.InvariantCulture));
var L = new FilteredElementCollector(document).OfClass(typeof(Level)).Cast<Level>().FirstOrDefault(l => l.Name == "Level 2");
if (L == null) return "ERROR: Level 2 not found. Run step 2 first.";
int already = new FilteredElementCollector(document).OfCategory(BuiltInCategory.OST_Doors).WhereElementIsNotElementType().Cast<FamilyInstance>().Count(d => d.LevelId == L.Id);
if (already > 0) return "SKIPPED: Level 2 already has " + already + " doors.";
var walls = new FilteredElementCollector(document).OfClass(typeof(Wall)).Cast<Wall>().Where(w => w.LevelId == L.Id && w.WallType.Kind == WallKind.Basic).ToList();
if (walls.Count == 0) return "ERROR: Level 2 has no walls. Run step 8 first.";
var syms = new FilteredElementCollector(document).OfClass(typeof(FamilySymbol)).OfCategory(BuiltInCategory.OST_Doors).Cast<FamilySymbol>().ToList();
var typeNames = new Dictionary<string, string> { { "S", "HH Single 36x84" }, { "S8", "HH Single 36x96" }, { "D6", "HH Pair 72x84" }, { "D88", "HH Pair 88x90" } };

string data = @"
293 S -7.0 33.69 -9 38|291 S 7.04 33.69 3 38|289 S 11.36 33.69 15 38|287 S 30.51 33.69 26 38|285 S 34.98 33.69 38 38|296 S -4.25 27.0 -9 23|295 S -4.25 31.45 -10 31|
290 S 9.36 12.86 5 8|288 S 13.62 12.86 18 8|284 S 32.91 12.86 28 8|282 S 37.11 12.86 42 8|278 S 55.76 12.86 51 8|276 S 59.82 12.86 66 8|274 S 72.04 12.86 74 8|272 S 85.19 12.86 83 8|270 S 89.27 12.86 92 8|268 S 105.67 12.86 102 5|266 S 109.8 13.83 114 5|
B71B S 45.84 34.34 50 40|B71A S 74.03 36.93 60 38|279 S 50.30 30.0 56 30|B70C S 78.37 17.96 78 22|260B S 100.77 17.96 101 22|B72 S 82.13 42.58 88 41|B76 S 83.25 31.49 78 31|269 S 95.32 24.68 90 22|
262 S 107.66 38.77 113 39|264 S 107.66 34.44 113 31|B70A D88 78.05 45.25 78 40";

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
return (problems.Count == 0 ? "OK: created " : "OK with problems: created ") + n + " doors on Level 2. " + string.Join("; ", problems);
```

## Expected result

`OK: created 29 doors on Level 2.` (or `SKIPPED: Level 2 already has 29 doors.`).

If the call times out, Revit is probably showing a warning dialog. Ask the human to click OK, then run this step again.

Then go to step 10.
