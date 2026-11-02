# Step 5: Level 1 walls

Creates the 42 walls of Level 1: 7 exterior wall segments and 35 interior partitions. They run from Level 1 up to Level 2. Each data row is `TYPE x1 y1 x2 y2`, in feet.

Tool: `send_code_to_revit`. Send this code exactly, as the `code` argument:

```csharp
var D = new Func<string, double>(s => double.Parse(s, System.Globalization.CultureInfo.InvariantCulture));
var lv = new FilteredElementCollector(document).OfClass(typeof(Level)).Cast<Level>().ToList();
var L = lv.FirstOrDefault(l => l.Name == "Level 1");
var top = lv.FirstOrDefault(l => l.Name == "Level 2");
if (L == null || top == null) return "ERROR: Level 1 or Level 2 not found. Run step 2 first.";
int already = new FilteredElementCollector(document).OfClass(typeof(Wall)).Cast<Wall>().Count(w => w.LevelId == L.Id);
if (already > 0) return "SKIPPED: Level 1 already has " + already + " walls.";
var types = new FilteredElementCollector(document).OfClass(typeof(WallType)).Cast<WallType>().ToList();

string data = @"
EXT -15.417 0.75 83.85 0.75|EXT 83.85 0.75 83.85 -3.25|EXT 83.85 -3.25 121.25 -3.25|EXT 121.25 -3.25 121.25 45.25|EXT 121.25 45.25 82.11 45.25|EXT 74.02 45.25 -15.417 45.25|EXT -15.417 45.25 -15.417 0.75|
P9A 7.07 45.25 7.07 26.05|P9A -15.417 28.05 0.46 28.05|P2A 0.46 28.05 0.46 0.75|P9A 0.46 26.05 43.77 26.05|P2A -15.417 18.33 0.46 18.33|P2A -15.417 10.01 0.46 10.01|
P2A 5.85 20.97 27.51 20.97|P2A 5.85 20.97 5.85 6.56|P2A 5.85 6.56 61.15 6.56|P2A 27.51 20.97 27.51 6.56|P2A 27.51 18.84 61.15 18.84|P2A 38.53 18.84 38.53 6.56|P2A 49.54 18.84 49.54 6.56|P2A 61.15 18.84 61.15 6.56|
P3A-1 43.77 45.25 43.77 26.05|P4 43.77 27.24 74.02 27.24|P14-1 43.77 34.35 74.02 34.35|P2 51.23 34.35 51.23 27.24|P2 60.22 34.35 60.22 27.24|P4 70.24 34.35 70.24 27.24|P3A-1 74.02 45.25 74.02 27.24|
P2A 82.11 45.25 82.11 36.66|P1 82.11 36.66 98.02 36.66|P9A 98.02 45.25 98.02 35.32|P3A-1 83.65 35.32 98.02 35.32|P2A 83.65 36.66 83.65 0.75|P3A-1 95.32 35.32 95.32 17.7|P3A-1 83.65 26.53 95.32 26.53|P2 83.65 20.01 95.32 20.01|
P2A 83.65 8.24 121.25 8.24|P2A 95.98 8.24 95.98 -3.25|P2A 107.78 8.24 107.78 -3.25|P2 69.95 8.24 83.65 8.24|P2 69.95 8.24 69.95 0.75|P2 74.02 45.25 82.11 45.25";

int n = 0;
foreach (var row in data.Split(new[] { '|' }, StringSplitOptions.RemoveEmptyEntries))
{
    var p = row.Trim().Split(' ');
    string prefix = p[0] == "EXT" ? "EXT Masonry" : p[0] + " (";
    var t = types.FirstOrDefault(x => x.Name.StartsWith(prefix));
    if (t == null) throw new Exception("ERROR: wall type " + p[0] + " is missing. Run step 2 first. Nothing was created.");
    var line = Line.CreateBound(new XYZ(D(p[1]), D(p[2]), L.Elevation), new XYZ(D(p[3]), D(p[4]), L.Elevation));
    var w = Wall.Create(document, line, t.Id, L.Id, top.Elevation - L.Elevation, 0, false, false);
    w.get_Parameter(BuiltInParameter.WALL_HEIGHT_TYPE).Set(top.Id);
    n++;
}
return "OK: created " + n + " walls on Level 1";
```

## Expected result

`OK: created 42 walls on Level 1` (or `SKIPPED: Level 1 already has 42 walls.`).

Then go to step 6.
