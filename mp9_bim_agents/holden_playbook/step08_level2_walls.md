# Step 8: Level 2 walls

Creates the 50 walls of Level 2: 7 exterior segments, 40 partitions, and 3 storefront glass walls between the MSE COLLAB rooms. The walls are 11'-0" high. Each data row is `TYPE x1 y1 x2 y2`, in feet; `SF` means storefront.

Tool: `send_code_to_revit`. Send this code exactly, as the `code` argument:

```csharp
var D = new Func<string, double>(s => double.Parse(s, System.Globalization.CultureInfo.InvariantCulture));
var L = new FilteredElementCollector(document).OfClass(typeof(Level)).Cast<Level>().FirstOrDefault(l => l.Name == "Level 2");
if (L == null) return "ERROR: Level 2 not found. Run step 2 first.";
int already = new FilteredElementCollector(document).OfClass(typeof(Wall)).Cast<Wall>().Count(w => w.LevelId == L.Id);
if (already > 0) return "SKIPPED: Level 2 already has " + already + " walls.";
var types = new FilteredElementCollector(document).OfClass(typeof(WallType)).Cast<WallType>().ToList();
var storefront = types.FirstOrDefault(x => x.Kind == WallKind.Curtain && x.Name.ToLower().Contains("storefront"))
    ?? types.FirstOrDefault(x => x.Kind == WallKind.Curtain) ?? types.FirstOrDefault(x => x.Name.StartsWith("P1 ("));

string data = @"
EXT -15.417 0.75 69.35 0.75|EXT 69.35 0.75 69.35 -3.25|EXT 69.35 -3.25 121.25 -3.25|EXT 121.25 -3.25 121.25 45.25|EXT 121.25 45.25 82.13 45.25|EXT 74.03 45.25 -15.417 45.25|EXT -15.417 45.25 -15.417 0.75|
P2A -15.417 33.69 43.85 33.69|P2A -2.78 45.25 -2.78 33.69|P2A 9.21 45.25 9.21 33.69|P2A 21.01 45.25 21.01 33.69|P2A 32.79 45.25 32.79 33.69|P3A-1 43.85 45.25 43.85 33.69|
P2A -15.417 29.21 -4.25 29.21|P2A -4.25 33.69 -4.25 17.91|P2A -15.417 17.91 -4.25 17.91|P2A -4.25 12.86 107.72 12.86|
P2A -0.58 12.86 -0.58 0.75|P2A 11.42 12.86 11.42 0.75|P2A 23.17 12.86 23.17 0.75|P2A 34.97 12.86 34.97 0.75|P2A 46.74 12.86 46.74 0.75|P2A 57.83 12.86 57.83 0.75|P2A 69.97 12.86 69.97 0.75|
P2A 78.09 12.86 78.09 -3.25|P2A 87.19 12.86 87.19 -3.25|P2A 96.21 12.86 96.21 -3.25|P2A 107.72 13.83 107.72 -3.25|P2A 107.72 13.83 121.25 13.83|
P14A-1 43.85 34.34 74.03 34.34|P3A-1 74.03 45.25 74.03 27.28|P2 50.30 34.34 50.30 27.28|P1 60.22 34.34 60.22 17.96|P4 70.26 34.34 70.26 27.28|P4 50.30 27.28 74.03 27.28|P2A 55.33 17.96 107.66 17.96|
P2A 82.13 45.25 82.13 36.57|P1 82.13 36.57 92.03 36.57|P2A 92.03 45.25 92.03 35.33|P2A-1 83.25 35.33 95.32 35.33|P2A-1 83.25 36.57 83.25 17.96|P2A-1 95.32 35.33 95.32 17.96|P2A-1 83.25 26.97 95.32 26.97|
P2A 107.66 45.25 107.66 25.24|P2A 107.66 36.69 121.25 36.69|P2A 107.66 25.24 121.25 25.24|P2 74.03 45.25 82.13 45.25|
SF 14 18.3 14 27.75|SF 28 18.3 28 27.75|SF 42 18.3 42 27.75";

int n = 0;
foreach (var row in data.Split(new[] { '|' }, StringSplitOptions.RemoveEmptyEntries))
{
    var p = row.Trim().Split(' ');
    WallType t = p[0] == "SF" ? storefront : types.FirstOrDefault(x => x.Name.StartsWith(p[0] == "EXT" ? "EXT Masonry" : p[0] + " ("));
    if (t == null) throw new Exception("ERROR: wall type " + p[0] + " is missing. Run step 2 first. Nothing was created.");
    var line = Line.CreateBound(new XYZ(D(p[1]), D(p[2]), L.Elevation), new XYZ(D(p[3]), D(p[4]), L.Elevation));
    Wall.Create(document, line, t.Id, L.Id, 11.0, 0, false, false);
    n++;
}
return "OK: created " + n + " walls on Level 2";
```

## Expected result

`OK: created 50 walls on Level 2` (or `SKIPPED: Level 2 already has 50 walls.`).

Then go to step 9.
