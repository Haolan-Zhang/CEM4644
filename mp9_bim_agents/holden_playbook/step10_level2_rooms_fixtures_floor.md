# Step 10: Level 2 rooms, toilet fixtures, and floor

This step adds to Level 2:

- 8 room separation lines (around the open collab, break, visitor, lobby, and BSA areas);
- 37 rooms with names and numbers;
- 1 lavatory and 1 water closet in UNIV TLT B72;
- the floor slab.

Tool: `send_code_to_revit`. Send this code exactly, as the `code` argument:

```csharp
var D = new Func<string, double>(s => double.Parse(s, System.Globalization.CultureInfo.InvariantCulture));
var L = new FilteredElementCollector(document).OfClass(typeof(Level)).Cast<Level>().FirstOrDefault(l => l.Name == "Level 2");
if (L == null) return "ERROR: Level 2 not found. Run step 2 first.";
var log = new List<string>();

string seps = "-4.25 17.91 -4.25 12.86|0 18.3 42 18.3|0 27.75 42 27.75|0 18.3 0 27.75|74.03 27.28 83.25 27.28|50.30 27.28 50.30 17.96|50.30 17.96 55.33 17.96|107.66 25.24 107.66 13.83";
string rooms = @"
-10.07 38.76 293 MSE FAC 16|3.92 38.76 291 MSE FAC 17|14.75 38.56 289 MSE FAC 18|25.72 40.07 287 MSE FAC 19|38.16 38.76 285 MSE FAC 20|-10.39 31.77 295 COAT|-8.12 22.6 296 MSE FAC 15|-6.35 4.16 294 MSE BREAK ROOM|
5.03 8.05 290 MSE FAC 14|17.9 8.05 288 MSE FAC 13|27.83 8.05 284 MSE FAC 12|42.14 8.05 282 MSE FAC 11|50.76 8.05 278 MSE FAC 10|66.58 8.05 276 MSE FAC 9|74.71 8.05 274 MSE FAC 8|82.95 2.13 272 MSE FAC 7|91.66 8.05 270 MSE FAC 6|
102.05 2.09 268 MSE DEP HEAD|114.31 4.19 266 ASSIS TO HEAD|6.42 22.64 280C MSE COLLAB|20.6 22.63 280B MSE COLLAB|35.2 22.63 280A MSE COLLAB|45.62 22.39 280 CORR|55.92 22.39 277 MSE VISIT.|56.06 29.65 279 STORAGE|
65.51 30.73 B77 ELEV|72.2 30.5 B73 RECYCLE|68.16 22.64 B70A ELEV. LOBBY|78.28 37.71 B70 CORR|58.68 38.0 B71 STAIR 3|88.26 40.79 B72 UNIV TLT|89.2 30.01 B76 ELECTRICAL|89.5 21.5 269 MSE WORK ROOM|
102.22 22.45 260 MSE SUITE|113.29 19.89 260A MSE - BSA|113.3 31.17 264 MSE BUS MANAGER|113.23 40.0 262 MSE FLEX OFFICE";
string fixtures = "LAV 85.6 36.77|WC 90.4 36.77";
double[] floorPts = { -16.167, 0, 68.6, 0, 68.6, -4, 122, -4, 122, 46, -16.167, 46 };

// Rooms and room separation lines
int roomCount = new FilteredElementCollector(document).OfCategory(BuiltInCategory.OST_Rooms).Cast<SpatialElement>().Count(r => r.LevelId == L.Id);
if (roomCount > 0) log.Add("rooms already exist (" + roomCount + "), skipped");
else
{
    var plan = new FilteredElementCollector(document).OfClass(typeof(ViewPlan)).Cast<ViewPlan>()
        .Where(v => !v.IsTemplate && v.ViewType == ViewType.FloorPlan && v.GenLevel != null && v.GenLevel.Id == L.Id)
        .OrderBy(v => v.Name == L.Name ? 0 : 1).FirstOrDefault();
    if (plan == null) return "ERROR: no floor plan view for Level 2. Run step 2 again.";
    var sp = SketchPlane.Create(document, Plane.CreateByNormalAndOrigin(XYZ.BasisZ, new XYZ(0, 0, L.Elevation)));
    var ca = new CurveArray();
    foreach (var row in seps.Split(new[] { '|' }, StringSplitOptions.RemoveEmptyEntries))
    {
        var p = row.Trim().Split(' ');
        ca.Append(Line.CreateBound(new XYZ(D(p[0]), D(p[1]), L.Elevation), new XYZ(D(p[2]), D(p[3]), L.Elevation)));
    }
    document.Create.NewRoomBoundaryLines(sp, ca, plan);
    document.Regenerate();
    int n = 0;
    foreach (var row in rooms.Split(new[] { '|' }, StringSplitOptions.RemoveEmptyEntries))
    {
        var p = row.Trim().Split(new[] { ' ' }, 4);
        var r = document.Create.NewRoom(L, new UV(D(p[0]), D(p[1])));
        r.Number = p[2];
        r.Name = p[3];
        if (r.Area <= 0) log.Add("room not enclosed: " + p[2]);
        n++;
    }
    log.Add(n + " rooms");
}

// Plumbing fixtures
int fixCount = new FilteredElementCollector(document).OfCategory(BuiltInCategory.OST_PlumbingFixtures).WhereElementIsNotElementType().Cast<FamilyInstance>().Count(f => f.LevelId == L.Id);
var wc = new FilteredElementCollector(document).OfClass(typeof(FamilySymbol)).Cast<FamilySymbol>().FirstOrDefault(f => f.FamilyName == "Water Closet - Simple");
var lav = new FilteredElementCollector(document).OfClass(typeof(FamilySymbol)).Cast<FamilySymbol>().FirstOrDefault(f => f.FamilyName == "Lavatory - Simple");
if (fixCount > 0) log.Add("fixtures already exist, skipped");
else if (wc == null || lav == null) log.Add("fixture families missing, fixtures skipped");
else
{
    if (!wc.IsActive) wc.Activate();
    if (!lav.IsActive) lav.Activate();
    int n = 0;
    foreach (var row in fixtures.Split('|'))
    {
        var p = row.Trim().Split(' ');
        document.Create.NewFamilyInstance(new XYZ(D(p[1]), D(p[2]), L.Elevation), p[0] == "WC" ? wc : lav, L, Autodesk.Revit.DB.Structure.StructuralType.NonStructural);
        n++;
    }
    log.Add(n + " fixtures");
}

// Floor
int floorCount = new FilteredElementCollector(document).OfClass(typeof(Floor)).Cast<Floor>().Count(f => f.LevelId == L.Id);
if (floorCount > 0) log.Add("floor already exists, skipped");
else
{
    var ft = new FilteredElementCollector(document).OfClass(typeof(FloorType)).Cast<FloorType>().Where(f => !f.IsFoundationSlab)
        .OrderBy(f => f.Name.ToLower().Contains("conc") ? 0 : 1).First();
    var loop = new CurveLoop();
    int k = floorPts.Length / 2;
    for (int i = 0; i < k; i++)
    {
        int j = (i + 1) % k;
        loop.Append(Line.CreateBound(new XYZ(floorPts[2 * i], floorPts[2 * i + 1], L.Elevation), new XYZ(floorPts[2 * j], floorPts[2 * j + 1], L.Elevation)));
    }
    Floor.Create(document, new List<CurveLoop> { loop }, ft.Id, L.Id);
    log.Add("1 floor");
}
return "OK: Level 2: " + string.Join("; ", log);
```

## Expected result

`OK: Level 2: 37 rooms; 2 fixtures; 1 floor`

If it lists `room not enclosed: ...`, tell the human the room numbers, then continue.

Then go to step 11.
