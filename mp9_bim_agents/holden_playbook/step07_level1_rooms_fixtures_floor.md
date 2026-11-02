# Step 7: Level 1 rooms, toilet fixtures, and floor

This step adds to Level 1:

- 1 room separation line (between RECEPTION 160 and ADVISING/OPEN STUDY 160F);
- 24 rooms with names and numbers;
- 2 water closets and 1 lavatory in W TLT A72;
- the floor slab.

Tool: `send_code_to_revit`. Send this code exactly, as the `code` argument:

```csharp
var D = new Func<string, double>(s => double.Parse(s, System.Globalization.CultureInfo.InvariantCulture));
var L = new FilteredElementCollector(document).OfClass(typeof(Level)).Cast<Level>().FirstOrDefault(l => l.Name == "Level 1");
if (L == null) return "ERROR: Level 1 not found. Run step 2 first.";
var log = new List<string>();

string seps = "95.32 17.7 95.32 8.24";
string rooms = @"
-0.37 36.33 191 ECP|32.82 34.81 185 LARGE CONFERENCE|48.37 29.38 185A STORAGE / CONF. SUP.|55.06 29.98 A81 CUST CL|65.34 30.42 A77 ELEV|72.2 30.5 A73 RECYCLE|58.36 38.19 A71 STAIR 3|
86.75 41.25 A72 W TLT|90.8 30.72 A76 ELECTRICAL|89.87 23.0 160E ADVISING STORAGE AND SUPPORT|89.5 13.34 160 RECEPTION|110.72 22.46 160F ADVISING/OPEN STUDY|
90.02 2.0 160A ADVISING STAFF|102.13 2.0 160B MSE ASSOC HEAD|114.42 2.0 160C ADVISING STAFF|76.8 4.5 A78 VESTIBULE|63.78 23.1 A70 CORRIDOR|18.62 12.24 190 MME - ADVISING|
33.37 12.24 184 FOCUS|44.39 12.24 182 FOCUS|55.67 12.24 178 TEAM / INTERVIEW|-6.59 22.7 195 ECP OFFICE|-5.33 14.0 196 TCD|-5.33 5.48 194 ACADEMIC ADVISOR";
string fixtures = "WC 91.1 36.84|WC 96.2 36.84|LAV 86.0 36.84";
double[] floorPts = { -16.167, 0, 83.1, 0, 83.1, -4, 122, -4, 122, 46, -16.167, 46 };

// Rooms and room separation lines
int roomCount = new FilteredElementCollector(document).OfCategory(BuiltInCategory.OST_Rooms).Cast<SpatialElement>().Count(r => r.LevelId == L.Id);
if (roomCount > 0) log.Add("rooms already exist (" + roomCount + "), skipped");
else
{
    var plan = new FilteredElementCollector(document).OfClass(typeof(ViewPlan)).Cast<ViewPlan>()
        .Where(v => !v.IsTemplate && v.ViewType == ViewType.FloorPlan && v.GenLevel != null && v.GenLevel.Id == L.Id)
        .OrderBy(v => v.Name == L.Name ? 0 : 1).FirstOrDefault();
    if (plan == null) return "ERROR: no floor plan view for Level 1. Run step 2 again.";
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
return "OK: Level 1: " + string.Join("; ", log);
```

## Expected result

`OK: Level 1: 24 rooms; 3 fixtures; 1 floor`

- If it lists `room not enclosed: ...`, tell the human the room numbers, then continue.
- If it says `fixture families missing`, step 4 could not build the fixtures. Continue anyway.

Then go to step 8.
