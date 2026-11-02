# Step 11: Room tags, plan crop, and 3D view

This step:

- tags every room on both floor plans;
- crops both plans to the building;
- creates a shaded 3D view called `Holden South Wing - 3D`.

Tool: `send_code_to_revit`. Send this code exactly, as the `code` argument:

```csharp
var log = new List<string>();
foreach (var name in new[] { "Level 1", "Level 2" })
{
    var lev = new FilteredElementCollector(document).OfClass(typeof(Level)).Cast<Level>().FirstOrDefault(l => l.Name == name);
    if (lev == null) { log.Add(name + " not found"); continue; }
    var v = new FilteredElementCollector(document).OfClass(typeof(ViewPlan)).Cast<ViewPlan>()
        .Where(x => !x.IsTemplate && x.ViewType == ViewType.FloorPlan && x.GenLevel != null && x.GenLevel.Id == lev.Id)
        .OrderBy(x => x.Name == name ? 0 : 1).FirstOrDefault();
    if (v == null) { log.Add("no floor plan for " + name); continue; }
    v.Scale = 96;
    try
    {
        v.CropBoxActive = true;
        v.CropBoxVisible = false;
        var bb = v.CropBox;
        bb.Min = new XYZ(-24, -12, bb.Min.Z);
        bb.Max = new XYZ(130, 58, bb.Max.Z);
        v.CropBox = bb;
    }
    catch (Exception e) { log.Add(name + " crop not set: " + e.Message); }
    int tagged = new FilteredElementCollector(document, v.Id).OfCategory(BuiltInCategory.OST_RoomTags).GetElementCount();
    if (tagged > 0) { log.Add(name + ": room tags already exist"); continue; }
    int n = 0;
    foreach (var r in new FilteredElementCollector(document).OfCategory(BuiltInCategory.OST_Rooms).Cast<SpatialElement>().Where(r => r.LevelId == lev.Id && r.Area > 0))
    {
        var p = (r.Location as LocationPoint).Point;
        try { document.Create.NewRoomTag(new LinkElementId(r.Id), new UV(p.X, p.Y), v.Id); n++; }
        catch (Exception e) { log.Add("tag failed: " + e.Message); break; }
    }
    log.Add(name + ": " + n + " room tags");
}
var v3name = "Holden South Wing - 3D";
if (new FilteredElementCollector(document).OfClass(typeof(View3D)).Cast<View3D>().Any(x => x.Name == v3name)) log.Add("3D view already exists");
else
{
    var vft = new FilteredElementCollector(document).OfClass(typeof(ViewFamilyType)).Cast<ViewFamilyType>().First(t => t.ViewFamily == ViewFamily.ThreeDimensional);
    var v3 = View3D.CreateIsometric(document, vft.Id);
    v3.Name = v3name;
    v3.DetailLevel = ViewDetailLevel.Medium;
    try { v3.get_Parameter(BuiltInParameter.MODEL_GRAPHICS_STYLE).Set(3); } catch { }
    log.Add("3D view created");
}
return "OK: " + string.Join("; ", log);
```

## Expected result

`OK: Level 1: 24 room tags; Level 2: 37 room tags; 3D view created`

Then go to step 12.
