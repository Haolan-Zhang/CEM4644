# Step 2: Levels, units, and wall types

This step:

- makes Level 1 at 0'-0" and Level 2 at 11'-10", each with a floor plan view of the same name;
- sets the project's length units to feet and fractional inches;
- creates 16 wall types, one per partition mark in the PDF, at the thickness listed on page 5.

Tool: `send_code_to_revit`. Send this code exactly, as the `code` argument:

```csharp
var log = new List<string>();
double l2Elev = 11.0 + 10.0 / 12.0;

// Levels: the two lowest existing levels become Level 1 and Level 2
var levels = new FilteredElementCollector(document).OfClass(typeof(Level)).Cast<Level>().OrderBy(l => l.Elevation).ToList();
Level l1 = levels.Count > 0 ? levels[0] : Level.Create(document, 0);
Level l2 = levels.Count > 1 ? levels[1] : Level.Create(document, l2Elev);
if (l1.Name != "Level 1" || l2.Name != "Level 2")
{
    l1.Name = "TMP_LEVEL_A"; l2.Name = "TMP_LEVEL_B";
    l1.Name = "Level 1"; l2.Name = "Level 2";
}
l1.Elevation = 0;
l2.Elevation = l2Elev;

// One floor plan per level, named after the level
var fpType = new FilteredElementCollector(document).OfClass(typeof(ViewFamilyType)).Cast<ViewFamilyType>().First(t => t.ViewFamily == ViewFamily.FloorPlan);
var plans = new List<Tuple<Level, ViewPlan>>();
foreach (var lev in new[] { l1, l2 })
{
    var fp = new FilteredElementCollector(document).OfClass(typeof(ViewPlan)).Cast<ViewPlan>()
        .Where(v => !v.IsTemplate && v.ViewType == ViewType.FloorPlan && v.GenLevel != null && v.GenLevel.Id == lev.Id)
        .OrderBy(v => v.Name.ToLower().Contains("site") ? 1 : 0).FirstOrDefault();
    if (fp == null) { fp = ViewPlan.Create(document, fpType.Id, lev.Id); log.Add("created floor plan for " + lev.Name); }
    plans.Add(Tuple.Create(lev, fp));
}
foreach (var p in plans) { try { p.Item2.Name = "TMP_PLAN_" + p.Item1.Name; } catch { } }
foreach (var p in plans) { try { p.Item2.Name = p.Item1.Name; } catch (Exception e) { log.Add("could not rename plan of " + p.Item1.Name + ": " + e.Message); } }

// Units: feet and fractional inches
var units = document.GetUnits();
var fo = new FormatOptions(UnitTypeId.FeetFractionalInches);
fo.Accuracy = 1.0 / 96.0;
units.SetFormatOptions(SpecTypeId.Length, fo);
document.SetUnits(units);

// Wall types: single layer, thickness in inches
var allWT = new FilteredElementCollector(document).OfClass(typeof(WallType)).Cast<WallType>().ToList();
var baseWT = allWT.First(t => t.Kind == WallKind.Basic && t.GetCompoundStructure() != null);
var mats = new FilteredElementCollector(document).OfClass(typeof(Material)).Cast<Material>().ToList();
var masonry = mats.FirstOrDefault(m => m.Name.ToLower().Contains("brick")) ?? mats.FirstOrDefault(m => m.Name.ToLower().Contains("masonry"));
var gyp = mats.FirstOrDefault(m => m.Name.ToLower().Contains("gypsum")) ?? mats.FirstOrDefault(m => m.Name.ToLower().Contains("plaster"));
var specs = new List<Tuple<string, double, bool>> {
    Tuple.Create("EXT Masonry + P13 (1'-6\")", 18.0, true),
    Tuple.Create("P1 (4 1/4\")", 4.25, false), Tuple.Create("P1A (4 1/4\")", 4.25, false),
    Tuple.Create("P2 (4 7/8\")", 4.875, false), Tuple.Create("P2A (4 7/8\")", 4.875, false), Tuple.Create("P2A-1 (4 7/8\")", 4.875, false),
    Tuple.Create("P3 (7 1/4\")", 7.25, false), Tuple.Create("P3A-1 (7 1/4\")", 7.25, false), Tuple.Create("P3-1 (7 1/4\")", 7.25, false),
    Tuple.Create("P4 (6 5/8\")", 6.625, false), Tuple.Create("P5A-2 (6 1/8\")", 6.125, false), Tuple.Create("P8-2 (6 1/4\")", 6.25, false),
    Tuple.Create("P9A (5 1/2\")", 5.5, false), Tuple.Create("P14-1 (9 1/4\")", 9.25, false), Tuple.Create("P14A-1 (9 1/4\")", 9.25, false),
    Tuple.Create("M3 CMU (11 5/8\")", 11.625, false)
};
foreach (var s in specs)
{
    var wt = allWT.FirstOrDefault(t => t.Name == s.Item1) ?? (WallType)baseWT.Duplicate(s.Item1);
    var matId = s.Item3 ? (masonry != null ? masonry.Id : ElementId.InvalidElementId) : (gyp != null ? gyp.Id : ElementId.InvalidElementId);
    wt.SetCompoundStructure(CompoundStructure.CreateSingleLayerCompoundStructure(MaterialFunctionAssignment.Structure, s.Item2 / 12.0, matId));
    wt.Function = s.Item3 ? WallFunction.Exterior : WallFunction.Interior;
}
return "OK: levels, units and 16 wall types ready. " + string.Join("; ", log);
```

## Expected result

`OK: levels, units and 16 wall types ready.` It may be followed by notes such as `created floor plan for Level 2`; those are fine.

Then go to step 3.
