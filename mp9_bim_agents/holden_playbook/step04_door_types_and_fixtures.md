# Step 4: Door types and plumbing fixture families

This step creates:

- 4 door types: `HH Single 36x84` (most doors), `HH Single 36x96` (A71A), `HH Pair 72x84` (A78A, A78B), and `HH Pair 88x90` (A70A, B70A);
- 2 simple plumbing families, `Water Closet - Simple` and `Lavatory - Simple`, built from the Generic Model family template (Revit has no toilet families installed by default).

Tool: `send_code_to_revit`. Send this code exactly, as the `code` argument:

```csharp
var log = new List<string>();

// Door types
var doorSyms = new FilteredElementCollector(document).OfClass(typeof(FamilySymbol)).OfCategory(BuiltInCategory.OST_Doors).Cast<FamilySymbol>().ToList();
if (doorSyms.Count == 0)
    return "ERROR: no door family is loaded. Ask the human to load one single door and one double door family (Insert > Load Autodesk Family > Doors), then run step 4 again.";
Func<FamilySymbol, string> key = f => (f.FamilyName + " " + f.Name).ToLower();
var sglBase = doorSyms.FirstOrDefault(f => key(f).Contains("sgl") || key(f).Contains("single"))
    ?? doorSyms.FirstOrDefault(f => !key(f).Contains("dbl") && !key(f).Contains("double")) ?? doorSyms.First();
var dblBase = doorSyms.FirstOrDefault(f => key(f).Contains("dbl") || key(f).Contains("double")) ?? sglBase;
if (dblBase == sglBase) log.Add("no double door family found, pairs use the single door family");
var specs = new List<Tuple<FamilySymbol, string, double, double>> {
    Tuple.Create(sglBase, "HH Single 36x84", 3.0, 7.0),
    Tuple.Create(sglBase, "HH Single 36x96", 3.0, 8.0),
    Tuple.Create(dblBase, "HH Pair 72x84", 6.0, 7.0),
    Tuple.Create(dblBase, "HH Pair 88x90", 88.0 / 12.0, 7.5)
};
foreach (var s in specs)
{
    var t = doorSyms.FirstOrDefault(f => f.Name == s.Item2) ?? (FamilySymbol)s.Item1.Duplicate(s.Item2);
    var pw = t.LookupParameter("Width");
    var ph = t.LookupParameter("Height");
    if (pw != null && !pw.IsReadOnly) pw.Set(s.Item3); else log.Add(s.Item2 + ": Width not editable");
    if (ph != null && !ph.IsReadOnly) ph.Set(s.Item4); else log.Add(s.Item2 + ": Height not editable");
    if (!t.IsActive) t.Activate();
}

// Plumbing fixture families
var have = new FilteredElementCollector(document).OfClass(typeof(Family)).Cast<Family>().Select(f => f.Name).ToList();
if (have.Contains("Water Closet - Simple") && have.Contains("Lavatory - Simple"))
    return "OK: door types and plumbing families ready (families already existed). " + string.Join("; ", log);
string root = @"C:\ProgramData\Autodesk\RVT " + document.Application.VersionNumber + @"\Family Templates";
string tpl = System.IO.Directory.Exists(root)
    ? System.IO.Directory.GetFiles(root, "*Generic Model.rft", System.IO.SearchOption.AllDirectories).OrderBy(p => p.Contains("English") ? 0 : 1).FirstOrDefault()
    : null;
if (tpl == null)
    return "ERROR: Generic Model family template not found under " + root + ". Door types were created. Fixtures will be skipped in steps 7 and 10; continue with step 5.";

Func<double, double, double, double, CurveArray> rect = (x0, y0, x1, y1) => {
    var ca = new CurveArray();
    ca.Append(Line.CreateBound(new XYZ(x0, y0, 0), new XYZ(x1, y0, 0)));
    ca.Append(Line.CreateBound(new XYZ(x1, y0, 0), new XYZ(x1, y1, 0)));
    ca.Append(Line.CreateBound(new XYZ(x1, y1, 0), new XYZ(x0, y1, 0)));
    ca.Append(Line.CreateBound(new XYZ(x0, y1, 0), new XYZ(x0, y0, 0)));
    return ca;
};
Func<double, double, double, double, CurveArray> ellipse = (cx, cy, rx, ry) => {
    var ca = new CurveArray();
    ca.Append(Ellipse.CreateCurve(new XYZ(cx, cy, 0), rx, ry, XYZ.BasisX, XYZ.BasisY, 0, Math.PI));
    ca.Append(Ellipse.CreateCurve(new XYZ(cx, cy, 0), rx, ry, XYZ.BasisX, XYZ.BasisY, Math.PI, 2 * Math.PI));
    return ca;
};
// Origin = back-center of the fixture at floor level; the fixture projects toward +Y
var defs = new Dictionary<string, List<Tuple<CurveArray, double, double>>> {
    { "Water Closet - Simple", new List<Tuple<CurveArray, double, double>> {
        Tuple.Create(rect(-0.83, 0, 0.83, 0.75), 0.0, 2.5),
        Tuple.Create(ellipse(0, 1.55, 0.6, 0.85), 0.0, 1.25),
        Tuple.Create(rect(-0.35, 0.6, 0.35, 1.0), 0.0, 1.25) } },
    { "Lavatory - Simple", new List<Tuple<CurveArray, double, double>> {
        Tuple.Create(rect(-0.83, 0, 0.83, 1.5), 2.33, 2.83),
        Tuple.Create(rect(-0.25, 0, 0.25, 0.4), 0.0, 2.33) } }
};
foreach (var kv in defs)
{
    if (have.Contains(kv.Key)) continue;
    var fdoc = document.Application.NewFamilyDocument(tpl);
    using (var t = new Transaction(fdoc, "build fixture"))
    {
        t.Start();
        fdoc.OwnerFamily.FamilyCategory = Category.GetCategory(fdoc, BuiltInCategory.OST_PlumbingFixtures);
        var sp = SketchPlane.Create(fdoc, Plane.CreateByNormalAndOrigin(XYZ.BasisZ, XYZ.Zero));
        foreach (var part in kv.Value)
        {
            var arr = new CurveArrArray();
            arr.Append(part.Item1);
            var ex = fdoc.FamilyCreate.NewExtrusion(true, arr, sp, part.Item3 - part.Item2);
            ex.StartOffset = part.Item2;
            ex.EndOffset = part.Item3;
        }
        t.Commit();
    }
    string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), kv.Key + ".rfa");
    fdoc.SaveAs(path, new SaveAsOptions { OverwriteExistingFile = true });
    fdoc.Close(false);
    Family fam;
    document.LoadFamily(path, out fam);
    var sym = (FamilySymbol)document.GetElement(fam.GetFamilySymbolIds().First());
    if (!sym.IsActive) sym.Activate();
}
return "OK: door types and plumbing families ready. " + string.Join("; ", log);
```

## Expected result

`OK: door types and plumbing families ready.` Notes after it are fine.

- If the result starts with `ERROR: no door family is loaded`: tell the human and wait. After they load the doors, run step 4 again.
- If the result starts with `ERROR: Generic Model family template not found`: tell the human, then continue with step 5. The model will have no toilet fixtures.

Then go to step 5.
