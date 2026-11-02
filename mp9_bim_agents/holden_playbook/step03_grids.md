# Step 3: Grids

Creates the north-south grids X2 to X10 and X5.5. X2 to X8 are 14'-0" apart, X8 to X10 are 12'-0" apart, and X5.5 is 6'-10" east of X5.

Tool: `send_code_to_revit`. Send this code exactly, as the `code` argument:

```csharp
var names = new FilteredElementCollector(document).OfClass(typeof(Grid)).Cast<Grid>().Select(g => g.Name).ToList();
if (names.Contains("X2")) return "SKIPPED: grids already exist";
var grids = new List<Tuple<string, double>> {
    Tuple.Create("X2", 0.0), Tuple.Create("X3", 14.0), Tuple.Create("X4", 28.0), Tuple.Create("X5", 42.0),
    Tuple.Create("X5.5", 42.0 + 82.0 / 12.0), Tuple.Create("X6", 56.0), Tuple.Create("X7", 70.0),
    Tuple.Create("X8", 84.0), Tuple.Create("X9", 96.0), Tuple.Create("X10", 108.0)
};
foreach (var g in grids)
{
    var gr = Grid.Create(document, Line.CreateBound(new XYZ(g.Item2, -10, 0), new XYZ(g.Item2, 56, 0)));
    gr.Name = g.Item1;
}
return "OK: created 10 grids";
```

## Expected result

`OK: created 10 grids` (or `SKIPPED: grids already exist`).

Then go to step 4.
