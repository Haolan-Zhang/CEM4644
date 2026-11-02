# Step 1: Check the connection

Tool: `send_code_to_revit`. Send this code exactly, as the `code` argument:

```csharp
var walls = new FilteredElementCollector(document).OfClass(typeof(Wall)).GetElementCount();
var levels = new FilteredElementCollector(document).OfClass(typeof(Level)).Cast<Level>().OrderBy(l => l.Elevation).Select(l => l.Name).ToList();
return "OK: connected to '" + document.Title + "', Revit " + document.Application.VersionNumber
    + ", walls in model: " + walls + ", levels: " + string.Join(", ", levels)
    + (walls > 0 ? ". NOTE: the model already has walls." : ". The model is empty, ready to start.");
```

## Expected result

`OK: connected to '...', Revit 20xx, walls in model: 0, levels: ... The model is empty, ready to start.`

## What to do next

- If the result says `The model is empty`: go to step 2.
- If it says `NOTE: the model already has walls` and you have **not** run any step yet, ask the human whether this is the right project before continuing.
- If it says `NOTE: the model already has walls` and you were already in the middle of the playbook, continue with the next step you haven't finished. Steps skip work that's already done.
- If the tool fails or times out, ask the human to check that Revit is open, the add-in is loaded, and `send_code_to_revit` is enabled in the add-in's Settings.
