# Reports (rendered notebooks)

HTML versions of the analysis notebooks, with all outputs (charts, tables) embedded.  
**View these in a browser** to see the insights without running any code.

| Report | Description |
|--------|-------------|
| [01_sub3_performance_modeling.html](01_sub3_performance_modeling.html) | Sub-3 marathon training: volume, pace zones, consistency, MP miles from FIT. |
| [02_lifetime_athlete_intelligence.html](02_lifetime_athlete_intelligence.html) | Lifetime fun insights: activity mix, most/rarest types, kudos, run volume over time. |

To regenerate after re-running the pipeline:

```bash
./scripts/export_reports.sh
```

Requires `jupyter` and `nbconvert` (in `requirements.txt`).
