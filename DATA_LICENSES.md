# Data Sources & Licenses

All data used by this project is publicly available and license-permitted for
reuse, including commercial reuse, subject to attribution.

## Central Statistics Office (CSO), Ireland — PxStat Open Data API

- **Provider:** Central Statistics Office (CSO), Ireland — https://www.cso.ie/
- **Access:** PxStat RESTful API (no key required)
  `https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/{MATRIX}/JSON-stat/2.0/en`
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
  CSO releases its open data under CC BY 4.0; reuse is permitted with attribution.
  See https://www.cso.ie/en/aboutus/lgdp/csodatapolicies/ and the open-data
  statement at https://data.cso.ie/.
- **Required attribution:** "Source: Central Statistics Office, Ireland (CSO).
  Licensed under CC BY 4.0." This attribution is embedded in every retrieved
  chunk's citation metadata and surfaced in answers.

### Datasets ingested

| Matrix | Title | Topic | License |
|--------|-------|-------|---------|
| CPM01  | Consumer Price Index | Prices / inflation | CC BY 4.0 |
| MUM01  | Monthly Unemployment | Labour market | CC BY 4.0 |
| FY001A | Census of Population | Census | CC BY 4.0 |

> Matrix list is configurable in `config.yaml` (`ingest.datasets`). Any CSO
> PxStat matrix code can be added; its license is the same CSO CC BY 4.0.

## Notes on compliance

- No scraping of restricted pages; only the official open-data API is used.
- No personal/microdata is ingested — CSO publishes only aggregated statistics,
  so there is no PII in the corpus. (A PII filter still runs on user queries and
  generated output as defence-in-depth.)
- The bundled `data/sample/` extract is a small real subset of the above,
  included only so the pipeline and tests run offline; it carries the same
  CC BY 4.0 license and attribution.
