CTMC-BHMM v7 — pre-fit dossiers
================================

00_OVERVIEW.pdf    comparative view across all four outbreaks
01_EBOLA.pdf       Kikwit, DR Congo, 1995
02_MEASLES.pdf     Hagelloch, Germany, 1861
03_INFLUENZA.pdf   English boarding school, 1978  (CAVEATED ARM)
04_NOROVIRUS.pdf   Derbyshire school, England, 2001
figures/           every panel as PNG, each with a .txt giving its reading

WHERE THE NUMBERS COME FROM
---------------------------
Each dossier reports both:
  RAW       data/raw/...       the records before any decision was made
  CANONICAL data/canonical/... the frozen analysis file the model reads

Every finding appears as: value in RAW -> decision taken -> status in CANONICAL.
A problem that has been fixed is shown AS FIXED, with before and after numbers.
It is not removed from the report. Running the audit only on the canonical file
would leave every data-handling decision unjustified, which is the first thing a
reviewer would ask about.

Regenerate:  python scripts/13_disease_dossiers.py
