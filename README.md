# SecularGapComplexity

This is the repository for the paper "Gap Complexity of a Tightly Packed Planetary System Perturbed by an Exterior Giant Companion" (Livesey & Becker, *in prep*). Welcome! Below are instructions for re-producing the figures from the paper.

**Figure 1:** Run the cells in the notebook `simulation-pairs.ipynb`. This notebook also includes some extra information on the gap complexity metric and our model.

**Figure 3 & 4:** Use the first 3 cells in the notebook. Setting different values for `rng_seed` will give different results. Tune the OG mass and semi-major axis to get extreme changes like in our Fig. 3, or minute changes like in our Fig. 4.

**Figure 5:** In the terminal, run the following.
```
% python makeplots.py 1600
```
This will generate an individual PDF for each subplot from our Fig. 5. Increase or decrease the sample size by using a perfect square other than 1600.
