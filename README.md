# SecularGapComplexity

This is the repository for the paper "Gap Complexity of a Tightly Packed Planetary System Perturbed by an Exterior Giant Companion" (Livesey & Becker, *submitted*). Welcome! Below are instructions for re-producing the figures from the paper.

**Figure 1:** Run the cells in the notebook `simulation-pairs.ipynb`. This notebook also includes some extra information on the gap complexity metric and our model.

**Figure 3 & 4:** Use the first 3 cells in `simulation-pairs.ipynb`. Setting different values for `rng_seed` will give different results. Tune the OG mass and semi-major axis to get extreme changes like in our Fig. 3, or minute changes like in our Fig. 4.

**Figure 5:** In the terminal, run the following.
```
% python makeplots.py 2500 4 10
```
This will generate the first subplot from our Fig. 5. Change the STIP multiplicity by entering a number other than 4, and change the inclination (in degrees) of the OG by selecting a number other than 10. Increase or decrease the sample size by using a perfect square other than 2500.

The notebook `time-dependence.ipynb` contains some plots justifying the choice of time span for computing gap complexities in the paper. The notebook `nbody-comparison.ipynb` contains some comparisons between $N$-body simulations and the secular models we run.
