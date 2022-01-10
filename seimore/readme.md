This repository contains a set of codes behind the obtained results of the article with (arXiv:2011.14774, DOI:10.3847/1538-4357/abe865). The main objectives of these set of codes were first) to transform the theoretical power spectrums of large-scale structure probes (e.g. galaxy clustering and gravitational weak lensing) to the replicated observational ones, second) to independently input various local and non-local forms of perturbations to the primordial scalar power spectrum, third) to develop an independent set of codes for Fisher analysis of components, fourth) to develop and independent set of codes for implementing the principle component analysis on the obtained results from the Fisher analysis related codes, and fifth) to obtain the finalized forms of deviation, namely scalar modes (SMs).

There are three main python codes with the names basicfuncs.py, main.py, SMC.py;

basicfuncs.py --> contains the future and current surveys specifications and other independent parameters that need to be set before the start of calculations.

main.py --> contains the main part of the codes which input pertubations, calculates observational power spectrums and obtains fisher matrices.

SMC.py --> contains the third and final part of calculations which is used for reconstruction of scalar modes.
