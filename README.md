Here are the python scripts that were used for the analysis in the master's thesis [https://uu.diva-portal.org/smash/get/diva2:1912343/FULLTEXT01.pdf](https://uu.diva-portal.org/smash/get/diva2:1912343/FULLTEXT01.pdf), alongside plots of the results. AmpLambda.py and AmpXi.py contains the amplitude for charm Λ and Ξ baryon respectively, both of which are built on additional formulas found in Kinematics.py.

Tool_MonteCarlo.py was used to generate the pseudodata.

Tool_Fitting was used to fit the pseudodata and extract the estimated values, uncertanties and covariance matrix.

Tool_Plotting was used to plot the Dalitz plots and the 1D projections of the phase-space variables.

Tool_Normalisation.py was used to determine all the individual integrals found in Eq. (127) in the thesis.

The AmplitudesGeneral directory contains the same amplitudes but in an alternative forms (starting from differently orientated rest frame). Outdated optimization

Main references:

Helicity amplitudes for generic multi-body particle decays featuring multiple decay chains: [https://arxiv.org/abs/1911.10025](https://arxiv.org/abs/1911.10025)

Extracting maximum information from polarised baryon decays via amplitude analysis: the Λ+c→pK−π+ case: [https://arxiv.org/abs/2004.12318](https://arxiv.org/abs/2004.12318)

Amplitude analysis of the Λ+c→pK−π+ decay and Λ+c baryon polarization measurement in semileptonic beauty hadron decays: [https://arxiv.org/abs/2208.03262](https://arxiv.org/abs/2208.03262)


Currently translating the codes to C++
