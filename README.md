# pairwise_barycentres
Various implementations for unbalanced and balanced, biased and debiased pairwise barycentres.


## BarycentreDataProcessor
This class will store your data given a dictionary and the tree graph the barycentre is definited on. For the pairwise approach the graph doesn't mean much, but give flexibilty for the MOT versions. 

ToDo: Allow different type of grid input. ATM all grid have to be the same type;
(N, 2) (n1, n2, 2) or ((n1), (n2)).

ToDo: If only one density given then assume debiaing version

ToDo: Convergence of test_asym_bary_with_same_grid_uniform_density_with_debiasing is very sensitive and I'm not sure why. The debaising may improve the barycentre but I'm not aware of results on convergence.