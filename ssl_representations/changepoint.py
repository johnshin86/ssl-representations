import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from more_itertools import pairwise


"""Implementation of Energy statistics in torch.
References
----------
[1] https://en.wikipedia.org/wiki/Energy_distance
[2] Matteson, David S., and Nicholas A. James. "A nonparametric approach for multiple change point analysis of multivariate data." Journal of the American Statistical Association 109.505 (2014): 334-345.
"""

def distance_matrix(Z: Tensor, alpha: float = 2.0) -> Tensor:
	r"""Retrieves the distance matrix for an n x m tensor, where
	n is the number of samples and m is the number of features.
	
	Parameters
	----------
	Z: Tensor
		An n x m tensor of n samples with m features.

	alpha: float
		A float for the norm of the distance.

	Returns
	-------
	distances: Tensor

		an n x n tensor of distances with norm alpha.
	"""
	return torch.cdist(Z, Z, p = alpha)

def calculate_stats(x: float, y: float, xy: float, n: int, m: int) -> Tuple[float, float, float]:
	r"""

	Parameters
	----------
	xy: float
		Sum of distances between sets X and Y.
	x: float
		Sum of distances within set X.
	y: float
		Sum of distances within set Y.
	n: int
		The length of set X.
	m: int
		The length of set Y. 

	Returns
	-------
	e: float
		The E-statistic.
	t: float
		The test statistic. E.g., Q in the
	h: float
		The E coefficient of inhomogeneity.
	"""
	xy_avg =  xy / (n * m) if n > 0 and m > 0 else 0
	x_avg = x / (n ** 2) if n > 0 else 0
	y_avg = y / (m ** 2) if m > 0 else 0

	e = 2 * xy_avg - x_avg - y_avg
	t = (n * m/ (n + m)) * e
	h = (e / (2 * xy_avg) if xy_avg > 0 else 0.0)

	return e, t, h

def calculate_t_stats(distance_matrix: Tensor) -> Tensor:
	r"""Computes the t-statistic values given a distance matrix.

	Parameters
	----------
	distance_matrix: Tensor

		A distance matrix.
	
	Returns
	-------
	statistics: Tensor

		The t-statistic for a given value of tau. 
	"""

	statistics = torch.zeros(len(distance_matrix))

	xy = 0
	x = 0 
	y = 0

	for row in range(0, len(distance_matrix)):
		# for each row in the distance matrix, 
		# sum the partial row vector
		# with indices greater than and equal to row.
		y += torch.sum(distance_matrix[row, row:])

	for tau in range(0, len(distance_matrix)):
		statistics[tau] = calculate_stats(2 * x, 2 * y, xy, tau, len(distance_matrix) - tau)[1]
		#samples from 0, ..., tau - 1, compared to sample tau
		column_delta = torch.sum(distance_matrix[:tau, tau])
		#sample tau compared to samples tau, ... , kappa
		row_delta = torch.sum(distance_matrix[tau, tau:])

		#just compute raw sums and normalization is handled in calculate_stats
		xy = xy - column_delta + row_delta
		x = x + column_delta
		y = y - row_delta

	return statistics

def get_next_significant_change_point(
	distances: Tensor,
	change_points: List[int],
	memo: Dict[Tuple[int, int], Tuple[int, float]],
	pvalue: float,
	permutations: int,
	) -> Optional[int]:
	r"""
	Calculate the next significant change point.

	Return the next significant change point within windows bounded by pre-existing change points, if there are any.

	Parameters
	----------
	distances: Tensor

	change_points: List[int]

	memo: Dict[Tuple[int, int], Tuple[int, float]]

	pvalue: float
	
	permutations: int

	Returns
	-------
	"""

	windows = [0] + sorted(change_points) + [len(distances)]
	candidates: List[Tuple[int, float]] = []

	#iterate over every pairwise combination of windows. 
	for bounds in pairwise(windows):

		# if bounds have been memo-ized, add them to candidates
		if bounds in memo:

			candidates.append(memo[bounds])

		else:
		# if not, calculate t-stats for that bound and add to candidates
			a, b = bounds
			stats = calculate_t_stats(distances[a:b, a:b])
			idx = int(torch.argmax(stats))
			new = (idx + a, stats[idx])
			candidates.append(new)
			memo[bounds] = new

	# Get the best candidate. 
	best_candidate = max(candidates, key = lambda x: x[1])
	better_num = 0

	#Shuffle all the indices in the window, and get the corresponding distances
	#Calculate the t-stats for this shuffle.
	#Get the best shuffled value
	for _ in range(permutations):

		permute_t = []
		for a, b in pairwise(windows):
			row_indices = np.arange(b-a) + a
			np.random.shuffle(row_indices)
			shuffled_distances = distances[np.ix_(row_indices, row_indices)]
			stats = calculate_t_stats(shuffled_distances)
			permute_t.append(torch.max(stats))

		best = max(permute_t)

	#If there is a better value, count it

		if best >= best_candidate[1]:
			better_num += 1

	# The probability is the number of better_nums over permutations + 1. 
	probability = better_num / (permutations + 1)

	# If this is greater than the pvalue, then there was no change point with 
	# the given significance. 
	return best_candidate[0] if probability <= pvalue else None


def e_divisive(
	series: Tensor,
	pvalue: float = 0.05,
	permutations: int = 100,
	) -> List[int]:

	r"""Calculate the change points in a time series using the e divisive method.

	Parameters
	----------
	series: Tensor

		A series tensor of size N x M, where N is the number of samples and M is the number of features.

	pvalue: float

		The p value for the permutation test.

	permutations: int

		Number of permutations for the permutation test.

	Returns
	-------
	change_points: List[int]

		A list of change point indices.
	"""
	change_points : List[int] = []
	distances = distance_matrix(series)
	memo: Dict[Tuple[int, int], Tuple[int, float]] = {}

	while signficant_change_point := get_next_significant_change_point(distances, change_points, memo, pvalue, permutations):
		change_points.append(signficant_change_point)

	return change_points

# DEPRECATED FUNCTIONS BELOW

def argmax_Q(tensor_sequence: Tensor, alpha: float = 1.0, use_recursion: bool = False):
	r"""Estimates the location of a change point by findindg the point of maximum divergence.
	
	Parameters
	----------
	tensor_sequence: Tensor
		
		A sequence of tensors shaped n x m, where n are the number of samples and m is the number of features.

	alpha: float

		A float indicating the moment from the divergence computation.

	use_recursion: bool

		A boolean indicating whether to use the recursive formula that related Q(tau -1) to Q(tau) to save computation.

	Returns
	-------
	tau_hat: int
		
		The value of tau in which the maximum divergence was found.

	kappa_hat: int
		
		The value of kappa in which the maximum divergence was found. 
	"""
	n_samples, _ = tensor_sequence.shape

	grid_search = torch.zeros((n_samples, n_samples))

	#iterate over kappa first, so we can use the recursive divergence
	#as tau changes

	# Make kappa and tau start at one so 
	# slicing is never empty. 
	for kappa in range(1, n_samples):
		for tau in range(1, n_samples):
			if tau < kappa:

				X_tau = tensor_sequence[:tau, :]
				Y_tau_kappa = tensor_sequence[tau:kappa, :]

				if tau == 1:
					# cannot use recursive formula for first iteration.
					scaled_divergences = empirical_divergence(X_tau, Y_tau_kappa, alpha = alpha, scaled = True)
					grid_search[tau][kappa] = sum(scaled_divergences)

				else:

					if use_recursion:
						scaled_divergences = recursive_divergence(scaled_divergences, X_tau, Y_tau_kappa, tau = tau, kappa = kappa, alpha = alpha)
						grid_search[tau][kappa] = sum(scaled_divergences)
					else:
						scaled_divergences = empirical_divergence(X_tau, Y_tau_kappa, alpha = alpha, scaled = True)
						grid_search[tau][kappa] = sum(scaled_divergences)

			else:
				pass

	tau_hat, kappa_hat = (grid_search == torch.max(grid_search)).nonzero().flatten().numpy()

	return tau_hat, kappa_hat

def empirical_divergence(first_sequence: Tensor, second_sequence: Tensor, alpha: float = 1.0, scaled: bool = True):
	r"""Takes two sets of input sequences of the same length and measures
	the divergence between them based on Euclidean distances.

	Computes the equation:

	E(X_n, Y_m, alpha) = 2/(m * n) \sum_{i = 1}^n \sum_{j=1}^m |X_i - Y_j|^{alpha}
						- (n choose 2)^{-1} \sum_{1 \leq i < k < n} |X_i - X_k|^{alpha}
						- (m choose 2)^{-1} \sum_{1 \leq j < k < m} |Y_j - Y_k|^{alpha}

	If the scaled boolean is true, scales this by a normalization factor:

	Q(X_n, Y_m, alpha) = (m * n)/ (m + n) E(X_n, Y_m, alpha)

	Parameters
	----------
	first_sequence: Tensor
		A tensor of sequences, where each row is a sample, and columns are features.

	second_sequence: Tensor
		A tensor of sequences, where each row is a sample, and columns are features.

	alpha: float
		A float indicating the moment of the divergence.

	scaled: bool
		Whether to return the scaled divergence.


	Returns
	-------
	divergence: list[Tensor]
		A list containing the three terms of the divergence. Scaled by a normalization factor if the scaled boolean is true.
	"""

	if len(first_sequence.shape) == 1:
		first_sequence = first_sequence.unsqueeze(0)

	if len(second_sequence.shape) == 1:
		second_sequence = second_sequence.unsqueeze(0)

	n, _ = first_sequence.shape 
	m, _ = second_sequence.shape

	n_choose_2 = 1./2.*n*(n-1)
	m_choose_2 = 1./2.*m*(m-1)

	#Need to handle length 1 cases for cdist. 


	first_matrix = torch.cdist(first_sequence, second_sequence, p = alpha)
	first_sum = torch.sum(first_matrix)

	# If first  or second sequence is a singleton, the sum should auto-collapse to zero. 

	if n == 1:
		second_sum = 0.0

	else:
		second_matrix = torch.cdist(first_sequence, first_sequence, p = alpha)
		second_matrix_upper = torch.triu(second_matrix, diagonal = 1)
		second_sum = torch.sum(second_matrix_upper)

	if m == 1:
		third_sum = 0.0

	else:
		third_matrix = torch.cdist(second_sequence, second_sequence, p = alpha)
		third_matrix_upper = torch.triu(third_matrix, diagonal = 1)
		third_sum = torch.sum(third_matrix_upper)

	divergence_terms = [first_sum, 
						second_sum if second_sum == 0.0 else -1./ n_choose_2 * second_sum, 
						third_sum if third_sum == 0.0 else -1./m_choose_2 * third_sum]

	if scaled == True:
		
		divergence_terms = [(m*n)/(m + n) * term for term in divergence_terms]

	return divergence_terms
	
def recursive_divergence(previous_divergences: list[Tensor], X_tau: Tensor, Y_tau_kappa: Tensor, tau: int, kappa: int, alpha: float = 1.0):
	r"""For fixed kappa, we can use a recursive formula to compute Q(X_tau, Y_tau (kappa), alpha) as a function of
	Q(X_{tau - 1}, Y_{tau -1} (kappa), alpha) using the distances {|Z_tau - Z_j|^{alpha} : 1 <= j < tau}.

	We have that:

	X_tau = {Z_1, ... , Z_tau}
	Y_tau (kappa) = {Z_{tau + 1}, ... , Z_{kappa}}

	Where n = tau and m = kappa - tau.

	We have the formula for Q at a given tau, kappa, and alpha:

	Q(X_tau, Y_tau (kappa), alpha) = 
		2/(m + n) \sum_{i = 1}^{tau} \sum_{j = tau + 1}^{kappa} |Z_i - Z_j|^{alpha}
		- mn / (m + n) {n choose 2}^{-1} \sum_{1 <= i < k <= tau} |Z_i - Z_k|^{alpha}
		- mn / (m + n) {m choose 2}^{-1} \sum_{tau + 1 <= j < k <= kappa} |Z_j - Z_k|^{alpha}

		= Q_first (X_tau, Y_tau(kappa), alpha)
		+ Q_second (X_tau, alpha)
		+ Q_third (Y_tau(kappa), alpha)

	For the next step, we have -
	Q(X_{tau + 1}, Y_{tau + 1}, alpha) = 
		2 / (m + n) \sum_{i = 1}^{tau + 1} \sum_{j = tau + 2}^{kappa} |Z_i - Z_j|^{alpha} 
		- (n + 1)(m - 1)/(m + n) {n + 1 choose 2}^{-1} \sum_{1 <= i < k <= tau + 1} |Z_i - Z_k|^{alpha}
		- (n + 1)(m - 1)/(m + n) {m - 1 choose 2}^{-1} \sum_{tau + 2 <= j < k <= kappa} |Z_j - Z_k|^{alpha}

	We can substitute the Q(X_tau, Y_tau(kappa), alpha) terms -
	Q(X_{tau + 1}, Y_{tau + 1}, alpha) = 

		For the first term, the tau + 1 term has moved from the j sum to the i sum. So we need to add
		all the comparisons to j, while removing the previous comparisons to i. 
		Q_first(X_tau, Y_tau(kappa), alpha) + 2/ (m + n) \sum_{j = tau+2}^{kappa} |Z_j - Z_{tau + 1}|^{alpha} - 2/ (m + n) \sum_{i = 1}^{tau} |Z_i - Z_{tau + 1}|^{alpha}


		The second term is a bit tricky, since it is dependent on the previous sum
		but requires a different prefactor. The previous sum is normalized to remove the previous prefactor,
		then the new prefactor is applied. The new term is added with the new prefactor.
		- ((m * n) * (n + 1)(m - 1) Q_second +  (n + 1)(m - 1)/(m + n)  {n + 1 choose 2}^{-1} \sum_{1 <= i < k = tau + 1} |Z_i - Z_k|^{alpha})

		The third term is a also bit tricky, since terms need to be removed from the previous sum,
		with it's previous normalization. We remove the term from the old sum using the old normalization. 
		We then apply the new normalization to the final new result.

		- (m * n) * (n + 1)(m - 1) ( Q_third - mn / (m + n) {m choose 2}^{-1} \sum_{tau + 1 = j < k <= kappa} |Z_j - Z_k|^{alpha})

	For further clarity, we now write tau in terms of tau - 1, starting with writing Q(X_{tau - 1}, Y_{tau - 1}(kappa), alpha):

	n = tau - 1
	m = kappa - (tau - 1)

	Q(X_{tau - 1}, Y_{tau - 1}(kappa), alpha) =
		2/(m + n) \sum_{i = 1}^{tau - 1} \sum_{j = tau}^{kappa} |Z_i - Z_j|^{alpha}
		- mn / (m + n) {n choose 2}^{-1} \sum_{1 <= i < k <= tau - 1} |Z_i - Z_k|^{alpha}
		- mn / (m + n) {m choose 2}^{-1} \sum_{tau <= j < k <= kappa} |Z_j - Z_k|^{alpha}

		= Q_first (X_{tau - 1}, Y_{tau - 1}(kappa), alpha)
		+ Q_second (X_{tau - 1}, alpha)
		+ Q_third (Y_{tau - 1}(kappa), alpha)

	Q(X_{tau}, Y_{tau}, alpha) = 
		The tau term has moved from j to i. We add the comparisons to j and remove the comparisons from i. 
		Q_first + 2/ (kappa) \sum_{j = tau + 1}^{kappa} |Z_j - Z_{tau}|^{alpha} - 2/ (kappa) \sum_{i = 1}^{tau - 1} |Z_i - Z_{tau}|^{alpha}
		- ((m * n) * (n + 1)(m - 1) Q_second +  (kappa - tau)/(kappa * (tau + 1) * 1/2) \sum_{1 <= i < k = tau} |Z_i - Z_k|^{alpha})
		- (m * n) * (n + 1)(m - 1) ( Q_third - (tau - 1)/(kappa * (kappa - tau)) \sum_{tau = j < k <= kappa} |Z_j - Z_k|^{alpha})

	Parameters
	----------
	previous_divergences: list[Tensor]
		The previous three divergence values computed at value tau.

	alpha: float
		A float value for the moment.

	Returns
	-------
	next_divergences: list[Tensor]
		The next three divergence terms with tau incremented by 1.	
	"""

	# X_tau will actually never be a singleton since we just added to it, but Y_tau can be a singleton. 
	if len(X_tau.shape) == 1:
		X_tau = X_tau.unsqueeze(0)

	if len(Y_tau_kappa.shape) == 1:
		Y_tau_kappa = Y_tau_kappa.unsqueeze(0)

	prev_n = tau - 1
	prev_m = kappa - (tau - 1)

	n = tau
	m = kappa - tau

	Q_first = previous_divergences[0]
	Q_second = previous_divergences[1]
	Q_third = previous_divergences[2]


	# Add the j comparisons
	first_terms_add = torch.cdist(X_tau[-1, :].unsqueeze(0), Y_tau_kappa, p = alpha)
	# Remove the i comparisons
	first_terms_subtract = torch.cdist(X_tau[:-1, :], X_tau[-1, :].unsqueeze(0), p = alpha)
	# X_tau cannot be a singleton in our call of the recursive function in argmax_Q
	second_terms = torch.cdist(X_tau[:-1, :], X_tau[-1, :].unsqueeze(0), p = alpha)
	# Y_tau_kappa can be a singleton, but we handled this above. 
	third_terms = torch.cdist(X_tau[-1, :].unsqueeze(0), Y_tau_kappa, p = alpha)

	first_sum = Q_first + 2./kappa * torch.sum(first_terms_add) - 2./kappa * torch.sum(first_terms_subtract)
	second_sum = (prev_n * prev_m) * (n * m) * Q_second + (kappa - tau)/(kappa * (tau + 1) * 1./2.) * torch.sum(second_terms)
	third_sum = (prev_n * prev_m) * (n * m) * (Q_third - (tau - 1)/(kappa * (kappa - tau)) * torch.sum(third_terms)) 

	return [first_sum, -second_sum, -third_sum]



