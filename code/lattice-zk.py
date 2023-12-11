#!/usr/bin/env python

import sys
import sage.all as sg
import numpy as np
import random
import time
from itertools import chain
from sage.stats.distributions.discrete_gaussian_polynomial import DiscreteGaussianDistributionPolynomialSampler

kappa = 128     # security parameter
d = 128         # degree of polynomial ring
q = 2**32 - 99  # modulus prime

IR = sg.IntegerRing()
P = sg.PolynomialRing(IR, 't')
t = P.gen()
R = sg.QuotientRing(P, t**d + 1, 'X')
X = R.gen()

IR_q = sg.IntegerModRing(q)
P_q = sg.PolynomialRing(IR_q, 't')
t_q = P_q.gen()
R_q = sg.QuotientRing(P_q, t_q**d + 1, 'X')
X_q = R_q.gen()

#------------------------- Representation conversions -------------------------

def Zq_2_ZZ(r):
    """
    Return r_p = r mod+- q (for odd interger q)
    as the unique element in range [-((q-1)/2), (q-1)/2]
    such that r_p = r mod q
    """
    if r <= (q - 1) // 2:
        return int(r)
    else:
        return int(r) - q


def Rql_2_ZZl(vec):
    """Convert polynomial vector to coefficient vector"""
    return sg.vector(sg.ZZ, chain.from_iterable(
        [[Zq_2_ZZ(coeff) for coeff in poly.list()] for poly in vec]))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def ZZl_2_Rql(vec):
    """Convert coefficient vector to polynomial vector"""
    assert(len(vec) % d == 0)
    return sg.vector(R_q, chunks(list(vec), d))


def stack_vec_Rql(vecs):
    """Stack sage vectors"""
    return sg.vector(R_q, chain(*vecs))


def IM(n):
    """Identity matrix"""
    return sg.identity_matrix(R_q, n)


def zv(n):
    """Zero vector"""
    return sg.zero_vector(R_q, n)


def ZM(n, m):
    """Zero matrix"""
    return sg.zero_matrix(R_q, n, m)


#---------------------------- Norm and Dot product ----------------------------

def inf_norm_Zq(x):
    """Compute the infitity norm of an integer mod q according to definition"""
    if x <= (q - 1) // 2:
        return int(x)
    else:
        return -(int(x) - q)


def p_norm_Rl(vec, p):
    """Compute p-norm of polynomial vector in R^l according to definition"""
    return sum([p_norm_R(poly, p) for poly in vec])**(1 / p)
         
         
def p_norm_R(poly, p):
    """Compute p-norm of a polynomial in R according to definition"""
    return sum([int(coeff)**p for coeff in poly])**(1 / p)
         
         
def norm_Rl(vec):
    """Compute Euclidean norm of polynomial vector in R^l"""
    return p_norm_Rl(vec, 2)
         

def p_norm_Rq(poly, p):
    """Compute p-norm of a polynomial in R_q according to definition"""
    return sum([inf_norm_Zq(coeff)**p for coeff in poly])**(1 / p)


def p_norm_Rql(vec, p):
    """Compute p-norm of polynomial vector in R_q^l according to definition"""
    return sum([p_norm_Rq(poly, p)**p for poly in vec])**(1 / p)


def norm_Rql(vec):
    """Compute Euclidean norm of polynomial vector in R_q^l according to definition"""
    return p_norm_Rql(vec, 2)


def norm_Rql_bound(l, max_coeff):
    """
    Compute upper bound of Euclidean norm of polynomial vector in R_q^l
    with only inf_norm_Zq <= max_coeff coefficients
    """
    return max_coeff * sg.sqrt(d * l)


def inf_norm_Rl(vec):
    """Compute infinity norm of a polynomial vector in R^l"""
    return max([max(poly) for poly in vec])


def inf_norm_Rql(vec):
    """Compute infinity norm of a polynomial vector in R_q^l"""
    return max([max([inf_norm_Zq(coeff) for coeff in poly]) for poly in vec])


def dot_Rql(v, w):
    """Compute dot product of polynomial vector in R_q^l according to definition"""
    return Rql_2_ZZl(v).dot_product(Rql_2_ZZl(w))


#------------------------------ Random Sampling ------------------------------

class GaussianSampler:
    """Sample from discrete gaussian distribution"""
    def __init__(self, R, s, l):
        self.R = R
        self.D = DiscreteGaussianDistributionPolynomialSampler(self.R, n=d, sigma=s)
        self.l = l

    def get(self):
        return sg.vector(self.R, [self.D() for _ in range(self.l)])


def rand_ZZ_mat_binomial(rows, cols, k):
    """Sample an integer matrix from a binomial distribution"""
    return sg.matrix(sg.ZZ, np.random.binomial(k, 0.5, (rows, cols)))


def rand_Rql(l):
    """Sample random polynomial vector in R_q^l"""
    return sg.vector(R_q, [R_q(random.choices(range(q), k=d)) for _ in range(l)])


def rand_Rql_small(l, max_coeff):
    """Sample random polynomial vector in R_q^l with only inf_norm_Zq <= max_coeff coefficients"""
    return sg.vector(R_q, [R_q(random.choices(range(-max_coeff, max_coeff + 1), k=d)) for _ in range(l)])


def rand_Rql_bin(l):
    """Sample random binary polynomial vector in R_q^l"""
    return sg.vector(R_q, [R_q(random.choices([0, 1], k=d)) for _ in range(l)])


def rand_Rql_first_zero(l):
    """Sample random polynomial vector in R_q^l where first element (constant coefficient) is 0"""
    return sg.vector(R_q, [R_q([0] + random.choices(range(q), k=d - 1)) for _ in range(l)])


def rand_Rq_mat(rows, cols):
    """Sample random polynomial matrix in R_q^{m x n} with only inf_norm_Zq <= 1 coefficients"""
    return sg.matrix(R_q, [[R_q(random.choices(range(q), k=d)) for col in range(cols)] for row in range(rows)])


def rand_Rq_mat_first_zero(rows, cols):
    """Sample random polynomial matrix in R_q^l where first element (constant coefficient) is 0"""
    return sg.matrix(R_q, [[R_q([0] + random.choices(range(q), k=d - 1)) for col in range(cols)]
                           for row in range(rows)])


def rand_Zq_mat(rows, cols):
    """Sample random matrix in Z_q^{m x n}"""
    return sg.matrix(IR_q, [[IR_q(random.randrange(q)) for col in range(cols)] for row in range(rows)])

#----------------------- Rejection Sampling Algorithms -----------------------

def rej1_M(gamma):
    return sg.exp(sg.sqrt(2 * (kappa + 1) / sg.log(sg.e, 2)) * (1 / gamma) + (1 / (2 * gamma**2)))


def rej1(z, v, std, M):
    """Rej1 rejection sampling algorithm"""
    v_norm = norm_Rql(v)
    expr = (1 / M) * sg.exp(((-2) * dot_Rql(z, v) + (v_norm**2)) / (2 * (std**2)))
    u = random.random()
    return u > expr


def rej1_M_test(std, v_norm):
    """Compute repetition rate M for Rej1 algorithm"""
    return sg.exp((28 * std * v_norm + v_norm**2) / (2 * (std**2)))


def test_rej1(num_tests):
    ell = 8  # polynomial ring vector dimension

    # Generate dummy v (in the full protocol v is obtained by multiplying the
    # challenge c with the secret s)
    v = rand_Rql(ell)
    v_norm = norm_Rql(v)
    print(f"v_norm = {float(v_norm)}")

    gamma = 13
    std = gamma * v_norm  # standard deviation
    print(f"std = {float(std)}")
    M = rej1_M_test(std, v_norm)  # repetition rate
    print(f"M = {float(M)}")

    smplr = GaussianSampler(R, std, ell)

    rejections = 0
    for _ in range(num_tests):
        y = smplr.get()
        z = y + v

        if rej1(z, v, std, M):
            rejections += 1

    if rejections == num_tests:
        print("All rejected")
    else:
        success_prob = 1 - (rejections / num_tests)
        print(f"Expected success prob: {float((1 - 2**-128) / M)}")
        print(f"Success prob: {float(success_prob)}")
        avg_reps = 1 / success_prob
        print(f"Avg repetitions: {float(avg_reps)}")


#print(test_rej1(500))


def rej2_M(gamma):
    return sg.exp((1 / (2 * gamma**2)))


def rej2(z, v, std, M):
    """Rej2 rejection sampling algorithm"""
    if dot_Rql(z, v) < 0:
        return True

    v_norm = norm_Rql(v)
    expr = (1 / M) * sg.exp(((-2) * dot_Rql(z, v) + (v_norm**2)) / (2 * (std**2)))
    u = random.random()
    return u > expr


def rej2_M_test(std, v_norm):
    """Compute repetition rate M for Rej2 algorithm"""
    return sg.exp((v_norm**2) / (2 * (std**2)))


def test_rej2(num_tests):
    ell = 8  # polynomial ring vector dimension

    # Generate dummy v (in the full protocol v is obtained by multiplying the
    # challenge c with the secret s)
    v = rand_Rql(ell)
    v_norm = norm_Rql(v)
    print(f"v_norm = {float(v_norm)}")

    gamma = 0.675
    std = gamma * v_norm  # standard deviation
    print(f"std = {float(std)}")
    M = rej2_M_test(std, v_norm)  # repetition rate
    print(f"M = {float(M)}")

    smplr = GaussianSampler(R, std, ell)

    rejections = 0
    for _ in range(num_tests):
        y = smplr.get()
        z = y + v

        if rej2(z, v, std, M):
            rejections += 1

    if rejections == num_tests:
        print("All rejected")
    else:
        success_prob = 1 - (rejections / num_tests)
        print(f"Expected success prob: {float(1/(2*M))}")
        print(f"Success prob: {float(success_prob)}")
        avg_reps = 1 / success_prob
        print(f"Avg repetitions: {float(avg_reps)}")


#print(test_rej2(500))


def rej_bimodal_M(gamma):
    return sg.exp((1 / (2 * gamma**2)))


def rej_bimodal(z, v, std, M):
    """Bimodal rejection sampling algorithm"""
    v_norm = norm_Rql(v)
    expr = 1 / (M * sg.exp(-(v_norm**2) / (2 * (std**2)))
                * sg.cosh(dot_Rql(z, v) / (std**2)))
    u = random.random()
    return u > expr


def rej_bimodal_M_test(std, v_norm):
    """Compute repetition rate M for bimodal rejection sampling algorithm"""
    return sg.exp((v_norm**2) / (2 * (std**2)))


def test_rej_bimodal(num_tests):
    ell = 8  # polynomial ring vector dimension

    # Generate dummy v (in the full protocol v is obtained by multiplying the
    # challenge c with the secret s)
    v = rand_Rql(ell)
    v_norm = norm_Rql(v)
    print(f"v_norm = {float(v_norm)}")

    gamma = 0.675
    std = gamma * v_norm  # standard deviation
    print(f"std = {float(std)}")
    M = rej_bimodal_M_test(std, v_norm)  # repetition rate
    print(f"M = {float(M)}")

    smplr = GaussianSampler(R, std, ell)

    rejections = 0
    for _ in range(num_tests):
        beta = random.choice([0, 1])
        y = smplr.get()
        z = y + (-1)**beta * v

        if rej_bimodal(z, v, std, M):
            rejections += 1

    if rejections == num_tests:
        print("All rejected")
    else:
        success_prob = 1 - (rejections / num_tests)
        print(f"Expected success prob: {float(1/(M))}")
        print(f"Success prob: {float(success_prob)}")
        avg_reps = 1 / success_prob
        print(f"Avg repetitions: {float(avg_reps)}")


#print(test_rej_bimodal(500))

#---------------------------- Challenge Sampling -----------------------------

def sigma_n1(poly):
    """Apply sigma_(-1) automorphism to a polynomial"""
    return R_q([poly[0]] + [-coeff for coeff in reversed(list(poly)[1:])])


def sigma_n1_vec(vec):
    """Apply sigma_(-1) automorphism to a vector of polynomials"""
    return sg.vector(R_q, [sigma_n1(poly) for poly in vec])


def sigma_n1_mat(mat):
    """Apply sigma_(-1) automorphism to a matrix of polynomials"""
    return sg.matrix(R_q, [[sigma_n1(poly) for poly in vec] for vec in mat])


k_sigma_1 = 1
k_sigma_n1 = 2


def sigma_n1_struct_poly(poly):
    """Create the <x>_(sigma_(-1)) structure for a polynomial"""
    return sg.vector(R_q, [poly, sigma_n1(poly)])


def sigma_n1_struct_vec(vec):
    """Create the <x>_(sigma_(-1)) structure for a vector (doubles vector size)"""
    return sg.vector(R_q, [f(poly) for poly in vec for f in (lambda x: x, sigma_n1)])


eta_challenge_v1 = 27  # smallness bound


def rand_c_v1():
    """
    Generate random challenge polynomial in R_q^l
    with only inf_norm_Zq <= 1 coefficients
    """
    return R_q(random.choices([-1, 0, 1], k=d))


def check_c_v1(c):
    """Return true if challenge polynomial fits the criteria"""
    k = 32
    c_pow_k = c**k
    norm = p_norm_Rq(sigma_n1(c_pow_k) * c_pow_k, 1)
    return bool(norm**(1 / (2 * k)) <= eta_challenge_v1)


def get_challenge_v1():
    c = rand_c_v1()
    while not check_c_v1(c):
        c = rand_c_v1()
    return c


def test_small_norm_lemma(get_challenge_func):
    """
    Test if norm does not increase by more than fixed factor
    after multiplying a challenge polynomial to a random vector in Rq^l
    """
    test_c = get_challenge_func()
    test_r = rand_Rql(8)
    norm_r = norm_Rql(test_r)
    print(f"r norm: {float(norm_r)}")
    res = test_c * test_r
    norm_res = norm_Rql(res)
    print(f"result norm: {float(norm_res)}")
    norm_increase = norm_res / norm_r
    print(f"norm increase: {float(norm_increase)}")
    k = 32
    norm_bound = p_norm_Rq(sigma_n1(test_c**k) * test_c**k, 1)**(1 / (2 * k))
    print(f"norm bound: {float(norm_bound)}")
    assert(norm_increase < norm_bound)


#print(test_small_norm_lemma(get_challenge_v1))


eta_challenge_v2 = 59  # smallness bound


def rand_c_v2():
    """
    Generate random challenge polynomial in R_q^l
    with only inf_norm_Zq <= 2 coefficients.
    In order to satisfy sigma_n1(c) == c it must have symmetric coefficients
    """
    first_half = random.choices([-2, -1, 0, 1, 2], k=d // 2)
    second_half = [-coeff for coeff in first_half[1:]]
    second_half.reverse()
    second_half = [0] + second_half
    return R_q(first_half + second_half)


def check_c_v2(c):
    """Return true if challenge polynomial fits the criteria"""
    k = 32
    c_pow_k = c**k
    norm = p_norm_Rq(sigma_n1(c_pow_k) * c_pow_k, 1)
    return bool(norm**(1 / (2 * k)) <= eta_challenge_v2)


def get_challenge_v2():
    c = rand_c_v2()
    while not check_c_v2(c):
        c = rand_c_v2()
    return c


#print(test_small_norm_lemma(get_challenge_v2))


#-------- ABDLOP Commitment Opening Proof with Linear Proofs over Rq ---------

def abdlop_commit(m1, m2, ell, n, N,
                  rej_u_1, rej_u_2, get_challenge_u,
                  std1, rep_M1, std2, rep_M2,
                  s1, s2, m, A1, A2, B, tA, tB,
                  R1, r0):
    """
    ABDLOP opening proof with linear proofs over Rq
    """

    # ================================ Prover =================================
    y1_smplr = GaussianSampler(R_q, std1, m1)
    y1 = y1_smplr.get()
    y2_smplr = GaussianSampler(R_q, std2, m2)
    y2 = y2_smplr.get()

    w = A1 * y1 + A2 * y2
    v = R1 * stack_vec_Rql([y1, -B * y2])

    # =============================== Verifier ================================
    c = get_challenge_u()

    # ================================ Prover =================================
    cs1 = c * s1
    z1 = cs1 + y1
    if rej_u_1(z1, cs1, std1, rep_M1):
        return "Rejected"

    cs2 = c * s2
    z2 = cs2 + y2
    if rej_u_2(z2, cs2, std2, rep_M2):
        return "Rejected"

    # =============================== Verifier ================================
    cond1 = (
        norm_Rql(z1) <= std1 *
        sg.sqrt(
            2 *
            m1 *
            d)) and (
        norm_Rql(z2) <= std2 *
        sg.sqrt(
            2 *
            m2 *
            d))
    cond2 = (A1 * z1 + A2 * z2 - c * tA == w)
    cond3 = (R1 * stack_vec_Rql([z1, c * tB - B * z2]) + c * r0 == v)

    return cond1 and cond2 and cond3


def test_abdlop_commit():
    """Call abdlop_commit with example parameters"""

    # Size parameters
    m1 = 8   # size of s1
    m2 = 25  # size of s2
    ell = 2  # size of m
    n = 9    # dimension of the Module-SIS problem
    N = 1    # number of linear functions to prove

    # Algorithm parameters
    is_sigma_1 = False  # automorphism: True for sigma_1, False for sigma_n1
    if is_sigma_1:
        eta = eta_challenge_v1              # challenge norm upper bound
        get_challenge_u = get_challenge_v1  # challenge sampling algorithm
    else:
        eta = eta_challenge_v2
        get_challenge_u = get_challenge_v2

    rej_u_1 = rej1  # rejection sampling algorithm
    rej_u_2 = rej2

    gamma1 = 19  # repetition rate parameter
    nu_s1 = 1    # s1 max coefficient
    alpha_s1 = norm_Rql_bound(m1, nu_s1)  # s1 norm upper bound
    std1 = gamma1 * eta * alpha_s1  # standard deviation
    rep_M1 = rej1_M(gamma1)  # repetition rate

    gamma2 = 2
    nu_s2 = 1
    alpha_s2 = norm_Rql_bound(m2, nu_s2)
    std2 = gamma2 * eta * alpha_s2
    rep_M2 = rej2_M(gamma2)

    global rep_rate_abdlop_commit
    rep_rate_abdlop_commit = rep_M1 * 2 * rep_M2

    # Private information
    s1 = rand_Rql_small(m1, nu_s1)  # ABDLOP Ajtai part message vector
    s2 = rand_Rql_small(m2, nu_s2)  # ABDLOP randomness vector
    m = rand_Rql(ell)               # ABDLOP BDLOP part message vector

    # Public information
    A1 = rand_Rq_mat(n, m1)   # ABDLOP A1
    A2 = rand_Rq_mat(n, m2)   # ABDLOP A2
    B = rand_Rq_mat(ell, m2)  # ABDLOP B

    # ABDLOP commitment
    tA_tB = A1.stack(ZM(ell, m1)) * s1 + \
        A2.stack(B) * s2 + stack_vec_Rql([zv(n), m])
    tA = tA_tB[:n]  # ABDLOP tA
    tB = tA_tB[n:]  # ABDLOP tB

    # Linear functions
    R1 = rand_Rq_mat(N, m1 + ell)
    r0 = -(R1 * stack_vec_Rql([s1, m]))

    return abdlop_commit(m1, m2, ell, n, N,
                         rej_u_1, rej_u_2, get_challenge_u,
                         std1, rep_M1, std2, rep_M2,
                         s1, s2, m, A1, A2, B, tA, tB,
                         R1, r0)


def repeat_until_accept(protocol_func):
    """Repeat protocol until it is not rejected anymore"""
    result = protocol_func()
    reps = 1
    while result == "Rejected":
        result = protocol_func()
        reps += 1
    return {"success": result, "reps": reps}


def measure_rep_rate(protocol_func, exp_rep_rate, runs=100):
    """Measure repetition rate of protocol"""
    print(f"Expected repetition rate: {float(exp_rep_rate)}")
    results = [protocol_func() for _ in range(runs)]
    result_dict = {res: results.count(res) for res in set(results)}
    print(f"Measured repetition rate: {float(runs / result_dict[True])}")
    print(f"Results: {result_dict}")



#print(measure_rep_rate(test_abdlop_commit, rep_rate_abdlop_commit))


#--------------------------- Linear Proofs over Zq ----------------------------

def abdlop_linear(m1, m2, ell, n, N, M, lambd,
                  rej_u_1, rej_u_2, get_challenge_u,
                  std1, rep_M1, std2, rep_M2,
                  s1, s2, m, s, A1, A2, B, Bg, tA, tB,
                  R1, r0, u1, u0):
    """ABDLOP linear proofs over Zq"""

    # ================================ Prover =================================
    g = rand_Rql_first_zero(lambd)
    tg = Bg * s2 + g

    # =============================== Verifier ================================
    Y = rand_Zq_mat(lambd, M)

    # ================================ Prover =================================
    h = g + Y * stack_vec_Rql([u1[i].row() * s + u0[i] for i in range(M)])

    # Append Bg to B, g to m and tg to tB
    tB = stack_vec_Rql([tB, tg])
    B = B.stack(Bg)
    m = stack_vec_Rql([m, g])
    ell = ell + lambd

    # To prove well-formedness of h[1],...,h[lambd]
    u1_mat = u1[0].row()
    for i in range(1, M):
        u1_mat = u1_mat.stack(u1[i].row())
    V1 = Y * u1_mat
    v0 = Y * stack_vec_Rql([u0[i] for i in range(M)]) - h
    R1 = R1.augment(ZM(N, lambd)).stack(
        V1.augment(IM(lambd)))
    r0 = stack_vec_Rql([r0, v0])

    # Run ABDLOP commit sub-protocol
    cond1 = abdlop_commit(m1, m2, ell, n, N,
                          rej_u_1, rej_u_2, get_challenge_u,
                          std1, rep_M1, std2, rep_M2,
                          s1, s2, m, A1, A2, B, tA, tB,
                          R1, r0)
    if cond1 == "Rejected":
        return "Rejected"

    # =============================== Verifier ================================
    # constant coefficient is 0
    cond2 = all(h[i][0] == 0 for i in range(lambd))

    return cond1 and cond2


def test_abdlop_linear():
    """Call abdlop_linear with example parameters"""

    # Size parameters
    m1 = 16    # size of s1
    m2 = 47    # size of s2
    ell = 8    # size of m
    n = 4      # dimension of the Module-SIS problem
    N = 1      # number of linear functions over Rq
    M = 1      # number of linear functions over Zq
    lambd = M  # number of random challenges for the M linear functions (must be M)

    # Algorithm parameters
    is_sigma_1 = False  # automorphism: True for sigma_1, False for sigma_n1
    if is_sigma_1:
        eta = eta_challenge_v1              # challenge norm upper bound
        get_challenge_u = get_challenge_v1  # challenge sampling algorithm
    else:
        eta = eta_challenge_v2
        get_challenge_u = get_challenge_v2

    rej_u_1 = rej1  # rejection sampling algorithm
    rej_u_2 = rej2

    gamma1 = 19  # repetition rate parameter
    nu_s1 = 1    # s1 max coefficient
    alpha_s1 = norm_Rql_bound(m1, nu_s1)  # s1 norm upper bound
    std1 = gamma1 * eta * alpha_s1  # standard deviation
    rep_M1 = rej1_M(gamma1)  # repetition rate

    gamma2 = 2
    nu_s2 = 1
    alpha_s2 = norm_Rql_bound(m2, nu_s2)
    std2 = gamma2 * eta * alpha_s2
    rep_M2 = rej2_M(gamma2)

    global rep_rate_abdlop_linear
    rep_rate_abdlop_linear = rep_M1 * 2 * rep_M2

    # Private information
    s1 = rand_Rql_small(m1, nu_s1)  # ABDLOP Ajtai part message vector
    s2 = rand_Rql_small(m2, nu_s2)  # ABDLOP randomness vector
    m = rand_Rql(ell)               # ABDLOP BDLOP part message vector
    s = stack_vec_Rql([s1, m])

    # Public information
    A1 = rand_Rq_mat(n, m1)      # ABDLOP A1
    A2 = rand_Rq_mat(n, m2)      # ABDLOP A2
    B = rand_Rq_mat(ell, m2)     # ABDLOP B
    Bg = rand_Rq_mat(lambd, m2)  # randomness matrix

    # ABDLOP commitment
    tA_tB = (A1.stack(ZM(ell, m1)) * s1
            + A2.stack(B) * s2
            + stack_vec_Rql([zv(n), m]))
    tA = tA_tB[:n]  # ABDLOP tA
    tB = tA_tB[n:]  # ABDLOP tB

    # Linear functions
    R1 = rand_Rq_mat(N, m1 + ell)
    r0 = -(R1 * stack_vec_Rql([s1, m]))
    u1 = [rand_Rql(m1 + ell) for i in range(M)]
    u0 = [-u1[i].row() * s for i in range(M)]

    return abdlop_linear(m1, m2, ell, n, N, M, lambd,
                         rej_u_1, rej_u_2, get_challenge_u,
                         std1, rep_M1, std2, rep_M2,
                         s1, s2, m, s, A1, A2, B, Bg, tA, tB,
                         R1, r0, u1, u0)



#print(measure_rep_rate(test_abdlop_linear, rep_rate_abdlop_linear))


#--------------------- Single Quadratic Relation over Rq ---------------------

def abdlop_single_quadratic(m1, m2, ell, n,
                            is_sigma_1, rej_u_1, rej_u_2, get_challenge_u,
                            std1, rep_M1, std2, rep_M2,
                            s1, s2, m, A1, A2, B, b, tA, tB,
                            R2, r1, r0):
    """ABDLOP proof of single quadratic relation"""
    
    #================================ Prover =================================
    s = stack_vec_Rql([s1, m])
    if not is_sigma_1:
        s = sigma_n1_struct_vec(s)
    
    y1_smplr = GaussianSampler(R_q, std1, m1)
    y1 = y1_smplr.get()
    y2_smplr = GaussianSampler(R_q, std2, m2)
    y2 = y2_smplr.get()

    w = A1 * y1 + A2 * y2
    y = stack_vec_Rql([y1, -B*y2])
    if not is_sigma_1:
        y = sigma_n1_struct_vec(y)
    g1 = s.dot_product(R2 * y) + y.dot_product(R2 * s) + r1.dot_product(y)
    t = b.dot_product(s2) + g1
    v = y.dot_product(R2 * y) + b.dot_product(y2)

    #=============================== Verifier ================================
    if is_sigma_1:
        c = get_challenge_v1()
    else:
        c = get_challenge_v2()

    #================================ Prover =================================
    cs1 = c * s1
    z1 = cs1 + y1
    if rej_u_1(z1, cs1, std1, rep_M1):
        return "Rejected"

    cs2 = c * s2
    z2 = cs2 + y2
    if rej_u_2(z2, cs2, std2, rep_M2):
        return "Rejected"
    
    #=============================== Verifier ================================
    z = stack_vec_Rql([z1, c*tB - B*z2])
    if not is_sigma_1:
        z = sigma_n1_struct_vec(z)
    f = c * t - b.dot_product(z2)

    cond1 = ((norm_Rql(z1) <= std1 * sg.sqrt(2 * m1 * d))
             and (norm_Rql(z2) <= std2 * sg.sqrt(2 * m2 * d)))
    cond2 = (A1 * z1 + A2 * z2 - c * tA == w)
    cond3 = (z.dot_product(R2 * z) + c * r1.dot_product(z) + c**2 * r0 - f == v)
    
    return cond1 and cond2 and cond3


def test_abdlop_single_quadratic():
    """Call abdlop_single_quadratic with example parameters"""

    # Size parameters
    m1 = 8     # size of s1
    m2 = 25    # size of s2
    ell = 2    # size of m
    n = 9      # dimension of the Module-SIS problem

    # Algorithm parameters
    is_sigma_1 = False  # automorphism: True for sigma_1, False for sigma_n1
    if is_sigma_1:
        k = k_sigma_1                       # degree of automorphism
        eta = eta_challenge_v1              # challenge norm upper bound
        get_challenge_u = get_challenge_v1  # challenge sampling algorithm
    else:
        k = k_sigma_n1
        eta = eta_challenge_v2
        get_challenge_u = get_challenge_v2

    rej_u_1 = rej1  # rejection sampling algorithm
    rej_u_2 = rej2

    gamma1 = 19  # repetition rate parameter
    nu_s1 = 1    # s1 max coefficient
    alpha_s1 = norm_Rql_bound(m1, nu_s1)  # s1 norm upper bound
    std1 = gamma1 * eta * alpha_s1  # standard deviation
    rep_M1 = rej1_M(gamma1)  # repetition rate

    gamma2 = 2
    nu_s2 = 1
    alpha_s2 = norm_Rql_bound(m2, nu_s2)
    std2 = gamma2 * eta * alpha_s2
    rep_M2 = rej2_M(gamma2)

    global rep_rate_abdlop_single_quadratic
    rep_rate_abdlop_single_quadratic = rep_M1 * 2 * rep_M2

    # Private information
    s1 = rand_Rql_small(m1, nu_s1)  # ABDLOP Ajtai part message vector
    s2 = rand_Rql_small(m2, nu_s2)  # ABDLOP randomness vector
    m = rand_Rql(ell)               # ABDLOP BDLOP part message vector
    s = stack_vec_Rql([s1, m])
    if not is_sigma_1:
        s = sigma_n1_struct_vec(s)

    # Public information
    A1 = rand_Rq_mat(n, m1)   # ABDLOP A1
    A2 = rand_Rq_mat(n, m2)   # ABDLOP A2
    B = rand_Rq_mat(ell, m2)  # ABDLOP B
    b = rand_Rql(m2)          # randomness vector

    # ABDLOP commitment
    tA_tB = A1.stack(ZM(ell, m1)) * s1 + \
        A2.stack(B) * s2 + stack_vec_Rql([zv(n), m])
    tA = tA_tB[:n]  # ABDLOP tA
    tB = tA_tB[n:]  # ABDLOP tB

    # Quadratic function
    R2 = rand_Rq_mat(k * (m1 + ell), k * (m1 + ell))
    r1 = rand_Rql(k * (m1 + ell))
    r0 = -(s.dot_product(R2 * s) + r1.dot_product(s))

    return abdlop_single_quadratic(m1, m2, ell, n,
                                   is_sigma_1, rej_u_1, rej_u_2, get_challenge_u,
                                   std1, rep_M1, std2, rep_M2,
                                   s1, s2, m, A1, A2, B, b, tA, tB,
                                   R2, r1, r0)



#print(measure_rep_rate(test_abdlop_single_quadratic, rep_rate_abdlop_single_quadratic))


#-------------------- Multiple Quadratic Relations over Rq --------------------

def abdlop_multiple_quadratic(m1, m2, ell, n, N,
                              is_sigma_1, rej_u_1, rej_u_2, get_challenge_u,
                              std1, rep_M1, std2, rep_M2,
                              s1, s2, m, A1, A2, B, b, tA, tB,
                              R2_is, r1_is, r0_is):
    """ABDLOP proof of multiple quadratic relations"""

    # =============================== Verifier ================================
    mu_is = rand_Rql(N)

    # ================================ Prover =================================
    # Linear-combine the N equations into one
    R2 = sum([mu_is[i] * R2_is[i] for i in range(N)])
    r1 = sum([mu_is[i] * r1_is[i] for i in range(N)])
    r0 = sum([mu_is[i] * r0_is[i] for i in range(N)])

    # Run adblop single quadratic with combined equation
    return abdlop_single_quadratic(m1, m2, ell, n,
                                is_sigma_1, rej_u_1, rej_u_2, get_challenge_u,
                                std1, rep_M1, std2, rep_M2,
                                s1, s2, m, A1, A2, B, b, tA, tB,
                                R2, r1, r0)


def test_abdlop_multiple_quadratic():
    """Call abdlop_multiple_quadratic with example parameters"""

    # Size parameters
    m1 = 8     # size of s1
    m2 = 8     # size of s2
    ell = 8    # size of m
    n = 4      # dimension of the Module-SIS problem
    N = 2      # number of quadratic equations

    # Algorithm parameters
    is_sigma_1 = False  # automorphism: True for sigma_1, False for sigma_n1
    if is_sigma_1:
        k = k_sigma_1                       # degree of automorphism
        eta = eta_challenge_v1              # challenge norm upper bound
        get_challenge_u = get_challenge_v1  # challenge sampling algorithm
    else:
        k = k_sigma_n1
        eta = eta_challenge_v2
        get_challenge_u = get_challenge_v2

    rej_u_1 = rej1  # rejection sampling algorithm
    rej_u_2 = rej2

    gamma1 = 19  # repetition rate parameter
    nu_s1 = 1    # s1 max coefficient
    alpha_s1 = norm_Rql_bound(m1, nu_s1)  # s1 norm upper bound
    std1 = gamma1 * eta * alpha_s1  # standard deviation
    rep_M1 = rej1_M(gamma1)  # repetition rate

    gamma2 = 2
    nu_s2 = 1
    alpha_s2 = norm_Rql_bound(m2, nu_s2)
    std2 = gamma2 * eta * alpha_s2
    rep_M2 = rej2_M(gamma2)

    global rep_rate_abdlop_multiple_quadratic
    rep_rate_abdlop_multiple_quadratic = rep_M1 * 2 * rep_M2

    # Private information
    s1 = rand_Rql_small(m1, nu_s1)  # ABDLOP Ajtai part message vector
    s2 = rand_Rql_small(m2, nu_s2)  # ABDLOP randomness vector
    m = rand_Rql(ell)               # ABDLOP BDLOP part message vector
    s = stack_vec_Rql([s1, m])
    if not is_sigma_1:
        s = sigma_n1_struct_vec(s)

    # Public information
    A1 = rand_Rq_mat(n, m1)   # ABDLOP A1
    A2 = rand_Rq_mat(n, m2)   # ABDLOP A2
    B = rand_Rq_mat(ell, m2)  # ABDLOP B
    b = rand_Rql(m2)          # randomness vector

    # ABDLOP commitment
    tA_tB = A1.stack(ZM(ell, m1)) * s1 + \
        A2.stack(B) * s2 + stack_vec_Rql([zv(n), m])
    tA = tA_tB[:n]  # ABDLOP tA
    tB = tA_tB[n:]  # ABDLOP tB

    # Quadratic functions
    R2_is = [rand_Rq_mat(k * (m1 + ell), k * (m1 + ell)) for i in range(N)]
    r1_is = [rand_Rql(k * (m1 + ell)) for i in range(N)]
    r0_is = [-(s.dot_product(R2_is[i] * s) + r1_is[i].dot_product(s)) for i in range(N)]

    return abdlop_multiple_quadratic(m1, m2, ell, n, N,
                                     is_sigma_1, rej_u_1, rej_u_2, get_challenge_u,
                                     std1, rep_M1, std2, rep_M2,
                                     s1, s2, m, A1, A2, B, b, tA, tB,
                                     R2_is, r1_is, r0_is)


#print(measure_rep_rate(test_abdlop_multiple_quadratic, rep_rate_abdlop_multiple_quadratic))


#------------------------- Quadratic Proofs over Zq --------------------------

def Tr(poly):
    """
    Create polynomials such that for any a,b in R_q,
    the polynomial y = Tr(a) + X^d/2 * Tr(b) has y_0 = 0 and y_d/2 = 0
    """
    return (poly + sigma_n1(poly)) / 2


def get_U(n):
    """Create permutation matrix that swaps every 2nd element of the vector it is multiplied with"""
    P = IM(n)
    for i in range(1, n, 2):
        P.swap_rows(i - 1, i)
    return P


def abdlop_quadratic_poly(m1, m2, ell, n, k, N, M, lambd,
                          is_sigma_1, is_opti, rej_u_1, rej_u_2, get_challenge_u,
                          std1, rep_M1, std2, rep_M2,
                          s1, s2, m, A1, A2, B, Bg, b, tA, tB,
                          R2_is, r1_is, r0_is,
                          R2_p_is, r1_p_is, r0_p_is):
    """
    ABDLOP proof of
    - multiple quadratic relations 
    - multiple quadratic polynomials have evaluations with constant coefficient zero
    """
    
    #================================ Prover =================================
    s = stack_vec_Rql([s1, m])
    if not is_sigma_1:
        s = sigma_n1_struct_vec(s)

    g = rand_Rql_first_zero(lambd)
    tg = Bg * s2 + g

    #=============================== Verifier ================================
    if not is_opti:
        Y = rand_Zq_mat(lambd, M)
    else:
        Y = rand_Zq_mat(2*lambd, M)

    #================================ Prover =================================
    if not is_opti:
        h = [g[i] + sum([Y[i][j] * (s.dot_product(R2_p_is[j] * s) + r1_p_is[j].dot_product(s) + r0_p_is[j])
                         for j in range(M)])
             for i in range(lambd)]
    else:
        h = [g[i] + sum([(Y[2*i][j] + X_q**(d//2) * Y[2*i + 1][j])
                          * Tr(s.dot_product(R2_p_is[j] * s) + r1_p_is[j].dot_product(s) + r0_p_is[j])
                         for j in range(M)])
             for i in range(lambd)]

    # Append Bg to B, g to m and tg to tB
    tB = stack_vec_Rql([tB, tg])
    B = B.stack(Bg)
    m = stack_vec_Rql([m, g])

    R2_is = [sg.block_matrix(R_q, [
                [R2_is[i], ZM(k*(m1 + ell), k*lambd)],
                [ZM(k*lambd, k*(m1 + ell)), ZM(k*lambd, k*lambd)]
            ]) for i in range(N)]

    r1_is = [stack_vec_Rql([r1_is[i], zv(k*lambd)]) for i in range(N)]
    
    if is_sigma_1:
        e = [sg.vector(R_q, [1 if j == i else 0 for j in range(lambd)]) for i in range(lambd)]
    else:
        e = [sg.vector(R_q, [1 if j == 2 * i else 0 for j in range(k*lambd)]) for i in range(lambd)]

    if not is_opti:
        R2_is.extend([sg.block_matrix(R_q, [
            [sum([Y[i][j] * R2_p_is[j] for j in range(M)]), ZM(k*(m1 + ell), k*lambd)],
            [ZM(k*lambd, k*(m1 + ell)), ZM(k*lambd, k*lambd)]
        ]) for i in range(lambd)])

        r1_is.extend([stack_vec_Rql([
            sum([Y[i][j] * r1_p_is[j] for j in range(M)]),
            e[i]
        ]) for i in range(lambd)])

        r0_is.extend([sum([Y[i][j] * r0_p_is[j] for j in range(M)]) - h[i] for i in range(lambd)])
        
        N = N + lambd
        ell = ell + lambd
    else:
        U = get_U(2*(m1 + ell))
        Ut = U.transpose()
        Y_ijs = [[(Y[2*i][j] + X_q**(d//2) * Y[2*i + 1][j]) / 2 for j in range(M)] for i in range(lambd)]
        R2pU_js = [R2_p_is[j] + Ut * sigma_n1_mat(R2_p_is[j]) * U for j in range(M)]
        r1p_js = [r1_p_is[j] + Ut * sigma_n1_vec(r1_p_is[j]) for j in range(M)]
        r0p_js = [r0_p_is[j] + sigma_n1(r0_p_is[j]) for j in range(M)]

        R2_is.extend([sg.block_matrix(R_q, [
            [sum([Y_ijs[i][j] * R2pU_js[j] for j in range(M)]), ZM(k*(m1 + ell), k*lambd)],
            [ZM(k*lambd, k*(m1 + ell)), ZM(k*lambd, k*lambd)]
        ]) for i in range(lambd)])

        r1_is.extend([stack_vec_Rql([
            sum([Y_ijs[i][j] * r1p_js[j] for j in range(M)]),
            e[i]
        ]) for i in range(lambd)])

        r0_is.extend([sum([Y_ijs[i][j] * r0p_js[j] for j in range(M)]) - h[i] for i in range(lambd)])

        N = N + lambd
        ell = ell + lambd

    # Run ABDLOP multiple quadratic sub-protocol
    cond1 = abdlop_multiple_quadratic(m1, m2, ell, n, N,
                                      is_sigma_1, rej_u_1, rej_u_2, get_challenge_u,
                                      std1, rep_M1, std2, rep_M2,
                                      s1, s2, m, A1, A2, B, b, tA, tB,
                                      R2_is, r1_is, r0_is)
    if cond1 == "Rejected":
        return "Rejected"

    #=============================== Verifier ================================
    cond2 = all(h[i][0] == 0 for i in range(lambd))  # check constant coefficient is 0
    
    return cond1 and cond2


def test_abdlop_quadratic_poly():
    """Call abdlop_quadratic_poly with example parameters"""

    # Size parameters
    m1 = 8     # size of s1
    m2 = 25    # size of s2
    ell = 2    # size of m
    n = 9      # dimension of the Module-SIS problem
    N = 1      # number of quadratic euqations
    M = 1      # number of quadratic polynomials with constant coefficient zero
    lambd = 4  # number of random masking polynomials (boosting soundness to q1^-lambda)

    # Algorithm parameters
    is_sigma_1 = False  # automorphism: True for sigma_1, False for sigma_n1
    is_opti = True      # proof size optimized sigma_n1 version with reduced garbage commitments
    assert(not(is_sigma_1 and is_opti))  # is_opti only works with sigma_n1
    if is_sigma_1:
        k = k_sigma_1                       # degree of automorphism
        eta = eta_challenge_v1              # challenge norm upper bound
        get_challenge_u = get_challenge_v1  # challenge sampling algorithm
    else:
        k = k_sigma_n1
        eta = eta_challenge_v2
        get_challenge_u = get_challenge_v2
        if is_opti:
            lambd = lambd // 2  # reduced garbage commitments

    rej_u_1 = rej1  # rejection sampling algorithm
    rej_u_2 = rej2

    gamma1 = 19  # repetition rate parameter
    nu_s1 = 1    # s1 max coefficient
    alpha_s1 = norm_Rql_bound(m1, nu_s1)  # s1 norm upper bound
    std1 = gamma1 * eta * alpha_s1  # standard deviation
    rep_M1 = rej1_M(gamma1)  # repetition rate

    gamma2 = 2
    nu_s2 = 1
    alpha_s2 = norm_Rql_bound(m2, nu_s2)
    std2 = gamma2 * eta * alpha_s2
    rep_M2 = rej2_M(gamma2)

    global rep_rate_abdlop_quadratic_poly
    rep_rate_abdlop_quadratic_poly = rep_M1 * 2 * rep_M2

    # Private information
    s1 = rand_Rql_small(m1, nu_s1)  # ABDLOP Ajtai part message vector
    s2 = rand_Rql_small(m2, nu_s2)  # ABDLOP randomness vector
    m = rand_Rql(ell)               # ABDLOP BDLOP part message vector
    s = stack_vec_Rql([s1, m])
    if not is_sigma_1:
        s = sigma_n1_struct_vec(s)

    # Public information
    A1 = rand_Rq_mat(n, m1)      # ABDLOP A1
    A2 = rand_Rq_mat(n, m2)      # ABDLOP A2
    B = rand_Rq_mat(ell, m2)     # ABDLOP B
    Bg = rand_Rq_mat(lambd, m2)  # randomness matrix
    b = rand_Rql(m2)             # randomness vector

    # ABDLOP commitment
    tA_tB = A1.stack(ZM(ell, m1)) * s1 + \
        A2.stack(B) * s2 + stack_vec_Rql([zv(n), m])
    tA = tA_tB[:n]  # ABDLOP tA
    tB = tA_tB[n:]  # ABDLOP tB

    # Quadratic functions
    R2_is = [rand_Rq_mat(k * (m1 + ell), k * (m1 + ell)) for i in range(N)]
    r1_is = [rand_Rql(k * (m1 + ell)) for i in range(N)]
    r0_is = [-(s.dot_product(R2_is[i] * s) + r1_is[i].dot_product(s)) for i in range(N)]

    # Quadratic polynomials with constant coefficient zero
    R2_p_is = [rand_Rq_mat_first_zero(
        k * (m1 + ell), k * (m1 + ell)) for i in range(M)]
    r1_p_is = [rand_Rql_first_zero(k * (m1 + ell)) for i in range(M)]
    r0_p_is = [-(s.dot_product(R2_p_is[i] * s) + r1_p_is[i].dot_product(s))
               for i in range(M)]

    return abdlop_quadratic_poly(m1, m2, ell, n, k, N, M, lambd,
                                 is_sigma_1, is_opti, rej_u_1, rej_u_2, get_challenge_u,
                                 std1, rep_M1, std2, rep_M2,
                                 s1, s2, m, A1, A2, B, Bg, b, tA, tB,
                                 R2_is, r1_is, r0_is,
                                 R2_p_is, r1_p_is, r0_p_is)


#print(measure_rep_rate(test_abdlop_quadratic_poly, rep_rate_abdlop_quadratic_poly))


#----------------- Proving Knowledge of a Module-LWE Secret ------------------

def get_J(n, k):
    """Get x from <x>_(sigma_(-1)) structure of sigma_n1_struct_vec(x)"""
    return IM(n).tensor_product(
        sg.matrix(R_q, [1] + [0 for _ in range(k - 1)]))


def get_pow(B_square):
    """Get polynomial representing powers of 2 up to B_square"""
    return R_q(
        sum([2**i * X_q**i for i in range(int(sg.log(B_square, 2)) + 1)]))


def N_2_binary_Rq(x):
    """Represent positive integer as binary polynomial compatible with get_pow()"""
    bin_rep = [int(i) for i in bin(x)[2:]]
    bin_rep.reverse()
    return R_q(bin_rep)


def abdlop_mlwe(m1, m2, n, k, Z, n_is, lambd,
                is_opti, rej_u_1, rej_u_2, rej_u_3, get_challenge_u,
                std1, rep_M1, std2, rep_M2, std3, rep_M3, roh,
                s1, s2, A1, A2, B_gamma, B_beta, B_ext, b_ext, theta, tA,
                E_s_is, v_is, B_is):
    """
    Prove knowledge of a Module-LWE Secret
    """
    is_sigma_1 = False  # always uses sigma_n1 automorphism

    #================================ Prover =================================
    n_ex = sum(n_is) + Z
    s3 = stack_vec_Rql([E_s_is[i]*s1 + v_is[i] for i in range(Z)] + [theta])

    y3_smplr = GaussianSampler(R_q, std3, 256//d)
    y3 = y3_smplr.get()
    beta3 = random.choice([-1, 1])

    t_gamma = B_gamma*s2 + y3
    t_beta = B_beta*s2 + sg.vector([beta3])

    #=============================== Verifier ================================
    R_ = rand_ZZ_mat_binomial(256, n_ex * d, 2)

    #================================ Prover =================================
    Rs3 = ZZl_2_Rql(R_ * Rql_2_ZZl(s3))
    z3 = y3 + beta3 * Rs3
    if rej_u_3(z3, Rs3, std3, rep_M3):
        return "Rejected"
    
    # Convenience matrices and vectors
    U = get_U(2*(m1 + Z + 256//d + 1))
    J = get_J(m1 + Z + 256//d + 1, 2)

    e_is = [sg.vector(R_q, [R_q([1 if k + d*j == i else 0 for k in range(d)]) for j in range(256//d)])
            for i in range(256)]

    K_s = sg.block_matrix(R_q, [
        [IM(m1), ZM(m1, Z), ZM(m1, 256//d + 1)]
    ]) * J

    k_theta_is = [stack_vec_Rql([zv(m1 + i), [1], zv(Z - i - 1 + 256//d + 1)]) * J for i in range(Z)]

    K_y3 = sg.block_matrix(R_q, [
        [ZM(256//d, m1 + Z), IM(256//d), ZM(256//d, 1)]
    ]) * J

    k_beta3 = (ZM(1, m1 + Z + 256//d).augment(sg.matrix(R_q, [1])) * J)[0]
    
    # Quadratic relations over Rq
    R2_is = []
    r1_is = []
    r0_is = []
    N = 0

    # Quadratic relations over Zq
    R2_p_is = []
    r1_p_is = []
    r0_p_is = []
    M = 0

    # Extend R2
    # 6.25, 6.33
    R2_is.extend([k_beta3.column() * k_beta3.row()])
    r1_is.extend([zv(2 * (m1 + Z + 256//d + 1))])
    r0_is.extend([R_q(-1)])

    N = N + 1

    # Extend R2_p
    # Exact Shortness Extensions
    # 6.24 (prove approximate shortness)
    block_PsEsi = sg.block_matrix(R_q, [
        [sg.block_matrix(R_q, [[E_s_is[j], ZM(n_is[j], Z)] for j in range(Z)])],
        [sg.block_matrix(R_q, [[ZM(Z, m1), IM(Z)]])]
    ])
    block_Kskthetai = sg.block_matrix(R_q, [
        [K_s], 
        [sg.block_matrix(R_q, [[k_theta_is[j].row()] for j in range(Z)])]
    ])
    block_PsEsi_Kskthetai = block_PsEsi * block_Kskthetai

    R2_p_is.extend([
        k_beta3.column()
        * sigma_n1_vec(ZZl_2_Rql(R_[i])).row()
        * block_PsEsi_Kskthetai
    for i in range(256)])

    block_fvi = stack_vec_Rql([v_is[j] for j in range(Z)] + [sg.zero_vector(Z)])

    r1_p_is.extend([
        sigma_n1_vec(e_is[i]) * K_y3
        + (sigma_n1_vec(ZZl_2_Rql(R_[i])) * block_fvi * k_beta3)
    for i in range(256)])

    r0_p_is.extend([R_q(-dot_Rql(z3, e_is[i])) for i in range(256)])

    M = M + 256

    # 6.26 (prove beta3 in {-1, 1})
    R2_p_is.extend([ZM(2*(m1 + Z + 256//d + 1), 2*(m1 + Z + 256//d + 1)) for i in range(1, d)])
    r1_p_is.extend([X_q**i * k_beta3 for i in range(1, d)])
    r0_p_is.extend([R_q(0) for i in range(1, d)])

    M = M + (d - 1)

    # Euclidean Norm Extensions
    # 6.29 (exact shortness over Zq)
    R2_p_is.extend([
        U.transpose() * sigma_n1_mat(K_s).transpose()
        * sigma_n1_mat(E_s_is[i]).transpose() * E_s_is[i]
        * K_s
    for i in range(Z)])

    r1_p_is.extend([
        v_is[i] * sigma_n1_mat(E_s_is[i]) * sigma_n1_mat(K_s) * U
        + sigma_n1_vec(v_is[i]) * E_s_is[i] * K_s
        + sigma_n1(get_pow(B_is[i]**2)) * k_theta_is[i]
    for i in range(Z)])

    r0_p_is.extend([R_q(dot_Rql(v_is[i], v_is[i]) - B_is[i]**2) for i in range(Z)])

    M = M + Z

    # Append (B_gamma, B_beta) to B, theta to s1, (y3, beta3) to m and (t_gamma, t_beta) to tB
    B = B_gamma.stack(B_beta)
    Bg = B_ext
    b = b_ext

    s1 = stack_vec_Rql([s1, theta])
    m1 = m1 + Z
    m = stack_vec_Rql([y3, [beta3]])
    ell = 256//d + 1

    tB = stack_vec_Rql([t_gamma, t_beta])

    # Run quadratic poly sub-protocol
    cond1 = abdlop_quadratic_poly(m1, m2, ell, n, k, N, M, lambd,
                                  is_sigma_1, is_opti, rej_u_1, rej_u_2, get_challenge_u,
                                  std1, rep_M1, std2, rep_M2,
                                  s1, s2, m, A1, A2, B, Bg, b, tA, tB,
                                  R2_is, r1_is, r0_is,
                                  R2_p_is, r1_p_is, r0_p_is)
    if cond1 == "Rejected":
        return "Rejected"

    #=============================== Verifier ================================
    cond2 = (float(norm_Rql(z3)) <= float(roh * std3 * sg.sqrt(256)))

    return cond1 and cond2


def test_abdlop_mlwe():
    """Call abdlop_mlwe with example parameters"""

    # Size parameters
    m1 = 8     # size of s1
    m2 = 25    # size of s2
    n = 9      # dimension of the Module-SIS problem
    Z = 1      # number of equations for Euclidean norm
    # number of random masking polynomials (boosting soundness to q1^-lambda)
    lambd = 4
    n_A = 8    # height of A matrix
    m_A = m1   # width of A matrix (must be same size as m1)

    # Algorithm parameters
    is_opti = False  # proof size optimized sigma_n1 version with reduced garbage commitments
    if is_opti:
        lambd = lambd // 2  # reduced garbage commitments

    k = k_sigma_n1                      # degree of automorphism
    eta = eta_challenge_v2              # challenge norm upper bound
    get_challenge_u = get_challenge_v2  # challenge sampling algorithm

    rej_u_1 = rej1  # rejection sampling algorithm
    rej_u_2 = rej2
    rej_u_3 = rej_bimodal

    gamma1 = 19  # repetition rate parameter
    nu_s1 = 1    # s1 max coefficient
    alpha_s1 = norm_Rql_bound(m1, nu_s1)  # s1 norm upper bound
    std1 = gamma1 * eta * sg.sqrt(alpha_s1**2 + d)  # standard deviation
    rep_M1 = rej1_M(gamma1)  # repetition rate

    gamma2 = 1
    nu_s2 = 1
    alpha_s2 = norm_Rql_bound(m2, nu_s2)
    std2 = gamma2 * eta * alpha_s2
    rep_M2 = rej2_M(gamma2)

    # Private information
    s1 = rand_Rql_small(m1, nu_s1)  # ABDLOP Ajtai part message vector
    s2 = rand_Rql_small(m2, nu_s2)  # ABDLOP randomness vector

    # Public information
    # ABDLOP commitment
    A1 = rand_Rq_mat(n, m1 + Z)
    A2 = rand_Rq_mat(n, m2)
    B_gamma = rand_Rq_mat(256 // d, m2)
    B_beta = rand_Rq_mat(1, m2)
    B_ext = rand_Rq_mat(lambd, m2)
    b_ext = rand_Rql(m2)

    # MLWE
    A = rand_Rq_mat(n_A, m_A)
    e = rand_Rql_bin(m_A)
    u = A * s1 + e

    # Shortness in the Euclidean norm
    n_is = [m_A + m_A]  # size of vectors for Euclidean norm equations
    E_s_is = [IM(m_A).stack(A)]
    v_is = [-stack_vec_Rql([zv(m_A), u])]
    norms = [norm_Rql(E_s_is[0] * s1 + v_is[0])]
    # integer Euclidean norm bound (sqrt(2048) is upper bound of norm for
    # nu_s1 = 1)
    B_is = [sg.sqrt(2048)]

    gamma3 = 6
    std3 = gamma3 * sg.sqrt(337) * sg.sqrt(d + B_is[0]**2)
    rep_M3 = rej_bimodal_M(gamma3)

    global rep_rate_abdlop_mlwe
    rep_rate_abdlop_mlwe = rep_M1 * 2 * rep_M2 * rep_M3

    roh = 1.64  # z3 norm bound (for kappa=128)

    # ABDLOP tA
    # squared norm is always an int (round float errors)
    squared_norm_diffs = [B_is[0]**2 - round(norms[0]**2)]
    theta = sg.vector(R_q, [N_2_binary_Rq(squared_norm_diffs[0])])
    tA = A1 * stack_vec_Rql([s1, theta]) + A2 * s2  # ABDLOP tA

    return abdlop_mlwe(m1, m2, n, k, Z, n_is, lambd,
                       is_opti, rej_u_1, rej_u_2, rej_u_3, get_challenge_u,
                       std1, rep_M1, std2, rep_M2, std3, rep_M3, roh,
                       s1, s2, A1, A2, B_gamma, B_beta, B_ext, b_ext, theta, tA,
                       E_s_is, v_is, B_is)


#print(measure_rep_rate(test_abdlop_mlwe, rep_rate_abdlop_mlwe))


#------------------- Toolbox for Proving Lattice Relations --------------------

def abdlop_toolbox(m1, m2, ell, n, k, N, M, n_bin, Z, n_is, n_p, lambd,
                   is_opti, rej_u_1, rej_u_2, rej_u_3, rej_u_4, get_challenge_u,
                   std1, rep_M1, std2, rep_M2, std3, rep_M3, std4, rep_M4, roh,
                   s1, s2, m, A1, A2, B, B_gamma, B_beta, B_ext, b_ext, theta, tA, tB,
                   R2_is, r1_is, r0_is,
                   R2_p_is, r1_p_is, r0_p_is,
                   P_s, P_m, f,
                   E_s_is, E_m_is, v_is, B_is,
                   D_s, D_m, u, B_p):
    """
    ABDLOP toolbox to prove various quadratic relations
    - Quadratic relations over Rq
    - Quadratic relations over Zq
    - Shortness in the infinity norm
    - Shortness in the Euclidean norm
    - Approximate shortness
    """
    is_sigma_1 = False  # always uses sigma_n1 automorphism

    #================================ Prover =================================
    n_ex = n_bin + sum(n_is) + Z
    s3 = stack_vec_Rql([P_s*s1 + P_m*m + f]
                       + [E_s_is[i]*s1 + E_m_is[i]*m + v_is[i] for i in range(Z)]
                       + [theta])
    s4 = D_s*s1 + D_m*m + u

    y3_smplr = GaussianSampler(R_q, std3, 256//d)
    y3 = y3_smplr.get()
    y4_smplr = GaussianSampler(R_q, std4, 256//d)
    y4 = y4_smplr.get()
    beta3 = random.choice([-1, 1])
    beta4 = random.choice([-1, 1])

    t_gamma = B_gamma*s2 + stack_vec_Rql([y3, y4])
    t_beta = B_beta*s2 + sg.vector([beta3, beta4])

    #=============================== Verifier ================================
    R_ = rand_ZZ_mat_binomial(256, n_ex * d, 2)
    R_p = rand_ZZ_mat_binomial(256, n_p * d, 2)

    #================================ Prover =================================
    Rs3 = ZZl_2_Rql(R_ * Rql_2_ZZl(s3))
    z3 = y3 + beta3 * Rs3
    if rej_u_3(z3, Rs3, std3, rep_M3):
        return "Rejected"

    Rps4 = ZZl_2_Rql(R_p * Rql_2_ZZl(s4))
    z4 = y4 + beta4 * Rps4
    if rej_u_4(z4, Rps4, std4, rep_M4):
        return "Rejected"
    
    # Convenience matrices and vectors
    U = get_U(2*(m1 + Z + ell + 512//d + 2))
    J = get_J(m1 + Z + ell + 512//d + 2, 2)

    e_is = [sg.vector(R_q, [R_q([1 if k + d*j == i else 0 for k in range(d)]) for j in range(256//d)])
            for i in range(256)]

    K_s = sg.block_matrix(R_q, [
        [IM(m1), ZM(m1, Z), ZM(m1, ell), ZM(m1, 512//d + 2)],
        [ZM(ell, m1), ZM(ell, Z), IM(ell), ZM(ell, 512//d + 2)]
    ]) * J

    K_s_sigma = sg.block_matrix(R_q, [
        [IM(2*m1), ZM(2*m1, 2*Z), ZM(2*m1, 2*ell),
         ZM(2*m1, 2*(512//d + 2))],
        [ZM(2*ell, 2*m1), ZM(2*ell, 2*Z), IM(2*ell),
         ZM(2*ell, 2*(512//d + 2))]
    ])

    k_theta_is = [stack_vec_Rql([zv(m1 + i), [1], zv(Z - i - 1 + ell + 512//d + 2)]) * J
                  for i in range(Z)]

    K_y3_K_y4 = sg.block_matrix(R_q, [
        [ZM(256//d, m1 + Z + ell), IM(256//d), ZM(256//d, 256//d),
         ZM(256//d, 2)],
        [ZM(256//d, m1 + Z + ell), ZM(256//d, 256//d), IM(256//d),
         ZM(256//d, 2)]
    ]) * J
    K_y3 = K_y3_K_y4[:256//d]
    K_y4 = K_y3_K_y4[256//d:]

    k_beta3_k_beta4 = sg.block_matrix(R_q, [
        [ZM(1, m1 + Z + ell + 512//d), sg.matrix(R_q, [1, 0])],
        [ZM(1, m1 + Z + ell + 512//d), sg.matrix(R_q, [0, 1])]
    ]) * J
    k_beta3 = k_beta3_k_beta4[0]
    k_beta4 = k_beta3_k_beta4[1]

    # Extend R2    
    # 6.20
    R2_is = [K_s_sigma.transpose() * R2_is[i] * K_s_sigma for i in range(N)]
    r1_is = [r1_is[i] * K_s_sigma for i in range(N)]    
    
    # 6.25, 6.33
    R2_is.extend([k_beta3.column() * k_beta3.row(), k_beta4.column() * k_beta4.row()])
    r1_is.extend([zv(2 * (m1 + Z + ell + 512//d + 2)), zv(2 * (m1 + Z + ell + 512//d + 2))])
    r0_is.extend([R_q(-1), R_q(-1)])

    N = N + 2

    # Extend R2_p
    # 6.21
    R2_p_is = [K_s_sigma.transpose() * R2_p_is[i] * K_s_sigma for i in range(M)]
    r1_p_is = [r1_p_is[i] * K_s_sigma for i in range(M)]   

    # Exact Shortness Extensions
    # 6.24
    block_PsEsi = sg.block_matrix(R_q, [
        [sg.block_matrix(R_q, [[P_s, P_m, ZM(n_bin, Z)]])], 
        [sg.block_matrix(R_q, [[E_s_is[j], E_m_is[j], ZM(n_is[j], Z)] for j in range(Z)])],
        [sg.block_matrix(R_q, [[ZM(Z, m1), ZM(Z, ell), IM(Z)]])]
    ])
    block_Kskthetai = sg.block_matrix(R_q, [
        [K_s], 
        [sg.block_matrix(R_q, [[k_theta_is[j].row()] for j in range(Z)])]
    ])
    block_PsEsi_Kskthetai = block_PsEsi * block_Kskthetai

    R2_p_is.extend([
        k_beta3.column()
        * sigma_n1_vec(ZZl_2_Rql(R_[i])).row()
        * block_PsEsi_Kskthetai
    for i in range(256)])

    block_fvi = stack_vec_Rql([f] + [v_is[j] for j in range(Z)] + [sg.zero_vector(Z)])

    r1_p_is.extend([
        sigma_n1_vec(e_is[i]) * K_y3
        + (sigma_n1_vec(ZZl_2_Rql(R_[i])) * block_fvi * k_beta3)
    for i in range(256)])

    r0_p_is.extend([R_q(-dot_Rql(z3, e_is[i])) for i in range(256)])

    M = M + 256

    # 6.26
    R2_p_is.extend([ZM(2*(m1 + Z + ell + 512//d + 2), 2*(m1 + Z + ell + 512//d + 2))
                    for i in range(1, d)])
    r1_p_is.extend([X_q**i * k_beta3 for i in range(1, d)])
    r0_p_is.extend([R_q(0) for i in range(1, d)])

    M = M + (d - 1)

    # Infinity Norm Extensions
    x = sg.vector(R_q, [R_q([1 for _ in range(d)]) for i in range(n_bin)])

    # 6.27
    R2_p_is.extend([
        U.transpose() * sigma_n1_mat(K_s).transpose() 
        * sg.block_matrix(R_q, [
            [sigma_n1_mat(P_s).transpose() * P_s, sigma_n1_mat(P_s).transpose() * P_m],
            [sigma_n1_mat(P_m).transpose() * P_s, sigma_n1_mat(P_m).transpose() * P_m]
        ])
        * K_s
    ])

    r1_p_is.extend([
        (f - x) * sg.block_matrix([[sigma_n1_mat(P_s), sigma_n1_mat(P_m)]]) * sigma_n1_mat(K_s) * U
        + sigma_n1_vec(f) * sg.block_matrix([[P_s, P_m]]) * K_s
    ])

    r0_p_is.extend([R_q(dot_Rql(f, f - x))])

    M = M + 1

    # 6.28
    R2_p_is.extend([U.transpose() * sigma_n1_vec(k_theta_is[i]).column() * k_theta_is[i].row() for i in range(Z)])
    r1_p_is.extend([R_q([-1 for _ in range(d)]) * sigma_n1_vec(k_theta_is[i]) * U for i in range(Z)])
    r0_p_is.extend([R_q(0) for i in range(Z)])

    M = M + Z

    # Euclidean Norm Extensions
    # 6.29
    R2_p_is.extend([
        U.transpose() * sigma_n1_mat(K_s).transpose()
        * sg.block_matrix(R_q, [
            [sigma_n1_mat(E_s_is[i]).transpose() * E_s_is[i], sigma_n1_mat(E_s_is[i]).transpose() * E_m_is[i]],
            [sigma_n1_mat(E_m_is[i]).transpose() * E_s_is[i], sigma_n1_mat(E_m_is[i]).transpose() * E_m_is[i]]
        ])
        * K_s
    for i in range(Z)])

    r1_p_is.extend([
        v_is[i] * sg.block_matrix(R_q, [[sigma_n1_mat(E_s_is[i]), sigma_n1_mat(E_m_is[i])]]) * sigma_n1_mat(K_s) * U
        + sigma_n1_vec(v_is[i]) * sg.block_matrix(R_q, [[E_s_is[i], E_m_is[i]]]) * K_s
        + sigma_n1(get_pow(B_is[i]**2)) * k_theta_is[i]
    for i in range(Z)])

    r0_p_is.extend([R_q(dot_Rql(v_is[i], v_is[i]) - B_is[i]**2) for i in range(Z)])

    M = M + Z

    # Approximate Shortness
    # 6.32
    R2_p_is.extend([k_beta4.column() * sigma_n1_vec(ZZl_2_Rql(R_p[i])).row() * sg.block_matrix(R_q, [[D_s, D_m]]) * K_s
                    for i in range(256)])
    r1_p_is.extend([sigma_n1_vec(e_is[i]) * K_y4 + sigma_n1_vec(ZZl_2_Rql(R_p[i])) * u * k_beta4 for i in range(256)])
    r0_p_is.extend([R_q(-dot_Rql(z4, e_is[i])) for i in range(256)])

    M = M + 256

    # 6.34
    R2_p_is.extend([ZM(2*(m1 + Z + ell + 512//d + 2), 2*(m1 + Z + ell + 512//d + 2))
                    for i in range(1, d)])
    r1_p_is.extend([X_q**i * k_beta4 for i in range(1, d)])
    r0_p_is.extend([R_q(0) for i in range(1, d)])

    M = M + (d - 1)

    # Append (B_gamma, B_beta) to B, theta to s1, (y3, y4, beta3, beta4) to m and (t_gamma, t_beta) to tB
    B = B.stack(B_gamma).stack(B_beta)
    Bg = B_ext
    b = b_ext

    s1 = stack_vec_Rql([s1, theta])
    m1 = m1 + Z
    m = stack_vec_Rql([m, y3, y4, [beta3], [beta4]])
    ell = ell + 256//d + 256//d + 1 + 1

    tB = stack_vec_Rql([tB, t_gamma, t_beta])

    # Run quadratic poly sub-protocol
    cond1 = abdlop_quadratic_poly(m1, m2, ell, n, k, N, M, lambd,
                                  is_sigma_1, is_opti, rej_u_1, rej_u_2, get_challenge_u,
                                  std1, rep_M1, std2, rep_M2,
                                  s1, s2, m, A1, A2, B, Bg, b, tA, tB,
                                  R2_is, r1_is, r0_is,
                                  R2_p_is, r1_p_is, r0_p_is)
    if cond1 == "Rejected":
        return "Rejected"

    #=============================== Verifier ================================
    cond2 = (float(norm_Rql(z3)) <= float(roh * std3 * sg.sqrt(256)))
    cond3 = (float(inf_norm_Rql(z4)) <= float(sg.sqrt(2 * kappa) * std4))

    return cond1 and cond2 and cond3


def test_abdlop_toolbox():
    """Call abdlop_toolbox with example parameters"""

    # Size parameters
    m1 = 8     # size of s1
    m2 = 25    # size of s2
    ell = 2    # size of m
    n = 9      # dimension of the Module-SIS problem
    N = 2      # number of quadratic equations
    M = 3      # number of quadratic polynomials with constant coefficient zero
    n_bin = 3  # size of infinity norm vector
    Z = 2      # number of equations for Euclidean norm
    n_is = random.choices(range(1, 4), k=Z)  # size of vectors for Euclidean norm equations
    n_p = 3    # size of approximate shortness vector
    lambd = 4  # number of random masking polynomials (boosting soundness to q1^-lambda)
   
    # Algorithm parameters
    is_opti = False  # proof size optimized sigma_n1 version with reduced garbage commitments
    if is_opti:
        lambd = lambd//2  # reduced garbage commitments

    k = k_sigma_n1                      # degree of automorphism
    eta = eta_challenge_v2              # challenge norm upper bound
    get_challenge_u = get_challenge_v2  # challenge sampling algorithm

    rej_u_1 = rej1  # rejection sampling algorithm
    rej_u_2 = rej2
    rej_u_3 = rej_bimodal
    rej_u_4 = rej_bimodal

    gamma1 = 19  # repetition rate parameter
    nu_s1 = 1    # s1 max coefficient
    alpha_s1 = norm_Rql_bound(m1, nu_s1)  # s1 norm upper bound
    std1 = gamma1 * eta * sg.sqrt(alpha_s1**2 + d)  # standard deviation
    rep_M1 = rej1_M(gamma1)  # repetition rate

    gamma2 = 2
    nu_s2 = 1
    alpha_s2 = norm_Rql_bound(m2, nu_s2)
    std2 = gamma2 * eta * alpha_s2
    rep_M2 = rej2_M(gamma2)

    # Private information
    s1 = rand_Rql_small(m1, nu_s1)  # ABDLOP Ajtai part message vector
    s2 = rand_Rql_small(m2, nu_s2)  # ABDLOP randomness vector
    m = rand_Rql(ell)               # ABDLOP BDLOP part message vector
    s = sigma_n1_struct_vec(stack_vec_Rql([s1, m]))

    # Public information
    # ABDLOP commitment
    A1 = rand_Rq_mat(n, m1 + Z)
    A2 = rand_Rq_mat(n, m2)
    B = rand_Rq_mat(ell, m2)
    B_gamma = rand_Rq_mat(512//d, m2)
    B_beta = rand_Rq_mat(2, m2)
    B_ext = rand_Rq_mat(lambd, m2)
    b_ext = rand_Rql(m2)

    # Quadratic relations over Rq
    R2_is = [rand_Rq_mat(k * (m1 + ell), k * (m1 + ell)) for i in range(N)]
    r1_is = [rand_Rql(k * (m1 + ell)) for i in range(N)]
    r0_is = [-(s.dot_product(R2_is[i] * s) + r1_is[i].dot_product(s)) for i in range(N)]

    # Quadratic relations over Zq
    R2_p_is = [rand_Rq_mat_first_zero(k * (m1 + ell), k * (m1 + ell)) for i in range(M)]
    r1_p_is = [rand_Rql_first_zero(k * (m1 + ell)) for i in range(M)]
    r0_p_is = [-(s.dot_product(R2_p_is[i] * s) + r1_p_is[i].dot_product(s)) for i in range(M)]

    # Shortness in the infinity norm
    P_s = rand_Rq_mat(n_bin, m1)
    P_m = rand_Rq_mat(n_bin, ell)
    f = -(P_s*s1 + P_m*m) + rand_Rql_bin(n_bin)  # P_s*s1 + P_m*m + f is polynomial vector with binary coefficients

    # Shortness in the Euclidean norm
    E_s_is = [rand_Rq_mat(n_is[i], m1) for i in range(Z)]
    E_m_is = [rand_Rq_mat(n_is[i], ell) for i in range(Z)]
    v_is = [-(E_s_is[i]*s1 + E_m_is[i]*m) + rand_Rql_bin(n_is[i]) for i in range(Z)]
    norms = [norm_Rql(E_s_is[i]*s1 + E_m_is[i]*m + v_is[i]) for i in range(Z)]
    B_is = [int(norms[i] + 2) for i in range(Z)]  # integer Euclidean norm bounds, all B_is should be < sqrt(q)

    gamma3 = 6
    std3 = gamma3 * sg.sqrt(337) * sg.sqrt(d + B_is[0]**2)
    rep_M3 = rej_bimodal_M(gamma3)

    roh = 1.64  # z3 norm bound (for kappa=128)

    # Approximate Shortness
    D_s = rand_Rq_mat(n_p, m1)
    D_m = rand_Rq_mat(n_p, ell)
    u = rand_Rql(n_p)
    B_p = norm_Rql(D_s*s1 + D_m*m + u)

    gamma4 = 6
    std4 = gamma4 * sg.sqrt(337) * B_p
    rep_M4 = rej_bimodal_M(gamma4)

    global rep_rate_abdlop_toolbox
    rep_rate_abdlop_toolbox = rep_M1 * 2*rep_M2 * rep_M3 * rep_M4

    # ABDLOP tA, tB
    # squared norm is always an int (round float errors)
    squared_norm_diffs = [B_is[i]**2 - round(norms[i]**2) for i in range(Z)]
    theta = sg.vector(R_q, [N_2_binary_Rq(squared_norm_diffs[i]) for i in range(Z)])
    tA_tB = (A1.stack(ZM(ell, m1 + Z)) * stack_vec_Rql([s1, theta])
             + A2.stack(B) * s2
             + stack_vec_Rql([zv(n), m]))
    tA = tA_tB[:n]  # ABDLOP tA
    tB = tA_tB[n:]  # ABDLOP tB

    return abdlop_toolbox(m1, m2, ell, n, k, N, M, n_bin, Z, n_is, n_p, lambd,
                          is_opti, rej_u_1, rej_u_2, rej_u_3, rej_u_4, get_challenge_u,
                          std1, rep_M1, std2, rep_M2, std3, rep_M3, std4, rep_M4, roh,
                          s1, s2, m, A1, A2, B, B_gamma, B_beta, B_ext, b_ext, theta, tA, tB,
                          R2_is, r1_is, r0_is,
                          R2_p_is, r1_p_is, r0_p_is,
                          P_s, P_m, f,
                          E_s_is, E_m_is, v_is, B_is,
                          D_s, D_m, u, B_p)


#print("Run Toolbox Proof...")
#print(repeat_until_accept(test_abdlop_toolbox))

#print(measure_rep_rate(test_abdlop_toolbox, rep_rate_abdlop_toolbox, runs=10))


#------------------------------ Benchmark Code -------------------------------

def benchmark_protocol(func, runs):
    """Measure runtime of a protocol, only counting successful runs"""

    i = 0
    total_time = 0
    while i < runs:
        start_time = time.time()
        res = func()
        end_time = time.time()
        if res != "Rejected":
            i += 1
            total_time += end_time - start_time

    return total_time / runs


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", help="run benchmarks", action="store_true")
    args = parser.parse_args()

    if args.benchmark:
        print("Running Benchmarks")
        print("this will take a few minutes")

        time_commit = benchmark_protocol(test_abdlop_commit, 10)
        print(f"Prove ABDLOP opening and 1 lin. rel. over Rq: {time_commit:.3f}s")

        time_linear = benchmark_protocol(test_abdlop_linear, 10)
        print(f"Prove ABDLOP opening, 1 lin. rel. over Rq and 1 lin. rel. over Zq: {time_linear:.3f}s")

        time_quadratic = benchmark_protocol(test_abdlop_single_quadratic, 10)
        print(f"Prove ABDLOP opening and 1 quad. rel. over Rq: {time_quadratic:.3f}s")

        time_quadratic_poly = benchmark_protocol(test_abdlop_quadratic_poly, 10)
        print(f"Prove ABDLOP opening, 1 quad. rel. over Rq and 1 quad. rel. over Zq: {time_quadratic_poly:.3f}s")

        time_mlwe = benchmark_protocol(test_abdlop_mlwe, 3)
        print(f"Prove Module-LWE secret: {time_mlwe:.3f}s")

    else:
        print("Testing Protocols")
        print(f"Using ring R_q: {R_q}")

        print("Test Linear Proofs over Zq...")
        print(repeat_until_accept(test_abdlop_linear))

        print("Test ABDLOP Commitment Opening Proof with Linear Proofs over Rq...")
        print(repeat_until_accept(test_abdlop_commit))

        print("Test Single Quadratic Relation over Rq...")
        print(repeat_until_accept(test_abdlop_single_quadratic))

        print("Test Multiple Quadratic Relations over Rq...")
        print(repeat_until_accept(test_abdlop_multiple_quadratic))

        print("Test Quadratic Proofs over Zq...")
        print(repeat_until_accept(test_abdlop_quadratic_poly))

        print("Test Module-LWE Secret Proof...")
        print(repeat_until_accept(test_abdlop_mlwe))
