from src.qdmt.evolve import *
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from examples.analyze_first import load_data, count_initial_nonzero, spit_out_state
from scipy.stats import unitary_group
from scipy.linalg import expm
import matplotlib.pyplot as plt



def random_unitary_near_identity(D, eps=0.01):
    A = np.random.randn(D, D) + 1j*np.random.randn(D, D)
    H = (A + A.conj().T) / 2

    # Normalize H so Frobenius norm = 1
    H = H / np.linalg.norm(H, 'fro')

    U = expm(1j * eps * H)
    return U





def check_gradient(A0, A_target, L, iterations, tolerance, cut_off=10*60*60):

    
    dt=1
    model = TransverseFieldIsing(g=0, delta_t=dt, h=0, J=0)
    # model = TransverseFieldIsing(g=1.05, delta_t=dtt, h=-0.5, J=-1)


    
    # times, state, cost, norm, duration = evolve(A_init, D, L, model, dt, time_total, iterations, tolerance, cut_off)

    trotterization_order=2
    M=Grassmann()
    max_iter=iterations
    tol=tolerance

    f = EvolvedHilbertSchmidt(A_target, model, L, trotterization_order)
    gd = ConjugateGradient(f, M, A0, max_iter, tol=tol, verbose=True)
    A, c, n, _ = gd.optimize()
    print(A.is_isometry(1e-14))
    print("done")
    print(c)
    return c



# d=2
# D=4

# A0=UniformMps.random(D, d)

# U = unitary_group.rvs(D)

# # print(U)
# # print(A0.V.shape)
# # print((A0.V@U).shape)

# W=UniformMps(A0.V@ U) 


# # print(A0.is_isometry(1e-14))
# # print(W.is_isometry(1e-14))



# # print(A0.is_isometry(1e-14))


# A_target=W
# L=4

# it=2000
# tol=1e-10

# Example
# D = 4
# U = random_unitary_near_identity(D, eps=0.01)
# print("Distance:", np.linalg.norm(U - np.eye(D)))


def probe(d,D,L,it,tol,sample,eps):
    pos=0
    for i in range(0,sample):
        print(f"This is iteration {i}.")
        A0=UniformMps.random(D, d)
        U = unitary_group.rvs(D)
        U = random_unitary_near_identity(D, eps)
        if  eps==0:
            U = unitary_group.rvs(D)
        W=UniformMps(A0.V@ U) 
        
        c = check_gradient(A0, W, L, it, tol)
        print(c)
        if c<1e-12:
            pos+=1
        print(f"successful in {pos} of {i+1} cases")        
    return pos/sample    



def plot_success_vs_eps(results):
    """
    results: structured numpy array with fields
        'D', 'L', 'dist', 'success_rate'
    """
    
    # Find all unique (D, L) pairs
    pairs = np.unique([(r['D'], r['L']) for r in results])

    plt.figure(figsize=(8, 6))

    for (D, L) in pairs:
        # filter rows for this (D,L)
        mask = (results['D'] == D) & (results['L'] == L)
        subset = results[mask]

        # sort by epsilon so the line is not scrambled
        order = np.argsort(subset['dist'])
        eps = subset['dist'][order]
        sr = subset['success_rate'][order]

        # plot
        plt.plot(eps, sr, marker='o', label=f"D={D}, L={L}")

    plt.xlabel("epsilon (dist)")
    plt.ylabel("success rate")
    plt.title("Success rate vs epsilon for all (D, L) pairs")
    plt.legend()
    plt.grid(True)
    plt.show()

samples=10
iter=500
tol=1e-10

results = []





# for D in [2,4]:
#     for L in [2]:
#         for eps in [0.01,0.1,0.5,1]:
#             print([D,L,eps])
#             success_rate=probe(2,D,L,iter,tol,samples,eps)
#             results.append((D, L, eps, success_rate))

# # convert to numpy structured array
# results = np.array(results, 
#         dtype=[('D', int), ('L', int), ('dist', float), ('success_rate', float)])            

#  worked for this as well
# print(probe(2,4,4,500,1e-10,10,0))            
# print(results)


# L = 2
# D_search = 4
# D_target = 16
# A0 =UniformMps.random(D_search, d)
# Atarget=UniformMps.random(D_target, d)
# c = check_gradient(A0, Atarget, L, it, tol)


def probe_general(d,D_search,D_target,L,it,tol,sample):
    pos=0
    for i in range(0,sample):
        print(f"This is iteration {i}.")
        A0 =UniformMps.random(D_search, d)
        Atarget=UniformMps.random(D_target, d)
        c = check_gradient(A0, Atarget, L, it, tol)
        if c<1e-14:
            pos+=1
    print(f"For D'={D_search}, representing rho(D={D_target},L={L}) was successful in {pos} of {i+1} cases")        
    return pos/sample    


it = 500
tol = 1e-10
d = 2
sample=10

L=4
D_search=16
D_target=20

probe_general(2,D_search,D_target,L,it,tol,sample)