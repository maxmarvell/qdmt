from qdmt.uniform_mps import UniformMps

def compute_correlation_length(A: UniformMps):
    return A.correlation_length()