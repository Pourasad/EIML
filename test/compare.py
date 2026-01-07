import numpy as np

soap_path = "test/soap_features.npy"
eiml_path = "test/eiml_features.npy"

X_soap = np.load(soap_path)
X_eiml = np.load(eiml_path)

print("SOAP shape:", X_soap.shape, "dtype:", X_soap.dtype)
print("EIML shape:", X_eiml.shape, "dtype:", X_eiml.dtype)

if X_soap.shape != X_eiml.shape:
    raise SystemExit(f"Shape mismatch: SOAP {X_soap.shape} vs EIML {X_eiml.shape}")

diff = X_eiml - X_soap

norm_soap = np.linalg.norm(X_soap)
norm_eiml = np.linalg.norm(X_eiml)
norm_diff = np.linalg.norm(diff)

dot = float(np.dot(X_soap.ravel(), X_eiml.ravel()))
cos = dot / (norm_soap * norm_eiml + 1e-30)

print("\n--- Norms ---")
print(f"||SOAP|| = {norm_soap:.8e}")
print(f"||EIML|| = {norm_eiml:.8e}")
print(f"||EIML - SOAP|| = {norm_diff:.8e}")
print(f"Relative diff = {norm_diff/(norm_soap + 1e-30):.8e}")

print("\n--- Similarity ---")
print(f"Cosine(SOAP, EIML) = {cos:.12f}")

print("\n--- Elementwise diff stats (EIML - SOAP) ---")
print(f"min  = {diff.min():.8e}")
print(f"max  = {diff.max():.8e}")
print(f"mean = {diff.mean():.8e}")
print(f"std  = {diff.std():.8e}")

X_soap_n = X_soap / np.linalg.norm(X_soap)
X_eiml_n = X_eiml / np.linalg.norm(X_eiml)
print("Normalized cosine:", np.dot(X_soap_n, X_eiml_n))

# Show top-k absolute changes
k = 20
idx = np.argsort(np.abs(diff))[-k:][::-1]
print(f"\n--- Top {k} |diff| entries ---")
for i in idx:
    print(f"i={int(i):6d}  SOAP={X_soap[i]: .6e}  EIML={X_eiml[i]: .6e}  diff={diff[i]: .6e}")
