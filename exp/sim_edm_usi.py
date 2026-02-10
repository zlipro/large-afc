import numpy as np
import matplotlib.pyplot as plt
import src.utils.geometry as geometry
import src.stress as stress
from scipy.spatial.distance import pdist, squareform
import src.utils.geometry as geometry
from src.utils.utils import stress_normalizer


def visualize_comparison(stress_gen, stress_usi, edm_mat, geometry = None, cmap = 'viridis'):
    """
    Plots Generic Stress, USI Stress, and EDM matrices.
    Stress matrices share the EXACT same color scale and map.
    Diagonals are forced to white.
    """
    # Create copies
    s_gen = stress_gen.copy()
    s_usi = stress_usi.copy()
    edm = edm_mat.copy()
    
    # Force diagonal to NaN (White)
    for mat in [s_gen, s_usi, edm]:
        np.fill_diagonal(mat, np.nan)

    # --- 1. Determine Common Scale for Stress ---
    # We find the min/max across BOTH matrices to lock the colorbar
    # Using nanmin/nanmax ignores the white diagonals
    vmin_stress = min(np.nanmin(s_gen), np.nanmin(s_usi))
    vmax_stress = max(np.nanmax(s_gen), np.nanmax(s_usi))
    
    # Setup Figure (1 Row, 3 Columns)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    # Define Colormaps with 'bad' (NaN) set to White
    # We use 'viridis' for all, or you can switch EDM to 'plasma' if preferred
    cmap_common = plt.get_cmap(cmap).copy()
    cmap_common.set_bad(color='white')

    # 1. Generic Stress (Numerical)
    im1 = axes[0].imshow(s_gen, cmap=cmap_common, interpolation='nearest',
                         vmin=vmin_stress, vmax=vmax_stress)
    axes[0].set_title("Generic")
    # We don't need a colorbar here if we share it, but keeping it for clarity
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. USI Stress (Geometric)
    im2 = axes[1].imshow(s_usi, cmap=cmap_common, interpolation='nearest',
                         vmin=vmin_stress, vmax=vmax_stress)
    axes[1].set_title("USI Method")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. EDM Matrix
    # EDM has a different scale (distances vs stress), so we let it auto-scale
    # But we use the same colormap style
    im3 = axes[2].imshow(edm, cmap=cmap_common, interpolation='nearest')
    axes[2].set_title("EDM Matrix")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    # concatenate geometry name if provided
    output_path = f'results/edm_usi_generic_{geometry}.pdf' if geometry else 'results/stress_comparison_usi_vs_generic.pdf'
    # save as pdf
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f">>> Comparison plot saved to '{output_path}'")
    plt.show()

if __name__ == "__main__":
    # Setup Circular Configuration
    N = 8
    print(f">>> Generating Circular Configuration with N={N}...")
    nominal_config = geometry.circular_config(N)
    
    # Compute EDM (for Visualization)
    # pdist computes pairwise distances, squareform makes it a matrix
    # We want squared distances for standard EDM
    dists = pdist(nominal_config.T)
    edm_mat = squareform(dists ** 2)
    
    print(">>> Computing Stress using Custom-generic...")
    # Using your existing hyperparameters
    hparams = {'alpha': 5.0, 'beta': 1.0, 'gamma': 0.1} 
    stress_gen = stress.optimize_stress(nominal_config, hparams)
    stress_gen = stress_normalizer(stress_gen)

    print(">>> Computing Stress using Custom-USI...")
    stress_usi = stress.optimized_stress_symm_acc(nominal_config, hparams)
    stress_usi = stress_normalizer(stress_usi)

    visualize_comparison(stress_gen, stress_usi, edm_mat, geometry = 'circular', cmap = 'winter')

    # cuboctahedron
    nominal_config_cubo = geometry.cuboctahedron()
    edm_cubo = squareform(pdist(nominal_config_cubo.T) ** 2)
    stress_usi_cubo = stress.optimized_stress_symm_acc(nominal_config_cubo, hparams)
    stress_usi_cubo = stress_normalizer(stress_usi_cubo) 
    stress_gen_cubo = stress.optimize_stress(nominal_config_cubo, hparams)
    stress_gen_cubo = stress_normalizer(stress_gen_cubo)    

    visualize_comparison(stress_gen_cubo, stress_usi_cubo, edm_cubo, geometry = 'cuboctahedron', cmap = 'rainbow')

    # football
    nominal_config_football = geometry.truncated_icosahedron()
    edm_football = squareform(pdist(nominal_config_football.T) ** 2)
    stress_usi_football = stress.optimized_stress_symm_acc(nominal_config_football, hparams)
    stress_usi_football = stress_normalizer(stress_usi_football) 
    stress_gen_football = stress.optimize_stress(nominal_config_football, hparams)
    stress_gen_football = stress_normalizer(stress_gen_football)    

    visualize_comparison(stress_gen_football, stress_usi_football, edm_football, geometry = 'truncated-icosahedron', cmap = 'plasma')

    

    
    