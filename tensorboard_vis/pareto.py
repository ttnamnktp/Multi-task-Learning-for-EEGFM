# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# points = {
#     'FAMO': (1.181, 0.0038),
#     'LS': (2.103, 0.0109),
#     'OGR': (1.6601, 0.0108)
# }

# x_famo, y_famo = points['FAMO']
# x_ls, y_ls = points['LS']
# x_ogr, y_ogr = points['OGR']

# # Setup plot
# plt.figure(figsize=(8, 6))

# # Plot points
# plt.scatter(x_famo, y_famo, color='red', s=150, zorder=5, label='FAMO (Dominant)')
# plt.scatter(x_ls, y_ls, color='blue', s=100, zorder=5, label='LS')
# plt.scatter(x_ogr, y_ogr, color='green', s=100, zorder=5, label='OGR')

# # Add labels to points
# plt.text(x_famo + 0.03, y_famo, 'FAMO\n(1.181, 0.0038)', fontsize=10, fontweight='bold', va='center')
# plt.text(x_ls - 0.03, y_ls + 0.0003, 'LS\n(2.103, 0.0109)', fontsize=10, ha='right')
# plt.text(x_ogr - 0.03, y_ogr - 0.0005, 'OGR\n(1.6601, 0.0108)', fontsize=10, ha='right')

# # Highlight the Pareto Dominated region by FAMO (assuming minimization)
# # Everything to the right and above FAMO is dominated by FAMO
# xlim_max = 2.4
# ylim_max = 0.013

# # Draw boundary lines for dominance zone
# plt.plot([x_famo, xlim_max], [y_famo, y_famo], 'r--', alpha=0.6)
# plt.plot([x_famo, x_famo], [y_famo, ylim_max], 'r--', alpha=0.6)

# # Fill the dominated region
# plt.fill_between([x_famo, xlim_max], y_famo, ylim_max, color='red', alpha=0.1, label='Region Dominated by FAMO')

# # Graph details
# plt.xlabel('$L_{con}$', fontsize=12)
# plt.ylabel('$L_{rec}$', fontsize=12)
# plt.title('Pareto Front Analysis of Loss Balancing Algorithms for CBraMod', fontsize=14, fontweight='bold', pad=15)
# plt.xlim(1.0, xlim_max)
# plt.ylim(0.002, ylim_max)
# plt.grid(True, linestyle=':', alpha=0.6)
# plt.legend(loc='lower right')

# # Save image
# plt.tight_layout()
# plt.savefig('pareto_dominance.png', dpi=300)
# plt.close()
# print("Successfully generated pareto_dominance.png")

import matplotlib.pyplot as plt

# Data for eeGPT
points_eegpt = {
    'FAMO': (0.2649, 0.7578),
    'LS': (0.3389, 0.7781),
    'OGR': (0.5504, 0.7729)
}

x_famo, y_famo = points_eegpt['FAMO']
x_ls, y_ls = points_eegpt['LS']
x_ogr, y_ogr = points_eegpt['OGR']

# Setup plot
plt.figure(figsize=(8, 6))

# Plot points
plt.scatter(x_famo, y_famo, color='red', s=150, zorder=5, label='FAMO (Dominant)')
plt.scatter(x_ls, y_ls, color='blue', s=100, zorder=5, label='LS')
plt.scatter(x_ogr, y_ogr, color='green', s=100, zorder=5, label='OGR')

# Add labels to points
plt.text(x_famo + 0.01, y_famo - 0.001, 'FAMO\n(0.2649, 0.7578)', fontsize=10, fontweight='bold', va='top')
plt.text(x_ls + 0.01, y_ls, 'LS\n(0.3389, 0.7781)', fontsize=10, va='center')
plt.text(x_ogr - 0.01, y_ogr - 0.001, 'OGR\n(0.5504, 0.7729)', fontsize=10, ha='right', va='top')

# Highlight the Pareto Dominated region by FAMO (minimization)
xlim_max = 0.65
ylim_max = 0.80

# Draw boundary lines for dominance zone
plt.plot([x_famo, xlim_max], [y_famo, y_famo], 'r--', alpha=0.6)
plt.plot([x_famo, x_famo], [y_famo, ylim_max], 'r--', alpha=0.6)

# Fill the dominated region
plt.fill_between([x_famo, xlim_max], y_famo, ylim_max, color='red', alpha=0.1, label='Region Dominated by FAMO')

# Graph details
plt.xlabel('$L_{con}$', fontsize=12)
plt.ylabel('$L_{rec}$', fontsize=12)
plt.title('Pareto Front Analysis of Loss Balancing Algorithms for EEGPT', fontsize=14, fontweight='bold', pad=15)
plt.xlim(0.20, xlim_max)
plt.ylim(0.74, ylim_max)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')

# Save image
plt.tight_layout()
plt.savefig('eegpt_pareto_dominance.png', dpi=300)
plt.close()
print("Successfully generated eegpt_pareto_dominance.png")