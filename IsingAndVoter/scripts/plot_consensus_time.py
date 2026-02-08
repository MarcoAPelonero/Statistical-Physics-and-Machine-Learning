import matplotlib.pyplot as plt
import numpy as np

# Read consensys_sweep_voter.csv
data = np.genfromtxt('consensus_sweep.csv', delimiter=',', names=True)
# Takt the last column as consensus time
consensus_times = data['avg_steps']
target_densities = data['target']
# Overlay the curve from -N\left[\rho\ln(\rho)+(1-\rho)\ln(1-\rho)\right] with N = 400
N = 625
theoretical_times = -N * (target_densities * np.log(target_densities) + (1 - target_densities) * np.log(1 - target_densities))


plt.figure(figsize=(8,6))
plt.plot(target_densities, consensus_times, marker='o')
plt.plot(target_densities, theoretical_times, label='Theoretical Curve', color='red', linestyle='--')
plt.legend()
plt.xlabel('Target Density of +1 Spins')
plt.ylabel('Average Consensus Time (steps)')
plt.title('Consensus Time vs Target Density in Voter Model')
plt.grid(True)
plt.savefig('consensus_time_vs_target_density.png')
plt.show()