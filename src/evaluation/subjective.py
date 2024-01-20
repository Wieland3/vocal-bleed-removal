import numpy as np

gate_quality = [25,90,90,50,90,25,50,60,65]
gate_bleed = [25,20,30,20,10,15,20,15, 40,10,15,5]
exploited_quality = [85, 50, 80, 60, 85, 40, 75, 80, 60]
exploited_bleed = [60,40,50,40,40,40,50,25, 60, 65, 50, 65]
unexploited_quality = [20,10,80,50,75,90, 30, 90, 70]
unexploited_bleed = [60,80,80,50,50,60,75,95, 50, 70, 80, 85]
musdb_quality = [20,20,70,55,60,80, 35, 45, 50]
musdb_bleed = [50,60,45,70,50,40,90,70, 40, 50, 50, 45]
moises_quality = [100,100,100,100,100,100, 95, 95, 100]
moises_bleed = [90,80,100,100,100,95,100,95, 95, 100, 95, 100]
exploited_gate_quality = [70,60,80,60,80,75,60, 65, 65]
exploited_gate_bleed = [70,50,60,55,55,30,70,85, 60, 50, 60, 70]
hidden_reference_quality = [100, 100, 100, 100, 100, 100, 95, 95, 95]
hidden_reference_bleed = [0, 0, 0, 0, 0, 0, 0, 0, 10, 5, 10, 5]

# Gates

gate_30_400_quality = [15,45,60, 15, 20, 80, 30, 45, 40]
gate_30_400_bleed = [30,60,70,100, 10, 30, 60, 100, 60, 65, 65, 100]
gate_40_400_quality = [20,70,80, 50, 60, 75, 50, 40, 40]
gate_40_400_bleed = [30, 65, 50, 100, 70, 60, 50, 95, 45, 70, 60, 75]
exploited_gate_40_400_quality = [25, 65, 80, 35, 80, 75, 50, 60, 85]
exploited_gate_40_400_bleed = [50, 75, 55, 100, 80, 70, 80, 100, 75, 60, 75, 100]

print(np.average(exploited_gate_40_400_quality))
print(np.average(exploited_gate_40_400_bleed))