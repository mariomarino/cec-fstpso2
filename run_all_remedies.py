import subprocess as sb

"""remedy_names = (["pso_ring_remedy_1", "pso_ring_remedy_2", "pso_ring_remedy_3", "pso_ring_remedy_4"] +
                ["fstpso_remedy_1", "fstpso_remedy_2", "fstpso_remedy_3", "fstpso_remedy_4", "stufstpso"])"""
"""remedy_names = (["pso_ring_remedy_1a", "pso_ring_remedy_2a", "pso_ring_remedy_3a", "pso_ring_remedy_4a"] +
                ["pso_ring_remedy_1a1", "pso_ring_remedy_2a1", "pso_ring_remedy_3a1", "pso_ring_remedy_4a1", "stufstpso"])"""
remedy_names = ["pso_ring_dilation_1", "pso_ring_dilation_2", "pso_ring_dilation_3", "pso_ring_dilation_4",
                "pso_ring_dilation_5", "pso_ring_dilation_6"]

# ["pso_ring_remedy_1", "pso_ring_remedy_2", "pso_ring_remedy_3", "pso_ring_remedy_4"]

for fitness in ["Rastrigin"]:
    for remedy_name in remedy_names:
        for D in [2, 30]:
            for R in range(30):
                print(f"f: {fitness}, {D}D, {R}R")
                try:
                    output = sb.check_output(
                        ['sbatch', 'sbatch_remedies.sh', fitness, str(D), str(R), str(remedy_name)])
                except sb.CalledProcessError as e:
                    print("return code: {}".format(e.returncode))
                    print("output: {}".format(e.output))
