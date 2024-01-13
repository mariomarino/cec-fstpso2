import subprocess as sb

ids = [i for i in range(1, 31) if i != 2]
function_names = [f"CEC17-F{i}" for i in ids]
remedy_names = ["pso_ring_dilation_5", "pso_ring_dilation_6", "stufstpso"]
# remedy_names = ["pso_ring_dilation_6"]

for D in [30]:
    for remedy_name in remedy_names:
        for fitness_id in ids:
            for R in range(30):
                print(f"f: {fitness_id}, {D}D, {R}R")
                try:
                    output = sb.check_output(
                        ['sbatch', 'sbatch_cec.sh', str(fitness_id), str(D), str(R), remedy_name])
                except sb.CalledProcessError as e:
                    print("return code: {}".format(e.returncode))
                    print("output: {}".format(e.output))
