import subprocess as sb

ids = [i for i in range(1, 31) if i != 2]
function_names = [f"CEC17-F{i}" for i in ids]

for D in [30]:
    for fitness_id in ids:
        for R in range(30):
            print(f"f: {fitness_id}, {D}D, {R}R")
            try:
                output = sb.check_output(
                    ['sbatch', 'sbatch_cec_baseline.sh', str(fitness_id), str(D), str(R)])
            except sb.CalledProcessError as e:
                print("return code: {}".format(e.returncode))
                print("output: {}".format(e.output))
