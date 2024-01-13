import subprocess as sb
import sys
sys.path.insert(0, "code")

ids = [i for i in range(1, 31) if i != 2]

for fitness_id in ids:
    for D in [30]:
        for B in ["4B"]:
            print(f"f: {fitness_id}, {D}D, {B}B")
            try:
                output = sb.check_output(
                    ['sbatch', 'sbatch_plots_cec.sh', str(fitness_id), str(D), B])
            except sb.CalledProcessError as e:
                print("return code: {}".format(e.returncode))
                print("output: {}".format(e.output))
