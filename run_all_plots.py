import subprocess as sb
import sys
sys.path.insert(0, "code")

for fitness in ["Rastrigin"]:
    for D in [2, 30]:
        for B in ["4B"]:
            print(f"f: {fitness}, {D}D, {B}B")
            try:
                output = sb.check_output(
                    ['sbatch', 'sbatch_plots.sh', fitness, str(D), B])
            except sb.CalledProcessError as e:
                print("return code: {}".format(e.returncode))
                print("output: {}".format(e.output))
