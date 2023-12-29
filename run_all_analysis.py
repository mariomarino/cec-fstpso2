import subprocess as sb

for fitness in ["Rastrigin"]:  # "Michalewicz", "Ackley", "Rastrigin", "Griewank"
    for D in [30]:
        for R in range(2, 30):
            print(f"f: {fitness}, {D}D, {R}R")
            try:
                output = sb.check_output(
                    ['sbatch', 'sbatch_rastrigin.sh', fitness, str(D), str(R)])
            except sb.CalledProcessError as e:
                print("return code: {}".format(e.returncode))
                print("output: {}".format(e.output))
